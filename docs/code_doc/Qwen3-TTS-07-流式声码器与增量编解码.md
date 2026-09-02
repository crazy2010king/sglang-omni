# Qwen3-TTS 深度解析(七):流式声码器与增量编解码

> 本篇覆盖 `streaming_vocoder.py`(1722 行)与 `incremental_codec.py`(471 行)。这是全模块对并发正确性与 CUDA 上下文安全要求最苛刻的文件:两个异步 worker、优先级流调度、pinned 传输槽、双层 CUDA 图、以及对"同步失败即进程级故障"的极端处理。

---

## 1. 位置与骨架:StreamingVocoderBase 的模板方法

`Qwen3TTSStreamingVocoderScheduler(StreamingVocoderBase[_Qwen3TTSStreamState, None])`。基座(`scheduling/streaming_vocoder.py`)的契约(其模块 docstring 是权威文档):

- 请求级状态注册表:首个 chunk/payload 时 `create_stream_state`,终态清理;
- 流合同锁定:`latch_stream_contract` 在首个流块上锁死 `num_quantizers`/`ref_code_len`/`initial_codec_chunk_frames`;
- 阈值循环:`on_stream_chunk → ingest → should_decode → decode_delta → emit`;
- `on_stream_done` 序列:冲剩余 → 末块 → 终态 result → 清态;
- 无输出时 `fallback_full_decode` 整段兜底;
- abort/stop → `release_stream_resources` → 弹态;迟到的块对已清理请求直接丢弃。

模型侧要实现的钩子:`create_stream_state / latch_stream_contract / validate_chunk / ingest / should_decode / decode_delta / final_result_data / fallback_full_decode`。

时间参数(构造默认,全部可由工厂 kwargs 覆盖):

| 参数 | 默认 | 语义 |
|---|---|---|
| `stream_stride` | 16 帧 | 稳态步长(16 帧 @12Hz ≈ 1.33s 音频) |
| `stream_followup_stride` | 8 帧 | 稳态**触发阈值**增量 |
| `initial_chunk_frames` | 8 帧 | 首块解码长度(TTFA 关键) |
| `stream_left_context_frames` | 16 帧 | 解码窗口的左侧上下文 |
| `initial_max_batch_size / wait` | 32 / 2ms | 首 worker 攒批 |
| `followup_max_batch_size / wait` | 8 / 1ms | 后续 worker 攒批 |

`stream_chunk_ramp` 是两旋钮的泛化替代(互斥,配置校验拒绝混用):`ramp[0]` 为首块,其余为逐次步长;且校验 `ramp[0] <= stream_stride`,注释:"请求期解析器会把首块 clamp 到稳态步长,更大的配置值会静默跑出另一种不可用图形状的调度"。

## 2. 流状态与合同

`_Qwen3TTSStreamState` 全字段即流式会话的账本:

```python
code_chunks: list[Tensor]      # 未解码码块(可裁剪)
total_frames / ref_frames / pruned_frames / emitted_generated_frames   # 四个游标
next_decode_generated_frames   # 下次触发阈值
decoded_chunks                 # 已解码次数(ramp/legacy 分支判据)
num_quantizers                 # 合同:Q
pending_ref_frames             # 首块携带的参考帧数(消费一次)
initial_chunk_frames           # 合同:首块长
initial/followup/final_pending # 三态在途标记(防重复入队)
playback_deadline_s            # 播放截止钟(优先级队列键)
incremental_codec_state        # 增量解码器状态(§6)
incremental_codec_fallback     # 增量失败后降级标记
```

`latch_stream_contract`(`streaming_vocoder.py:313`):

- payload 来源:从 `request.params` 解析 `initial_codec_chunk_frames`;
- metadata 来源(流块):`num_quantizers` 必须存在且**与会话值一致否则 raise**(合同不可变);`ref_code_len` 只允许出现在首个块且不得迟于流开始(`"reference codes arrived after stream start"`);`initial_codec_chunk_frames` 可在流上更新(经 `resolve_initial_codec_chunk_frames` clamp 到稳态步长)。

`validate_chunk`:`[Q]` 或 `[T, Q]` 归一为 [T, Q];Q 与合同比对;**CPU 张量**才做范围检查(`[0, 2048)`)。为什么跳过 CUDA 张量?§4 的答案(设备端断言会毒化上下文,改在解码前统一 clamp+筛查)。

## 3. 解码计划与窗口数学

核心不变量:码流 = `[参考段 ref_frames | 生成段]`,绝对帧号;已发出 `emitted_generated_frames`;窗口取 **`[max(0, 已发绝对帧 - left_context), ref + generated)`**:

```python
def _build_decode_plan(state, *, is_final, max_generated_frames=None):
    available = state.total_frames - state.ref_frames
    if available <= state.emitted: return None                       # 无新帧
    next_frames = self._next_decode_threshold(state)
    if not is_final and available < next_frames:
        state.next_decode_generated_frames = next_frames; return None  # 未达阈值
    generated = min(available, max_generated_frames or inf)
    absolute_emitted = state.ref_frames + state.emitted
    window_start = max(0, absolute_emitted - self._stream_left_context_frames)
    window_end   = state.ref_frames + generated
    while code_chunks and pruned + len(chunk0) <= window_start:      # 整块裁剪
        state.pruned_frames += len(state.code_chunks.pop(0))
    codes = torch.cat(state.code_chunks, dim=0)
    decoder_input = codes[window_start - pruned : window_end - pruned]
                    .transpose(0, 1).unsqueeze(0)                    # → [1, Q, T]
    return _Qwen3TTSDecodePlan(...)
```

- **left context 的必要性**:codec 解码器的因果卷积/transformer 有感受野;只解码新帧会用错历史。经典流式方案 = 携带窗口重解码 + 裁掉重叠输出(`_extract_delta` 按 `absolute_emitted - window_start` 帧数 trim 掉头部)。
- **整块裁剪**的注释:"window_start 只前进,后面的帧是死帧;整块 pop 保持 cat O(窗口) 而非每解码 O(流长)"——`pruned_frames` 是绝对坐标到裁剪坐标的换算偏移。
- `should_decode`:非 final 时 `generated_frames >= 初始阈值或 next_decode_generated_frames`。

`_commit_decode_plan`:提交时校验顺序(`emitted == plan.emitted`,乱序即 RuntimeError)、更新 `emitted_generated_frames`、`decoded_chunks++`、计算下次阈值(`plan.generated + _next_followup_stride`)、并推进**播放截止钟**:

```python
duration_s = len(delta) / sample_rate
state.playback_deadline_s = max(state.playback_deadline_s, now) + duration_s
```

**播放截止钟 = 该请求音频播完的时刻**。它是后续 followup 优先级队列的键:欠账越多(播放截止越近),解码越优先——这是把"实时播放"硬实时约束映射进调度优先级的一手设计。

`_next_followup_stride` 两种模式:legacy(首次 8、其后 16)或 ramp(按 emitted 帧数在 ramp 累积表上定位,注释:"backlog 冲过头会回到稳态步长")。

## 4. 异步解码:双 worker + 优先级流 + pinned 传输

### 4.1 两级 worker

`on_serving_start` 启动两个守护线程:

- **initial worker**(`_initial_queue`, FIFO):每个请求的**第一次**解码。`_run_initial_batch` 在 `_state_lock` 内构建计划(且限 `max_generated_frames=initial_chunk_frames`,防 backlog 冲爆首块),按**解码器输入形状分组**(`_group_decode_plans`),组内拼批解码。提交后 commit(`_commit_initial`):发流块、视剩余决定转 followup 或收尾。
- **followup worker**(`_followup_queue`, **PriorityQueue 键=(playback_deadline_s, seq)**):后续解码,优先级=播放截止钟(同键按入队序稳定)。

两级分离的动机:**首解码决定 TTFA,必须最优先且批容忍度大**(32/2ms);后续解码按截止钟公平排序(8/1ms)。`_schedule_initial/_schedule_followup` 的 `pending` 标记防重复入队;`_collect_async_batch` 是标准的攒批循环(阻塞取首 + 截止窗内凑批)。

失败隔离:`_decode_group` 捕获 `_Qwen3TTSInvalidCodeRows`,**只淘汰坏行所在请求**(`bad` 集合过滤后重试),其余请求不受牵连;其它异常则整组失败。`_commit_*` 里 commit 失败 → `_emit_error` + abort + 清理。

### 4.2 解码流与 CUDA 图

两条专用低优先级 CUDA stream(`priority_range()` 的最次优先级+1):`_decode_stream`(initial)与 `_followup_decode_stream`。**低优先级是刻意设计**:声码器与 AR 引擎共卡时,codec 解码要让路给骨干前向——流优先级是 GPU 硬件级的抢占顺序。

双层 CUDA 图(`_Qwen3TTSInitialDecodeGraphs`,同一个类服务两处):

```python
initial 图:input_frames = left_context + initial_chunk(= 16+8 = 24)
followup 图:input_frames ∈ {left+ramp 各步, left+steady}(= 24/24/…/16+16)
batch 桶:(1,2,4,8);确定性模式只 (1,)
```

`capture()`:`(frames, batch)` 全组合,边流 warmup×2 → capture stream 捕获,**共享 graph_pool**;失败仅跳过该组合(warning)。

`decode(codes)`:

```python
codes [B, Q, T]:T ∈ 图输入帧集合 且 Q 匹配 → 找 ≥B 的桶
static_input.zero_(); [:B].copy_(codes); replay; return outputs[:B].clone()
```

窗口形状不在图集合时(短参考/早期流被截断,见 §1 注释)→ 图返回 None → eager `chunked_decode`。

### 4.3 pinned 传输槽:`_DecodeSlot` 与生命周期状态机

每线程一个槽(`threading.local`,`_thread_decode_slot`),由 `GrowablePinnedBuffer`(输入码)+ `PinnedTransferSlot`(输出音频,自带复用 Event)构成。数据流(类 docstring 原文):

```
CPU codes [B,Q,T] → input_codes(pinned) → CUDA decoder
CUDA 音频增量 [S] → output_transfer(pinned + event) → 独立 CPU 张量(非 pinned)
```

`busy`(在途)/`broken`(粘性,永不再用)两态。`_launch_async` 的关键序列:

```python
stream.wait_stream(current_stream)          # 与当前流建立依赖
with torch.cuda.stream(stream):
    gpu_input = stage_decoder_input(...)     # H2D(pinned 非阻塞)
    waveform = graphs.decode(gpu_input) or chunked_decode(gpu_input)
    deltas = [extract_delta(...)]
    staged = stage_deltas(deltas, slot)      # D2H 写 pinned 视图
    slot.output_transfer.record(stream)      # 事件即使全空也记录——注释:"事件同时 fence H2D 与解码"
return _Qwen3TTSDecodeHandle(staged, bad_rows, slot=slot, ...)
```

**零长度 `.cpu()` 陷阱**(非 pinned fallback 分支注释):"零元素 `.cpu()` 不会入队任何 D2H,也就不会等待前面的解码——所以必须显式 `stream.synchronize()`"。这类细节是异步正确性的常见翻车点。

`resolve()`(handle 的终态方法):

1. 等待事件(event 失败 → 退而 `stream.synchronize()`);
2. 事件失败但流成功:丢视图、槽标记 broken(事件永不再信)、释放、**仍然 raise**(本批音频作废,但资源回收);
3. **事件与流都失败:进程级故障**。`_CONTEXT_FATAL_RETAINED` 全局列表**永久持有**该解码触碰的一切(槽、输入、输出、keepalives),`owner._cuda_decode_failed=True` 永久禁用 CUDA 解码。注释逻辑链:"事件和流都无法同步 ⇒ 无法证明 GPU 已用完这些缓冲 ⇒ 分配器可能把它们交给后续工作 ⇒ 唯一安全做法是让它们活到进程结束"。这是一条**宁可停服也不产出未定义音频**的底线设计。
4. 成功:`[delta.clone() for delta in deltas]`——把 pinned 视图物化为独立 CPU 张量再还槽(clone 的必要性同 06 篇快照)。
5. `resolve` 幂等:重复调用返回缓存结果或重抛同错,不再触碰 event/槽。

`_screen_out_of_range_codes`(`streaming_vocoder.py:628`)值得整段理解:

```python
bad_rows = ((decoder_input < 0) | (decoder_input >= 2048)).flatten(1).any(1)
decoder_input.clamp_(0, 2047)
```

注释:越界 id 会让 embedding lookup 触发**设备端断言,毒化 CUDA 上下文并杀死本进程所有在途流**;`validate_chunk` 拦不住设备张量,故统一 clamp 保 lookup 安全,同时返回 per-row 判定——CPU/确定性路径在解码前 raise,异步 CUDA 路径在 `resolve()`(事件后)读取判定再对坏行 raise,**不为筛查添加任何同步点**。被 clamp 的行按失败处理,永不发声。

### 4.4 出口整形

`_decode_and_emit`(同步路径)有一个补偿逻辑:首解码若冲掉积压(如客户端慢,阈值前攒了很多帧),**把 delta 按 ramp 切成多块发出**,保持客户端可见块形的 ramp 形状——`split_frames = (initial,) + ramp[...]`。注释:"without a ramp only the legacy initial-boundary split applies"。这是流式 API 的"块形稳定性"细节:音频总量不变,但块序列的形状影响客户端的播放平滑度。

`_store_vocoder_result`(非流式终态)按 `ref_code_len / total_frames` 的**帧数比例**裁掉波形头部参考段——比例裁剪而非按 samples_per_frame 精确裁,因为整段解码器的上采样不一定整除,注释以 `int(ref/total × len)` 实现,音质无感但边界对齐到可用长度。

## 5. 确定性模式在声码器的落点

`enable_deterministic_inference=True` 时(02 篇 §1.3):

- `_launch_decode_plans` 对多计划**逐个 B=1 重解**:注释引用 issue #1475——批内其它请求影响 kernel 的归约顺序,输出不确定;逐个解则输出与批组成无关。校验合并批(坏行索引仍指原组)后循环单发。
- CUDA 图 initial/followup 桶退化为 `(1,)`。
- `_vocode_payloads` 的非流式整段解码也走逐请求循环。

## 6. `incremental_codec.py`:有状态流式解码器(免窗口重解)

窗口重解(left-context 方案)每步重复计算重叠帧;`enable_stateful_codec_decoder=True` 时改走**真增量解码**:解码器内部状态(卷积历史 + transformer KV)跨步保持,每步只算新帧。

### 6.1 状态结构

```python
@dataclass
class Qwen3TTSIncrementalCodecState:
    frame_position: int                    # 已消费绝对帧
    transformer_context_length: int        # pre_transformer KV 有效长
    transformer_keys/values: dict[layer→Tensor]
    conv_histories: dict[key→Tensor]       # 每个因果 Conv1d 的左历史
    transconv_overlaps: dict[key→Tensor]   # 每个转置卷积的右重叠
    def clone(self): ...                   # 深拷贝所有张量
```

### 6.2 三类原语

**增量因果卷积**(`incremental_causal_conv1d`):`cat(历史, 新帧)` → 正常 conv → 输出新帧长 → **历史 = 合并尾 padding 长度的 clone**。要求 stride=1(转置卷积另算)。

**增量转置卷积**(`incremental_causal_transconv1d`):上采样块的核心难题。`conv_transpose1d(新帧)` 的输出含"上一帧的右重叠"区:`overlap` 存在则 `output[..., :overlap_len] += overlap`(跨块求和,因为转置卷积的重叠贡献是加性的);输出分为 `emitted(新帧数×stride)` 与 `tail(== right_pad,校验)`;`tail` 存为下步 overlap;bias 单独加到 emitted。**数学**:标准转置卷积对分块输入不是线性的(邻块贡献重叠区),手动维护重叠和才能与整段解码逐位一致。

**增量 transformer**(`_incremental_transformer`):标准 KV 缓存 + 滑窗:

```python
query_pos = arange(frame_position, +fresh)
key_pos   = arange(frame_position - ctx_len, +fresh)
allowed   = key_pos <= query_pos;  滑窗: & key_pos > query_pos - sliding_window
retained  = max(0, window_size - 1)
next_keys[layer] = key[..., -retained:, :].clone()      # 逐层裁剪保留
state.transformer_context_length = min(retained, ctx_len + fresh)
```

注意 attention 内显式构造 mask(而非 SDPA 因果标志),因为 key 序列起点是全局坐标偏移过的。

### 6.3 结构守护:`_require_attrs`

构造时对解码器逐模块断言字段存在(pre_conv/pre_transformer/upsample 每块/decoder 每块……),**布局不符立刻 TypeError**——一个防 qwen-tts 升级悄悄改模块结构、增量路径静默走错的前置闸。`decode()` 末尾校验 `state.frame_position == 期望终点`,任何漂移即 RuntimeError(上游 `decode_delta` 捕获后 `incremental_codec_fallback=True`,**该请求其余部分永久降级到窗口方案**,注释 "using the legacy left-context decoder for the rest of the request")。

`decode_delta` 的增量分支还有两道一致性断言:`consumed_frames == ref + emitted`(位置账本对齐)与 `consumed >= pruned`(裁剪不超前)。增量路径输出按 `reference_frames × samples_per_frame` 精确裁参考段——比窗口路径的整段解码裁剪更精确(7bit 对齐)。

## 7. 与 Omni `Code2WavScheduler` 的对照(同为 StreamingVocoderBase 实现)

| 维度 | Qwen3-TTS 流式声码器 | Omni Code2Wav |
|---|---|---|
| 状态 | `_Qwen3TTSStreamState`(游标 + 增量 codec 态) | `Code2WavStreamState`(游标 + unchecked 扫描) |
| 触发模型 | `should_decode` 阈值(初始 8/稳态 16/阈值增量 8/ramp) | `_ready` 扫描 + `_step_frames` + serial threshold(上下文按 chunk 增长至 cap) |
| 图键 | `(frames 固定集合, batch 1/2/4/8)` ×initial/followup 两层 | `GraphKey(batch, frames)`,帧序列=按 chunk 递增至 `chunk+left_context`;批组合 `_DECOMPOSE_SIZES=(8,4,2,1)` **分解**(B=6→4+2) |
| 批组策略 | 双 worker 两种批参数 + playback deadline 优先级 | 单线程 inbox 泵 + `_batch_deadline` + 形状分组分解 |
| D2H | 每线程 pinned 槽 + 双层故障保留 | `_PinnedSlot` 池(acquire/release/quarantine/reap 四态,事件复用) |
| 输出重叠 | pinned slot + record/resolver | 输出重叠开(1024 行处 `output_overlap`)+ chunk-aligned dispatch |
| 增量解码器 | 独立实现(卷积历史+转置重叠+滑窗 KV) | 无(HF Code2Wav 窗口解码) |
| 特殊保障 | clamp+bad_rows 筛查;`_CONTEXT_FATAL_RETAINED`;全批禁用 | slot 隔离区与回收;`gpu_memory_fraction=0.02` 显式预算(图缓冲预算来自 `total_gpu_memory_fraction`) |

两者共有的深层模式:**"固定形状图 + 死行清零 + pinned 暂存 + 事件物化 + 槽故障粘性"**——同一条流式 GPU 服务蓝图在两个模型上的两份实现,差异只在触发策略(TTS 的实时播放优先级 vs Omni 的 chunk 对齐)与解码器内部状态(TTS 有增量 codec,Omni 没有)。

---

## 8. 本篇要点回顾

1. 窗口解码 = left-context 重解 + trim;增量解码 = 卷积历史 + 转置重叠 + 滑窗 KV,两者互为降级。
2. 双 worker(首块 TTFA / 后续截止钟优先级)把实时播放约束编入调度。
3. pinned 槽 + 事件是 D2H 异步化的最小完整方案;`resolve()` 的故障矩阵(事件失败→流同步→进程级保留)定义了"什么时候可以继续服务"的边界。
4. clamp-then-check 模式:先保 CUDA 上下文活着,再按行问责。
5. 确定性模式的全链路代价:预处理串行 + AR 确定性 + 声码器 B=1 逐个解。

最后一篇把两个模型放在一起做总账式对比。
