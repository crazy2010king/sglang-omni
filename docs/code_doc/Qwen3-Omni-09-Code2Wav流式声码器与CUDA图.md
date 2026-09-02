# 09 Code2Wav：流式声码器、输出重叠与精确形状 CUDA 图

> 主角：`components/code2wav_scheduler.py`（1065 行）、`components/code2wav_cuda_graph.py`（723 行）、
> `scheduling/streaming_vocoder.py`（模板基类，02 篇已述契约）。

---

## 1. 阶段定位与工厂

`code2wav` 是 terminal 阶段：消费 talker 的 codec 码帧流，产出音频。
工厂 `create_code2wav_scheduler` 的默认参数：

- `stream_chunk_size=10`：每 10 帧码开一窗（24kHz 下 codec 帧率约 12.5/s，
  即约 0.8s 音频/窗；`total_upsample` 由模型给出）；
- `left_context_size=25`：滑窗携带最多 25 帧历史上下文（声码器对长程上下文敏感，
  裁剪输出时只保留新增部分）；
- `sample_rate=24000`、`codec_eos_token_id=2150`；
- `enable_batching=False` 默认关（开了才有跨请求合批/图分解）；
- `enable_output_overlap=True` 默认开；
- `enable_cuda_graph` 默认关，平台开关 `current_platform.enable_code2wav_graph()` 注入
  （03 篇 stage_factory_kwargs）；
- **硬性约束**：开图必须给 `total_gpu_memory_fraction`，否则工厂直接
  `ValueError`——图缓冲的显存预算没有边界就是隐患（09 §5）。

模型加载：HF `Qwen3OmniMoeCode2Wav._from_config(code2wav_config)` +
`load_module(prefix="code2wav.")`，`strict=False`（保留 DSP 缓冲等非权重成员）。

---

## 2. 流状态与 EOS 处理

`Code2WavStreamState`：

```python
chunks: list[Tensor]        # 已入帧（每帧 [num_quantizers] long）
emitted: int                # 已消耗帧游标
audio_parts: list[ndarray]  # 已产出的音频段（拼接即整段）
stream_enabled: bool | None # 由 latch_stream_contract 闩锁（首个流块 metadata["stream"]）
due_since: float | None     # 合批的"到期时间"
checked: int                # EOS lazy scan 的已扫描下标
pending: _PendingWindow | None  # 深度-2 重叠流水线的在途窗
```

### 2.1 EOS 的两种模式

- **eager**（未开重叠）：`ingest` 时逐帧 `codes[0].item() == eos` 判断——每帧一次
  **主机同步**；
- **lazy scan**（`enable_output_overlap and not batching`）：ingest 无条件入队，
  `should_decode` 在攒够一窗时调 `_scan_unchecked` **批量扫描**：

```python
unchecked = chunks[start:]
if all(codes.ndim == 1 for codes in unchecked):     # 生产者恒发 1-D 帧
    heads = torch.stack([codes[0] for codes in unchecked])
    is_eos = (heads == self._codec_eos_token_id).tolist()   # 一次 GPU 求值 + 一次同步
else:
    is_eos = [codes.ndim >= 1 and codes[0].item() == ... for ...]  # 异形回退逐帧
chunks[start:] = [c for c, eos in zip(unchecked, is_eos) if not eos]
state.checked = len(chunks)
```

注释点出设计动机："逐帧检查=每帧一次 sync；批量扫描=每窗一次"。
且**先扫描再判定就绪**：若不扫描，滞留的 EOS 帧会虚增帧数，触发一个短窗
而错失图 key（graph key 按精确帧数命中）。`decode_delta(is_final=True)` 与
`on_stream_done` 也会补扫（流终止不走 should_decode）。

---

## 3. 滑窗解码：decode_delta 的精确数学

```python
start, end = state.emitted, len(state.chunks)
context = min(self._left_context_size, start)
window = torch.stack(state.chunks[start - context : end], dim=0)   # [T, Q]
codes = window.transpose(0, 1).unsqueeze(0)                        # [1, Q, T]
wav, meta = self._forward_codes(codes, graph_eligible=not is_final)
wav = wav[..., -(end - start) * self._total_upsample :]            # 裁掉上下文对应样本
```

- 窗口帧数 = `context + 新帧`；输出裁剪保证**只发新增样本**（增量流）。
- `_forward_codes`：无图 → 直接 `self._model(codes)`；有图 →
  `runner.run(codes, eligible=graph_eligible)`（final 窗永不走图——尾窗帧数
  几乎必然不在 key 集里，标记 ineligible 省一次 key_miss 计数）。
- 返回 `execution_mode / graph_key / fallback_reason` 三元组进 profiler 事件，
  这是判定"图有没有真正命中"的第一手证据。

---

## 4. 输出重叠：深度-2 的 D2H 流水线（本文件最精巧部分）

### 4.1 槽位池

```python
@dataclass(eq=False)                      # 恒等相等：池按"这是几号槽"提问，不能比较张量
class _PinnedSlot:
    buffer: torch.Tensor                  # pinned float32，只增不缩
    event: torch.cuda.Event               # 复用，不逐窗分配

_MAX_PINNED_SLOTS = 32                    # 每流 1 槽 + 1 在途，注释给出推导
_pinned_free / _pinned_created / _pinned_retired / _pinned_quarantined
```

槽位生命周期：`_acquire_slot`（free 优先 → 未达上限新建 → **返回 None 背压**）
→ 在途（`state.pending`）→ `_flush_pending` 后 `_release_slot`；
异常时 `_retire_slot`（可重查回收）或 `_quarantine_slot`（event 不可信，**永不复用**
但仍占名额，防止容量凭空膨胀），quarantine 同时 `_pipeline_active = False`
（整体退回同步路径）。
`_reap_retired_slots` 用**非阻塞** `event.query()` 把已完成 retire 槽收回 free——
在 `_next_message` 循环里调用（wenyao 注释：query 非阻塞，不会复现它所替代的
abort 卡死问题）。

### 4.2 流水线时序（decode_delta 内）

```
第一窗（start==0）：同步路径 —— D2H 立即 .cpu()，保证 TTFA 不被流水线伤害
后续窗（pipeline_active 且非 final）：
  slot = acquire(samples)
  ├─ None 且有 pending → flush 自己的 pending（保序！只 flush 自己的窗）
  │                      再 acquire；再 None → 老实走同步路径
  └─ 拿到 → buffer[:samples].copy_(wav.reshape(-1), non_blocking=True)
            event.record()                        ← 同一 CUDA stream 保证顺序
            flush 上一窗 pending（等待其 event → host 拷贝 → append → 返回波形）
            state.pending = _PendingWindow(slot, samples)
```

正确性三根柱子（代码注释逐一写明）：

1. **同 stream 序**："same stream is what makes this safe —— 拷贝先于任何后续
   replay 排队，借来的静态输出不可能在排空完成前被覆写"（`Code2WavRunResult`
   docstring 亦强调：图输出是借来的，必须在下一次 replay 前完成读或同 stream
   排队读）；
2. **先 launch 后 flush**：`_flush_pending` 放在本次 launch 之后，主机等待
   上一窗的时间与 GPU 计算本窗重叠；
3. **异常槽位处理**：record 成功后异常 → retire（可回收）；record 前异常 →
   quarantine（无可信完成标记，绝不回池）。

`_flush_pending` 的顺序同样讲究：`event.synchronize() → numpy().copy() →`
**之后**才 `state.pending=None; _release_slot(slot)`——任何一步都可能抛，
提前归还等于把一个"活的 D2H 目标"交给池。

---

## 5. 精确形状 CUDA 图（code2wav_cuda_graph.py）

### 5.1 key 体系与捕获计划

`GraphKey(batch_size, frames)`。键集由调度器构造（scheduler.py:26-56）：

- 串行键：`frames ∈ {c, 2c, ..., c+left_context}` 逐步推进（c=stream_chunk_size），
  每步一窗、上下文线性增长到上限；
- 批量键：同帧长 × `batch_size ∈ {2,4,8}`（`_DECOMPOSE_SIZES=(8,4,2,1)`，
  注释：分解表、捕获键、批上限全部由这个 tuple 派生，键不可能缺失）。

`Code2WavCudaGraphRunner` 的构建策略（`_build` / `_capture_attempt`）：

1. **两层**：`batch_size==1` 是"原子层"（`_tier0_keys`）——任何失败=整体禁用，
   绝不发布残缺矩阵；`batch_size>1` 是"尽力层"（`_tier1_keys`）。
2. **共享 mempool**：所有 key 共用一个 `graph_pool_handle`，总占用趋近**最大成员**
   的峰值而非逐成员累加；代价是"池内存只能整体回收"，所以重试单位是**整轮捕获**。
3. **最大优先**：`_priority_order` 按 `(batch_size, frames)` 降序——最大图先铺
   池子的峰值块，后续小图复用。
4. **逐键预算检查**：每捕获一个 tier1 键 → synchronize → `empty_cache()`
   （注释：warmup 的 eager 激活残留会让 reserved 虚高，必须清掉再量）→
   `footprint > graph_budget` 则停手。预算 =
   `total_bytes * fraction - 已加载模型 footprint`。
5. **失败收缩**：`shrink` 回调按"剔除最大 batch 档"（首键即违规/合计违规）或
   "截断到违规位"（中间违规）收敛，最多 `_TIER1_MAX_ATTEMPTS=6`
   （backstop，注释：重试集严格递减，正常情况到不了 6）。
6. **等价性验证**：每键捕获后 `eager_output vs graph.replay()` 逐位
   `torch.equal` + 形状 + 有限性（`_verify_equivalence`），不过即 `_BuildFailure`。
7. **发布前同步**：capture/replay/等价检查都排了 CUDA 工作，
   全部 key 完成并 device synchronize 后才 `_enabled=True`。

### 5.2 运行时

`run(codes, eligible)` 的守卫链：**PID 所有权**（跨进程使用直接 RuntimeError——
图绑定进程，spawn 后必须重建）→ enabled/eligible → 形状/dtype/设备/量化器数校验
→ key 命中 → `static_input.copy_ + graph.replay()`。
运行期 replay 异常 → `_disable_runtime`（清全部图、释放池、empty_cache、
`logger.exception`），之后恒走 eager（fallback_reason=`disabled`）。
`stats()` 输出严格 JSON 安全的构建/运行快照（attempted/published、内存账本、
fallback 计数），工厂启动时打一条 `Code2Wav CUDA graph startup stats=...`。

---

## 6. 跨请求合批（enable_batching=True 时）

三个钩子的实现（scheduler.py:820-1000）：

- **`select_step_participants`**：
  - 首窗流（`emitted==0` 且攒够 `initial_codec_chunk_frames` 或整窗）：
    同 bucket 的首窗流优先出队（注释：同 bucket ⇒ 一个 trim 标量对整批成立），
    截到 `_batch_ceiling`；
  - 稳态流：`ready >= chunk_size` 时记 `due_since`，按 bucket
    `(context, context+step_frames)` 分桶；选**最老等待时间最短的那桶**
    （`min(due, key=最老等待)`），凑满 `batch_floor` 或等满 `max_batch_wait_s`
    或 drain 模式才触发（fire_reason ∈ {first, floor, deadline, drain}）。
- **`build_step_plan`**：图对齐开启时，按该窗长**实际发布**的批大小
  （`runner.available_batch_sizes(window)`，内存压力可能缺大图）把 N 请求
  分解成 `_decompose_batch(N, sizes=(8,4,2,1))` 的子批序列；**没有可用图的
  尺寸时退 `[1]*N`**——注释：批图缺席可能因为其 eager warmup OOM，
  把该批按 eager 重试并不安全，宁可全串行图。
- **`run_step`**：逐子批 `_run_sub_batch`（同窗长断言 → stack 成
  `[B, Q, T]` → 前向 → 按请求切分 → `state.emitted=window_ends[i]`）；
  某子批失败时**只 abort 尚未 decode 的尾部参与者**
  （`participants[cursor:]`）——已 decode 的音频是欠客户端的，不能丢。
  profiler 事件记录 `sub_batch_execution` 列表，混合模式汇总为 `"mixed"`。

---

## 7. 终态与兜底

- `on_stream_done`：先 flush 在途 pending（独立消息发出，注释：不并入尾块，
  消息边界必须与同步路径一致，且必须先于基类的终态 result），
  再走基类（flush 尾窗 → 终 result → 清状态）；
- `final_result_data`：流式时只回 `{modality, sample_rate}`（客户端已收全样本）；
  非流式 `np.concatenate(audio_parts)` 打包整段（`audio_waveform_payload`）；
  **一段音频都没有 → RuntimeError**（绝不返回空音频装成功）；
- `release_stream_resources`（abort 时）：**非阻塞** retire 在途槽
  （abort 在 stage 事件循环线程上跑，不能等 CUDA）；
- `on_serving_stop`：关机时同步排干 retire 槽（此时阻塞无害）。

---

## 8. 小结

1. 帧窗数学：`window = [start-context, end)`，输出裁剪 `[-(end-start)*upsample:]`——
   上下文只影响质量不影响长度。
2. EOS lazy scan 把"每帧一次 sync"降为"每窗一次"；先扫描后判窗的顺序
   是图命中率的前提。
3. 输出重叠 = 槽位池 + 事件 + 同 stream 排序 + 保守降级（首窗同步、槽尽同步、
   异常整体退同步）。
4. CUDA 图的工程学：原子层/尽力层、共享池、最大优先、逐键预算、等价验证、
   运行期一键禁用、PID 所有权——每条都是生产事故的解药。
5. 合批的三层闸门（参与者选择→分解计划→子批执行）保证了
   "图没命中就退串行、子批失败不殃及已完成音频"。

下一篇（10）：文本链的终端——流式反分词器的边界处理。
