# Qwen3-TTS 深度解析(六):ModelRunner 执行流水线

> 本篇覆盖 `model_runner.py`(370 行)及其底座 `model_runner/base.py`(1007 行)。这是"调度器输出 → GPU 前向 → 请求状态更新"的缝合层,也是文档 `qwen3-tts.md` 与当前代码漂移最大的一篇。

---

## 1. 底座协议:`ModelRunner` 的钩子面

base.py 定义了所有 AR 模型的执行骨架,先列全钩子再讲 TTS 怎么填:

```
execute(scheduler_output)                     # 同步全流程
  ├─ _build_forward_batch                     # ForwardBatch.init_new(+ capture_hidden_mode)
  ├─ _prepare_and_forward
  │    ├─ before_prefill / before_decode      ← 子类钩子
  │    ├─ custom_prefill_forward / custom_decode_forward(返回 None 则走 tp_worker 标准前向)
  │    ├─ sample_before_post_* → _sample_next_token_ids(可选提前采样)
  │    └─ finally: clear_omni_prefill_inputs + cleanup_prefill
  ├─ post_prefill / post_decode               ← 子类钩子
  ├─ _ensure_next_token_ids / _publish_next_tokens   # 令牌轨道发布(下一步前向的 input_ids)
  └─ _finalize                                 # output_processor.process → generation_steps++ → ModelRunnerOutput

execute_launch(scheduler_output) → _PendingStep    # 异步前瞻:GPU 侧
execute_resolve(pending) → ModelRunnerOutput       # 异步前瞻:host 侧
```

异步前瞻的关键不变量(base.py `_PendingStep` docstring):**同一时刻至多一个 pending step**;launch 采样并发布下一步 token 轨道,record event;resolve 等 event、读 launch_buf、跑收集循环。`lookahead_eligible` 默认把带历史依赖采样参数(重复/频率/存在惩罚、min_new_tokens、自定义 logit processor)的批关在同步路径——**因为 launch 的采样比 resolve 的 append 早一步,读到的 output_ids 会滞后一个 token**。TTS 没有覆写它:层0 采样依赖 SGLang penalizer(历史敏感),理应被 gate;但注意 `generation_defaults` 里 `disable_overlap_schedule=True` 已在调度器层面禁用了 overlap,这条 gate 是第二重保险。

底座还内置两个通用 logit 处理(与 TTS runner 的覆写形成显式对照):

- `_apply_repetition_penalty`(base):host 侧增量维护 `unique(output_ids)` 集合(`_rep_penalty_unique_tokens` 的前缀折叠优化,注释点明"每步重建 unique 是二次方"),构造 (row, token, penalty) 索引张量打惩罚。**TTS 覆写为 no-op**(§4)。
- `_apply_codec_suppress_tokens`(base):按请求的 `suppress_tokens` 内容哈希缓存设备张量,按行组打 -inf。**TTS 覆写为固定区间版**(§4)。

## 2. TTS runner 的钩子填空总表

| 钩子 | Qwen3TTSModelRunner 的实现 | 一句话 |
|---|---|---|
| `before_prefill` | `prepare_decode_buffers` + `attach_omni_prefill_inputs(input_embeds=_build_prefill_input_embeds)` | 预填 embedding 注入 |
| `before_decode` | `prepare_decode_buffers` + `_write_feedback_buffers` | 行号寻址输入 + 采样参数暂存 |
| `post_prefill` / `post_decode` | `_collect_codes` | 层0码 + hidden → 预测器 |
| `sample_before_post_*` | True(两个阶段都提前采样) | 让 `_collect_codes` 拿得到 next_token_ids |
| `_sample_next_token_ids` | 先装 semantic seed 再 super() | §5 篇 |
| `_apply_repetition_penalty` | **del(空操作)** | 交给 SGLang penalizer |
| `_apply_codec_suppress_tokens` | 尾 1024 固定区间抑制 | 保留 codec_eos |
| `post_process_outputs` | 快照 → output_codes/反馈队列 | 每步的账本 |
| `_execution_context` | extend 时先恢复惩罚历史 | retraction 修复 |

## 3. Prefill 路径:embedding 切片回填

`_build_prefill_input_embeds`(`model_runner.py:296`):

```python
for sched_req in requests:
    req = data.req
    req_len = int(req.extend_range.length)          # 本段 extend 长度
    prefix_len = len(req.prefix_indices)            # radix cache 命中的前缀(可 >0!)
    if data.prefill_input_embeds is None:
        data.prefill_input_embeds = data.prompt_input_embeds   # 首次:全量 prompt
    piece = QwenTalkerModelRunner._projected_prefill_slice(
        sched_req=sched_req, prefix_len=prefix_len, extend_len=req_len, device=...)
    if piece is None or piece.shape[0] != req_len:
        raise RuntimeError(f"prefill embed mismatch ... have {have} rows, need {req_len}")
    pieces.append(piece)
return torch.cat(pieces).to(device, dtype)
```

这就是 02 篇"伪 token id"体系的另一半:**radix cache 命中前缀时,只有余下 extend 段需要前向,embedding 必须按 `[prefix_len : prefix_len+extend_len]` 精确切出**。`_projected_prefill_slice`(复用 omni talker runner)还处理 retraction 重放:`prefill_input_embeds` 不够长时,从 `_decode_input_history`(decode 输入的历史账本)补齐生成段 embedding。`attach_omni_prefill_inputs` 把张量挂到 forward_batch 的 sidecar,SGLang 前向读到 `input_embeds` 时走 embedding 注入而非查表(`req._input_embeds_are_projected=True` 告诉框架勿再投影)。

**对比 Omni thinker 的 prefill 注入**:`thinker_model_runner.py`(324 行)在 prefill 时识别图像/视频/音频占位 token 位置并替换为编码器 embedding——Omni 是"token 序列 + 局部占位替换",TTS 是"全序列 embedding"。前者保持 radix cache 逐 token 语义,后者用哈希伪 id 重建前缀语义。两个模型对"embedding prefill 与 KV cache 复用如何共存"给出了两个方向的答案。

## 4. Decode 路径:三个缓冲写手

### 4.1 `_write_feedback_buffers`(行号寻址的写入侧)

```python
decode_feedback_embedding = self.model._decode_feedback_embedding
input_ids = forward_batch.input_ids
# 断言:input_ids 至少 B 个;B 不超过嵌入行数
row_ids = self._decode_row_ids(batch_size, input_ids)       # arange 缓存(≥64 复用)
for row_idx, sched_req in enumerate(requests):
    combined = QwenTalkerModelRunner._take_next_decode_input_embed(sched_req=..., device, dtype)
    if combined is None:                                     # 无反馈且文本队列空(非正常)
        token_id = input_ids[row_idx:row_idx+1]
        combined = get_input_embeddings()(token_id)          # 兜底:codec 查表
    _append_decode_input_history(sched_req.data, combined)   # retraction 重放账本
    rows.append(combined)
torch.stack(rows, dim=0, out=decode_feedback_embedding.weight[:B])   # 一次栈写
input_ids[:batch_size].copy_(row_ids)                         # input_ids 变行号!
```

四个精度点:

1. **`torch.stack(out=...)` 直写权重行**——`_decode_feedback_embedding.weight` 就是写目标,无中间分配。
2. **`input_ids` 在 decode 期承载行号**(03 篇通道②),`copy_` 是 in-place——CUDA 图里 `embedding(input_ids)` 读到的每步内容都由这一行决定。
3. **`combined = 反馈.peek() + 文本.popleft()`** 的合成在 host 循环里逐行做(小张量 GPU 运算),失败兜底查 codec embedding(理论上不该发生,但留着比崩溃好——注意这行兜底 Omni 没有,Omni 直接 raise)。
4. **`_append_decode_input_history`**:每个用于前向的输入都记账,retraction 时 `_generated_prefill_slice` 用它重放生成段——TTS 的 decode 输入是 embedding,回收重算时没有 token 可重建 embedding,**账本就是重放机制本身**。

### 4.2 `prepare_decode_buffers` 的调用时序

TTS 在 `before_prefill` 与 `before_decode` **都**调用它(03 篇 §5 已拆内部)。prefill 时暂存采样参数的意义:prefill 采出的首 token(经 `_collect_codes` → 预测器)同样需要子说话人采样参数。Omni talker 的对应函数在 forward 内部还有 `invalidate_decode_buffers`(extend 后强制全量重暂存,因为 prefill 采样绕过了 `_sampled_token_ids` 缓冲),TTS 用 `(rid, epoch)` 幂等键达到同一效果。

### 4.3 `_collect_codes`:一步的产出

```python
layer0_codes = result.next_token_ids([B]) → unsqueeze
hidden = result.logits_output.hidden_states([B,1,H] 规整)
semantic_positions = self._sample_positions(forward_batch, device)   # decode: positions;prefill: seq_lens-1
self.model.code_predictor_forward(layer0_codes, hidden, semantic_positions=...)   # 04 篇
self._stage_token_ids(result, result.next_token_ids)    # pinned ping-pong 预暂存(见下)
self._has_pending_code_step = True
```

`_stage_token_ids`(base 提供):next_token_ids 异步拷进 pinned 双缓冲并 record event——**后续 `.tolist()` 等的是 event 而不是阻塞的页锁定拷贝**(base.py 注释:"decode 循环里的 c32 热点")。这是 runner 层能做的最后一段异步化。

## 5. `post_process_outputs`:快照、账本、流块

```python
eos_id = codec_eos_token_id
codes_snap = self.model._output_codes[:B].detach().clone()    # 一次性批克隆!
embeds_snap = self.model._output_embeds[:B].detach().clone()
for row_idx, sched_req in enumerate(scheduler_output.requests):
    req_output = outputs[sched_req.request_id]
    if req_output.data is None or int(req_output.data) == eos_id:  continue
    code_chunk = codes_snap[row_idx]
    sched_req.data.output_codes.append(code_chunk)             # 码本账本(终态汇总用)
    sched_req.data.latest_stream_code_chunk = code_chunk       # 流式出口(02 篇 §5.3)
    sched_req.data.pending_feedback_queue.append(embeds_snap[row_idx])  # 反馈账本
```

**为什么必须 clone?** 注释给出性能+正确性双重理由:"per-row clones 曾是 c32 解码环热点;行必须是快照的视图,绝不能是复用图缓冲区的视图"。`_output_codes/_output_embeds` 是**持久缓冲**(CUDA 图的固定地址),下一步前向原地覆写;若把视图直接 append 进 Python 账本,账本里所有"历史"都别名同一块显存——数据错乱。`[:B].detach().clone()` 一次分配搞定全批,clone 的所有权随后交给账本。eos 行跳过:codec_eos 是音频终点,该帧不计入码流。

**对比 Omni 同名函数**(`QwenTalkerModelRunner.post_decode`):Omni 的 post_decode 从 `model._sampled_token_ids`(图内采样缓冲)读回 next_token_ids,然后**直接往 outbox 发 stream 消息**给 code2wav,同样往 `pending_feedback_queue` 快照反馈。TTS 的差异是流消息不在这里发——`stream_output_builder` 由调度器在 output processor 阶段调用(职责分离:runner 管模型状态,适配器管消息),且首块拼 ref 码的协议在适配器里。**两份代码同一血脉,消息发送的层次不同,源于 TTS 的流协议更复杂(ref 前置 + num_quantizers 合同)。**

## 6. 重复惩罚的现状:与旧文档的对齐说明

旧文档 `qwen3-tts.md` §2.4 描述的 `_shape_masks`/`_mask_fingerprint` **在当前代码中不存在**。当前设计(以及为什么更优):

```python
def _apply_repetition_penalty(self, logits_output, requests):
    """Leave repetition-penalty ownership to SGLang's sampling state.
    ScheduleBatch.prepare_for_decode accumulates committed output tokens
    in SGLang's device-resident repetition penalizer. ModelRunner.sample
    applies that state once using the public SamplingParams value."""
    del logits_output, requests
```

SGLang 的 `BatchedRepetitionPenalizer` 在 `prepare_for_decode` 里**增量**累积已提交 token(base.py 的 `_rep_penalty_unique_tokens` 那套 host 端前缀折叠正是它的 Python 对照物),sample 时一次应用。TTS 的层0 采样走 SGLang sampler,惩罚状态机自然归它——runner 不再自建设备掩码。**代价是 retraction**:SGLang 的 `prepare_for_extend` 会为重新入队的请求建全新 sampling state,而 TTS 的 retraction 保留语义 `output_ids`(因为输入是 embedding,重放需要它们,03 篇 §4.6 账本)。于是:

```python
@staticmethod
def _restore_repetition_penalty_history(schedule_batch):
    """Seed a fresh SGLang penalizer before a request is re-prefilled. ..."""
    orchestrator = schedule_batch.sampling_info.penalizer_orchestrator
    penalizer = orchestrator.penalizers.get(BatchedRepetitionPenalizer)
    if penalizer is None or not penalizer.is_prepared(): return
    for row, req in enumerate(schedule_batch.reqs):
        retained_ids = {int(v) for v in req.output_ids if 0 <= v < vocab_size}
        rows/tokens/penalties 收集
    scaling_penalties[row_indices, token_indices] = penalty_values    # 批量直写
```

`_execution_context` 覆写:

```python
if schedule_batch.forward_mode.is_extend():
    self._restore_repetition_penalty_history(schedule_batch)
return super()._execution_context(schedule_batch, isolate_sampling=isolate_sampling)
```

——只在 extend(re-prefill)瞬间把保留 token 的惩罚一次性种回 penalizer 的 `scaling_penalties`,后续 decode 恢复正常每步累积。**这是"绕过框架状态机复用其能力"的完整样本**:不是重新实现 penalizer,而是在它的数据结构上做一次迁移。

抑制 token(`_apply_codec_suppress_tokens`)则仍然是 runner 职责——SGLang 没有对应的原生机制:

```python
configured_vocab = model.config.vocab_size
suppress_start = max(0, configured_vocab - 1024)      # 词表末 1024 个保留给 codec 控制
if suppress_start <= codec_eos < suppress_stop:
    logits[:, suppress_start:codec_eos] = -inf
    logits[:, codec_eos+1:suppress_stop] = -inf        # 只放行 codec_eos
else:
    logits[:, suppress_start:suppress_stop] = -inf
```

与 base 版(任意 token 集 + 内容哈希缓存)不同,TTS 是**固定区间**——无需缓存键,直接切片写。区间语义:词表尾部 1024 个 id 是 codec 控制符空间,生成期除 EOS 外全部禁出。

## 7. 每步生命周期的完整时序(收拢)

```
scheduler_output(step N)
  │ before_decode: prepare_decode_buffers(幂等)→ _write_feedback_buffers(行号+反馈+文本)
  │ forward(Qwen3TTSTalker,图或 eager)→ codec_head → logits
  │ _sample_next_token_ids:装 semantic seed → SGLang sample(penalizer 已在 prepare_for_decode 累积)
  │ _collect_codes:层0码+hidden → 预测器(04 篇)→ _output_codes/_output_embeds
  │ _stage_token_ids:pinned 预暂存
  │ _publish_next_tokens:下一步前向的 token 轨道(execution bridge)
  │ _finalize:output_processor.process(等 event 读 host ids)→ post_process_outputs:
  │      快照 clone → output_codes.append / latest_stream_code_chunk / pending_feedback_queue
  │      generation_steps++ → 引擎层判定 eos/length → result 或继续
  ▼
stream_output_builder → vocoder(07 篇);终态 result_adapter → 汇总 codes
```

## 8. 与 Omni talker runner 的差异小结(呼应 03/05 篇)

| | Qwen3TTSModelRunner | QwenTalkerModelRunner(Omni) |
|---|---|---|
| decode 就绪门 | 无(输入总能合成,兜底查表) | `is_decode_batch_ready` + `before_decode` 硬 raise(必须等 thinker 喂文本) |
| post_decode 的 next_token_ids | SGLang sampler 产出 | `model._sampled_token_ids` 图内采样回读 |
| 流消息 | 不发(适配器发) | runner 直发 outbox(含 `metadata={"stream": is_streaming}`) |
| 重复惩罚 | SGLang penalizer + retraction 恢复 | 模型内掩码暂存(`prepare_decode_buffers` 的 rep_rows 增量置位) |
| prefill 清理 | 同 omni:不清 `prefill_input_embeds`(注释:retract 会重新入队) | 同左 |
| 复用关系 | 复用 QwenTalkerModelRunner 的 4 个静态方法 | 被复用者 |

下一篇进入数据平面的最后一站:流式声码器与增量编解码——`streaming_vocoder.py` 1722 行的并发与显存安全设计。
