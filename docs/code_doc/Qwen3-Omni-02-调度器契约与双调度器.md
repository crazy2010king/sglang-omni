# 02 调度器契约与双调度器：SimpleScheduler / OmniScheduler / QwenTalkerScheduler

> 主角：`sglang_omni/scheduling/simple_scheduler.py`（327 行）、
> `sglang_omni/scheduling/omni_scheduler.py`（2677 行）、
> `sglang_omni/models/qwen3_omni/talker_scheduler.py`（158 行）、
> `sglang_omni/scheduling/streaming_vocoder.py`（Code2Wav 的模板基类）。

---

## 1. 唯一契约：inbox / outbox

框架对 scheduler 的全部要求（`Stage` 只依赖这些）：

```python
scheduler.inbox:  queue.Queue[IncomingMessage]   # new_request / stream_chunk / stream_done
scheduler.outbox: queue.Queue[OutgoingMessage]   # result / stream / error
scheduler.start()          # 阻塞当前线程（Stage 把它放进 daemon 线程）
scheduler.stop()
scheduler.abort(request_id)
scheduler.admin(action, payload) -> dict | AdminResult   # 可选
scheduler.requires_tp_work_fanout: bool                  # 可选
scheduler.enqueue(msg)                                   # 可选：OmniScheduler 的准入入口
```

`IncomingMessage.type` 只有三种与数据有关：`new_request`（整包 payload）、
`stream_chunk`（`data` 是 `StreamItem(chunk_id, data, from_stage, metadata)`）、
`stream_done`。**模型代码永远不碰 ZMQ/relay**——这就是"Stage = IO 壳"换来的解耦。

---

## 2. SimpleScheduler：非 AR 阶段的极简实现

### 2.1 主循环

`_start_serial`：`inbox.get(timeout=0.1)` → 对 `new_request` 调 `_collect_batch` →
`_run_batch` → outbox。所有异常被捕获后**逐请求**发 error 消息；abort 过的请求
（`_consume_if_aborted`，带锁的 check-and-clear）静默丢弃。

### 2.2 批量收集算法（编码器批量依赖它）

`_collect_batch(first_msg)`：

- 没配 `batch_compute_fn` 或 `max_batch_size<=1`：单条返回；
- 否则循环 `get_nowait` 攒批：
  - `batch_wait_when_idle=True` 时从**第一条消息到达**起算 deadline
    （`_encoder_batch_wait_ms` 环境变量控制，默认 0 = 不等）；
  - **代价上限**：`request_cost_fn` 计算每条消息的字节成本，
    `batch_cost + msg_cost > max_batch_cost` 时把该消息**塞回队首**
    （`_pending_messages.appendleft`）并结束攒批。
    图像编码器的成本函数（`stages.py:_create_image_encoder_request_cost_fn`）=
    `(原始像素字节 + 视觉 token 输出字节) × 5`，预算 `10 GiB`；
    视觉 token 数 = `grid_thw.prod(-1) // merge²`，输出层含 deepstack 的
    `1 + deepstack_layers` 份。
- 非 `new_request` 消息（流块等）一律塞 `_pending_messages` 延后处理，不打断攒批。

### 2.3 并发模式

`max_concurrency > 1` 时切换到 `_start_concurrent`：N 个 worker 协程 +
`asyncio.to_thread(self._run_compute_in_thread, ...)`，把同步 compute_fn 挪出事件循环；
与 batch 路径互斥（构造时校验）。Qwen 的 preprocessing/encoders 都用默认串行 +
批量路径。

### 2.4 Qwen 怎么用它

- preprocessing：`compute_fn = await preprocessor(payload)`（异步），单条；
- image/audio encoder：`_encode` 单条 + `_encode_batch` 批量（`stages.py:505-640`），
  `max_batch_size=32`；
- aggregate：`_identity`（join 已由 Stage 完成，merge 已在 Stage 侧做过）。

---

## 3. OmniScheduler：SGLang AR 引擎的适配层

`OmniScheduler` 是整个仓库最重的类（2677 行）。它**不是**重新实现调度，而是把
SGLang 的 `Scheduler`（schedule_batch / forward_batch / kv pool / cuda graph 那一套）
包进 inbox/outbox 契约。理解它只需要抓主循环和五个挂点。

### 3.1 主循环（`start` → `event_loop`）

```python
def start(self):
    if self.enable_async_decode:   self._event_loop_async_decode()
    elif self.enable_overlap:      self._event_loop_overlap()
    else:                          self._event_loop_normal()
```

Qwen thinker 默认 `enable_async_decode=True`（`config.py` FactoryArgs）。
每轮循环的骨架（各 loop 变体共享）：

1. `recv_requests` → `_recv_scheduler_messages`：从 SGLang 的输入通道 + 本地 inbox 排空，
   合成 recv 列表；
2. `process_input_requests`：
   - `new_request` → **请求构建队列**：`_run_request_builder` 在独立线程池执行
     `request_builder(payload)`（即 `build_sglang_thinker_request`，含 CPU 上的 M-RoPE
     计算，不能阻塞调度循环）；构建完 `_admit_or_defer_built_request` 做准入
     （KV 容量检查 `_request_kv_capacity_error` / 等待队列满 `_reject_queue_full`）；
   - `stream_chunk` → `_on_stream_chunk`：`request_id` 已在册 →
     `stream_chunk_handler(req_data, chunk)`（talker 是 `prefill_builder.append_text_chunk`）；
     不在册 → 放进 `_pending_stream_ingress`（**流先于 payload** 的兜底）；
   - `stream_done` → `_on_stream_done`：置 `thinker_chunks_done`
     （talker 还会把 `tts_eos_embed` 追加进 pending_text_queue，见 07 篇）；
3. `get_next_batch_to_run` → SGLang 批次组装（含 deferred 请求的 recheck）；
4. `_run_batch`：`model_runner.execute`（进 ModelRunner 的
   before_prefill/before_decode/…钩子，见 06/08 篇）；
5. `process_batch_result` → `stream_output(...)`：把每步产出转成 outbox 消息
   （result / stream / error）。

### 3.2 五个挂点（模型定制面）

| 挂点 | thinker 的实现 | talker 的实现 |
|------|----------------|---------------|
| `request_builder` | `make_thinker_scheduler_adapters()[0]` → `build_sglang_thinker_request` | `make_talker_scheduler_adapters()[0]` → `_build_talker_request_data`（要求已有 prefetched chunks） |
| `result_adapter` | `apply_thinker_result` 回写 `state.thinker_out` | 恒等（保持 payload 原样） |
| `stream_chunk_handler` | 无（thinker 是流的**生产者**） | `prefill_builder.append_text_chunk`：把隐藏态投影成 talker 行，追加进 pending_text_queue |
| `stream_done_handler` | 无 | `prefill_builder.mark_thinker_done`：置 done + 追加 tts_eos 行 |
| `stream_output_builder` | `make_thinker_stream_output_builder`：token→decode、hidden→talker_ar | 无（Code2Wav 的码帧由 ModelRunner 的 post_* 直接 outbox） |

### 3.3 partial start（部分启动）的调度策略

`QwenTalkerScheduler` 覆写三个决策函数（`talker_scheduler.py:51-108`）：

1. `_is_request_build_ready`：非 pending_stream_done 时，必须
   `enable_partial_start` 且可用块数 ≥ `partial_start_min_chunks` 才允许构建 talker 请求。
   "可用块数"由 `_count_usable_prefetched_chunks` 计算：**末块若是 `<|im_end|>`
   （token_id == im_end_token_id）要扣掉**——它只是会话边界标记，不是语音内容。
   硬下限：`MIN_PARTIAL_START_CHUNKS=3`（构造时校验，`partial_start_min_chunks < 3` 直接
   ValueError）。分散式配置传 5（`config.py:181`），共置式关掉 partial start。
2. `_should_recheck_deferred_request_on_stream_chunk`：partial start 开启时，
   每来一个流块都把 deferred 请求标脏重查（攒够就立刻晋升为可构建）。
3. `_is_batch_ready_to_run` + `get_next_batch_to_run`：decode 批次还得过
   `model_runner.is_decode_batch_ready`（08 篇：每个请求都有 feedback+text 输入）。
   不就绪就 `_rollback_decode_prep_after_skip(batch)`——**这是全仓库最精妙的逆向操作之一**：

```python
# SGLang 的 prepare_for_decode 已经：分配了 KV 槽、递增了 seq_lens、
# 写了 req_to_token、递增了每请求的 decode_batch_idx / kv_committed_len。
# 回滚必须对称地撤销，但 seq_lens_sum 保持不动（此时恒为 None，
# 下次 forward 前会重算），penalizer 的 cumulate scatter_ 在 talker
# 自己的 SamplingBatchInfo 下是幂等的，可以不逆。
self.token_to_kv_pool_allocator.free(batch.out_cache_loc)
for req in batch.reqs:
    req.decode_batch_idx -= 1
    req.kv_committed_len -= 1
    req.kv.kv_allocated_len -= 1
batch.seq_lens.sub_(1); batch.seq_lens_cpu.sub_(1); batch.orig_seq_lens.sub_(1)
batch.req_to_token_pool.req_to_token[batch.req_pool_indices, batch.seq_lens] = 0
```

没有这段，"decode 批组好了但文本输入还没到"就会导致 KV 泄漏或池越界。

### 3.4 talker 的 server_args 覆写

`configure_talker_server_args`（`talker_scheduler.py:14-27`）：

- `disable_radix_cache=True`：talker 输入是投影 embedding，前缀共享无意义且危险；
- `chunked_prefill_size=0`：talker prefill 一次性吃完（其输入本来就是自己构造的 9+行）；
- feedback 开启时 `disable_overlap_schedule=True`：overlap 调度会在 forward 与
  结果处理之间引入异步窗口，而 talker 的 `before_decode` 要同步写 feedback buffer，
  两者冲突。

返回值 `want_cuda_graph` 决定调用方是否在 ModelWorker 建好后补跑 `init_sglang_cuda_graphs`
（`bootstrap.py:233-236`）。

### 3.5 流输出的落点

`stream_output(reqs, ...)`（`omni_scheduler.py:1579`）把 `stream_output_builder` 产出的
`list[OutgoingMessage]` 经 `_put_stream_messages` 压进 outbox。thinker 的构建器
（`request_builders.py:make_thinker_stream_output_builder`）有三条关键规则：

1. `req.inflight_middle_chunks > 0`（chunked prefill 还有中段没吃完）→ **整体不发**。
   注释原文：过早发出的块会让 prompt 侧隐藏态"冒充"第一个 assistant token，
   把用户/ref 文本泄进 TTS。
2. `stream=True` 的请求才发 token→decode（非流式请求不发 per-token 消息），
   但 **talker_ar 的隐藏态流无条件发**（只要请求要音频）——talker 不关心你的客户端
   是否流式，它只关心自己的输入。
3. 隐藏态要拆双层：`extra["hidden_states"]` 可能是 dict
   `{"embed": 层0输出, 24: 层24输出}`（对应捕获层 `[0, 24]`），
   embed 缺失时回退 layer_hidden；单个 1D/2D 张量归一化成 1D。

---

## 4. StreamingVocoderBase：Code2Wav 的模板基类

`streaming_vocoder.py` 顶部的 docstring 就是权威规格，摘要：

- 请求级流状态注册表：首个 chunk/payload 时 `create_stream_state`，
  结束/abort 时 pop，**迟到的 chunk 永远不能重建已清理的状态**；
- `is_streaming_payload = bool(params["stream"])`；
- 契约闩锁 `latch_stream_contract`：首个流块到达时把 stream 开关锁进状态
  （Code2Wav 用它锁 `stream_enabled`，之后不再逐块判断）；
- 阈值-累积→decode_delta→emit 的固定顺序，波形统一包成
  `OutgoingMessage(type="stream", metadata={"modality":"audio"})`；
- `on_stream_done` 顺序：flush 尾窗 → 末块流消息 → 终态 result → 清状态；
- 一块都没发过时走 `fallback_full_decode` 整段兜底；
- 跨请求合批通过四个可选钩子：`select_step_participants / build_step_plan / run_step /
  on_step_failure`——失败的 step 只会 error 掉其参与者，**绝不能殃及别的流**。

Code2Wav 在此之上叠加滑窗/EOS/重叠/图（09 篇），decode 的
`StreamingDetokenizeScheduler` 则没有用它（自建 inbox 循环，10 篇讲原因）。

---

## 5. 小结

- 调度器契约极窄（两个队列 + 生命周期方法），所有"分布式复杂性"被 Stage 吸收；
- SimpleScheduler 的价值在批量收集的**代价模型**（成本函数 + 预算 + 回塞队首）；
- OmniScheduler 的价值在**挂点设计**：request_builder（CPU 重活进线程池）、
  stream_chunk/done handler（流的消费端注入）、stream_output_builder（流的生产端格式）；
- talker 的三处覆写（部分启动门槛、流块触发重查、decode 就绪回滚）是
  "流式 AR 引擎"这个概念落到工程上的全部代价。

下一篇：三份 PipelineConfig 如何把以上所有组件拼成可部署拓扑（04 篇前的框架终章）。
