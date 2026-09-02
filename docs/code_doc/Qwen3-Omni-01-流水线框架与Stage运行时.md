# 01 流水线框架与 Stage 运行时

> 主角：`sglang_omni/pipeline/stage/runtime.py`（1843 行）、`pipeline/coordinator.py`（875 行）、
> `pipeline/control_plane.py`、`pipeline/stage/input.py`、`pipeline/stage/stream_queue.py`。
> 上一篇建立了全景，本篇拆开框架的承重墙：**Stage 到底替你管了什么**。

---

## 1. 一条消息在框架里的分类学

Stage 收到的所有东西只有六种（`runtime.py:_handle_message`）：

| 消息 | 来源 | 语义 |
|------|------|------|
| `SubmitMessage` | Coordinator | 新请求的入口 payload（仅 entry 阶段） |
| `DataReadyMessage` | 上游 stage | "有一份数据可以来读"（payload / 流块 / 流信号 三态由字段区分） |
| `DataAckMessage` | 下游 stage | 对我方发出的数据的确认（配额/清理依据） |
| `CompleteMessage` | 上游 terminal | 终态回执 |
| `ProfilerStart/Stop`、`AdminMessage`、`ShutdownMessage` | Coordinator | 治理面 |
| `TPWorkMessage` | TP leader fanout | TP follower 收到的"干活"指令 |

`DataReadyMessage` 内部再三分流（`_schedule_receive_task`，`runtime.py:391-407`）：

- `is_done=True` 或 `error!=None` → `_on_stream_signal`（流终止/异常信号）；
- `chunk_id!=None` → `_on_stream_chunk`（时序流块）；
- 否则 → `_on_data_ready`（一次性 payload）。

三分流对应三种读取路径：**direct CUDA IPC 反序列化 → inline（消息内嵌小张量）→ relay 读**。
区分点在 `data_ref` 的 `object_id` 约定（`stage_io.is_direct_cuda_ipc_payload_ref` 等谓词）。
direct-IPC 与 inline 分支不经过 `CommEngine.read_stream_chunk`，因此必须**自己补发**
`comm_stream_read` 追踪事件（代码里有明确注释：否则同 GPU 边沿的 held-byte 统计会漏掉）。

---

## 2. 输入聚合：`wait_for` / `wait_for_fn` / `merge_fn`

这是理解 Qwen3-Omni 语音拓扑的钥匙。`mm_aggregate`（文本拓扑）和 `thinker`/`talker_ar`
（语音拓扑）都要等三个上游。

### 2.1 聚合状态机

`InputHandler`（`stage/input.py`，两种实现：`DirectInput` 单源直通 / 多源聚合）按
`(request_id, 逻辑源名)` 收 payload：

1. 每个 `DataReadyMessage` 到达后 payload 被塞进聚合器；
2. 聚合器检查 `wait_for` 集合是否集齐；
3. 集齐 → `receive()` 返回 merged payload → Stage 调 `_execute(merged)` → 塞进 scheduler.inbox。

### 2.2 动态等待集合：`wait_for_fn`

静态 `wait_for=["preprocessing","image_encoder","audio_encoder"]` 有个问题：纯文本请求没有
encoder 源，会永远等不齐。所以 Qwen 配置挂了 `wait_for_fn =
request_builders.resolve_mm_aggregate_wait_sources`（`request_builders.py:96-105`）：

```python
def resolve_mm_aggregate_wait_sources(request_id, from_stage, payload):
    if from_stage != "preprocessing":
        return None            # 维持现有等待集合，不缩
    state = Qwen3OmniPipelineState.from_dict(payload.data)
    return ["preprocessing", *_active_encoder_stages(state.encoder_inputs)]
```

语义约定：**preprocessing 的 payload 到达时**才有资格收窄等待集合——因为只有
preprocessing 知道本次请求带了哪些模态（`encoder_inputs` 里的 `_active` 标记，
由 `_has_encoder_model_input` 判定：图像看 `pixel_values/pixel_values_videos`，
音频看 `input_features`，且 `_skip=True` 的直接剔除）。
其它源到达时返回 `None` = "别动"。

### 2.3 merge_fn：集齐之后做什么

集齐的 payload 们进 `merge_fn`。语音拓扑里 thinker 与 talker_ar 共用同一个上游集合，
但 merge 不同：

- thinker：`merge.merge_for_thinker` —— 生成完整 `thinker_inputs`（05 篇详拆）；
- talker_ar：`request_builders.merge_for_talker` = `project_mm_aggregate_to_talker_ar(
  merge_for_thinker(payloads))` —— **复用 thinker 的融合结果再做一次"早提交投影"**：
  剔除 deepstack 键，只保留 `prompt + thinker_inputs`（`request_builders.py:213-232`）。
  talker 的 V1 prefill 策略就是把整个 thinker prompt 重放为投影 embedding，
  所以它需要的输入和 thinker 几乎一样，只在特征取舍上不同。

### 2.4 为什么 join 放进 AR 阶段而不是独立聚合阶段

原文本拓扑有独立 `mm_aggregate` 阶段（`_aggregate_stage`），语音拓扑取消之（见 00 篇）。
收益：省一次全量 payload 的收发与两次投影；代价：`OmniScheduler` 必须能接收
"一份数据到达时可能还差别的源"的语义，并且聚合发生在 stage 线程（asyncio）而非调度线程。
注意 `disable_direct_cuda_ipc_payload=True` 同时挂在 mm_aggregate 与 audio_encoder 上——
聚合阶段与音频塔明确禁用 direct IPC payload 捷径（多源合并前 payload 必须是可复制的普通内存）。

---

## 3. 接收顺序的硬保证：receive lane

多源、多流块的到达是乱序的，但 Stage 必须保证**同一逻辑源内的事件按到达序处理**。
实现是"每 lane 一个 future 链"（`runtime.py:391-434`）：

```python
lane = (request_id, from_stage)
predecessor = self._receive_lane_tails.get(lane)      # 该 lane 上一个任务
completion = loop.create_future()
self._receive_lane_tails[lane] = completion
task = create_task(self._run_receive_task(handler(msg, predecessor), lane, predecessor, completion))
```

每个接收任务先 `await predecessor` 再继续，形成**单 lane 串行、跨 lane 并行**的 DAG。
这解决了两个真实竞态：
1. 同一上游的 stream_chunk 与 payload 乱序（payload 先到才能建 `StreamQueue` 条目）；
2. abort 后迟到数据的丢弃顺序。

---

## 4. 流通道：StreamQueue 与 pre-payload 流

`StreamQueue`（`stage/stream_queue.py`）维护 `(request_id → 有序 StreamItem 队列)`。
关键开关是 Stage 构造参数 `can_accept_stream_before_payload`：

- `decode`、`talker_ar`、`code2wav` 都为 `True`（`config.py` 中三个 stage 均显式声明）；
- 原因：thinker 一边跑 prefill 一边就开始每 token 发流，此时 thinker 的终态 payload
  （`next="decode"` 那份）还没到。接收端必须允许"流先于 payload"。

`_open_pre_payload_stream_if_allowed`（`runtime.py:1193-1202`）：
队列已有 → 直接路由；没有且允许 → 先 `stream_queue.open(request_id)` 再路由；
不允许 → abort 该请求并向 Coordinator 发 failure（"stream arrived before the request
payload, but this stage is not configured to accept pre-payload stream data"）。

`stream_done` 同样受此开关管辖（`_receive_stream_signal`），到达后除入队外还向 scheduler.inbox
投一条 `IncomingMessage(type="stream_done")`——这是 OmniScheduler/Code2Wav 感知"上游说完了"
的唯一入口。

---

## 5. `_execute`：把请求交给调度器

```python
async def _execute(self, payload):
    if payload.request_id in self._aborted: return
    if self.role == "leader" and self._tp_fanout is not None \
       and getattr(self.scheduler, "requires_tp_work_fanout", False):
        self._tp_fanout.fanout_work(payload)      # TP: rank0 把活分给 followers
    msg = IncomingMessage(request_id=..., type="new_request", data=payload)
    if (enqueue := getattr(self.scheduler, "enqueue", None)):
        enqueue(msg)                               # OmniScheduler 有带准入控制的 enqueue
    else:
        self.scheduler.inbox.put(msg)              # Simple/Code2Wav 直接进队列
```

三个细节：
- **TP fanout**：`role="leader"` 且调度器声明 `requires_tp_work_fanout` 时，leader 先把
  payload 广播给 followers 再自己跑（follower 的 Stage 主循环直接把 `TPWorkMessage`
  转 `_execute`）。
- **scheduler 线程模型**：scheduler 在独立 daemon 线程跑（`Stage.start` 里创建，
  `gpu_id is not None` 时先 `set_device`），与 Stage 的 asyncio 主循环通过两个
  `queue.Queue`（inbox/outbox）解耦。线程崩溃通过
  `asyncio.run_coroutine_threadsafe(self._handle_scheduler_crash(exc), loop)` 回传主循环。
- **admin 操作**走 `loop.run_in_executor` 调 `scheduler.admin(action, payload)`，
  TP leader 会额外收集 followers 的 `AdminResult` 并聚合（`_on_admin`），失败判定为
  `all(rank_results.success)`。

---

## 6. Outbox 排空与结果路由

`_drain_outbox` 是一个常驻任务：`loop.run_in_executor` 以 0.1s 超时阻塞取 outbox，
单轮最多连取 `_OUTBOX_DRAIN_BATCH_SIZE=64` 条（取满后 `await asyncio.sleep(0)` 让出事件循环）。
三类出口（`_drain_outbox_external`）：

1. `type="result"` → `_route_result`：
   - 先给 `get_stream_done_targets(request_id, result)`（thinker 用
     `resolve_thinker_stream_done_targets`：要音频 → `[talker_ar, decode]`）逐个发 `stream_done`；
   - 再问 `get_next(request_id, result)`：
     - 返回 `None` → terminal，`send_complete` 给 Coordinator（携带 `result.data`）；
     - 返回 stage 名（单/多）→ `_send_to_stage`，并按目标数选择
       `allow_local_object`（单目标可走进程内对象直传）或
       `allow_projected_local_object`（多目标时投影后直传）。
2. `type="stream"` → `target is None` 时走 `stream_targets`（构造时静态配置，如 talker→code2wav）
   或直接发 Coordinator（terminal 流，如 decode 的文本增量）；有 `target` 则点对点。
3. `type="error"` → `_send_failure`。

follower 的排空是阉割版（`_drain_outbox_follower`）：result 只清状态、stream 直接丢、
error 直接抛——**外部 IO 是 leader 独占的**。

### 路由前的投影：project_payload

`_send_to_stage` 在真正发送前查 `self._project_payload[target]`，把 payload 压成
目标阶段需要的形状。Qwen 使用的全部投影（`request_builders.py`）：

| 投影 | 从 → 到 | 保留什么 |
|------|---------|----------|
| `project_preprocessing_to_image_encoder` | preprocessing → image_encoder | 仅该塔的 `encoder_inputs` |
| `project_preprocessing_to_audio_encoder` | preprocessing → audio_encoder | 仅该塔的 `encoder_inputs` |
| `project_preprocessing_to_mm_aggregate` | preprocessing → join 目标 | `prompt` + 轻量 `mm_inputs`（只留 grid/mask/lengths 等元数据）+ `encoder_inputs` 元数据（cache_key/_skip/_active） |
| `project_encoder_to_mm_aggregate` | encoder → join 目标 | 仅本塔 `encoder_outs`（强制单键，否则 ValueError） |
| `project_encoder_to_talker_ar` | encoder → talker_ar | 同上，但剔除 `deepstack_visual_embeds_*`（talker 不用，见 `_TALKER_UNUSED_ENCODER_OUT_KEYS`） |
| `project_thinker_to_decode` | thinker → decode | 清 `thinker_inputs`、`extra_model_outputs`，深拷贝可变容器（张量除外） |
| `project_talker_to_code2wav` | talker_ar → code2wav | **空 data**（纯 latch，码张量走流） |

`project_thinker_to_decode` 里 `_copy_mutable_containers` 值得注意：dict/list/tuple/set/bytearray
递归复制，但 **torch.Tensor 原样返回**——共享引用以省拷贝，因为下游 decode 只读 token id。

---

## 7. 复制/_abort 与生命期管理

- `_active_requests` 集合是"我正在参与该请求"的事实记录；outbox 消息只有
  `request_id ∈ _active_requests` 才会被路由，`_clear_request_state` 在 result 路由完成后清理。
- `_aborted` 集合挡住迟到的 payload/流块（各 `_on_*` 入口第一件事就是查它）。
  abort 监听是常驻任务 `_abort_listener`（control plane 推 `AbortMessage`）。
- 请求结束后还必须**排干在途数据**（`_discard_data` / `_discard_stream_chunk_data`）：
  即便请求已 abort，已分配的 relay 对象仍要读一次并发 Ack，否则发送端的配额窗口会泄漏。
  KV_PAGES 是例外：直接拒绝并报"request was aborted"（KV 池页有独立回收路径）。

---

## 8. 崩溃语义：谁坏了、坏多大

框架的故障模型在多处注释里写得很清楚：

1. **scheduler 线程崩溃** → `_handle_scheduler_crash` → Stage `_running=False`，
   主循环 `run()` 在 finally 里重抛，进程级故障域（`stage_workers._run_process` 中同进程
   的多个 `role="single"` stage 共享进程也共享故障域）。
2. **单请求计算失败**（scheduler 内部 try/except）→ 只发该请求的 error 消息，例如
   `StreamingDetokenizeScheduler.start` 的注释明确说：异常若逃出 `start()` 会触发
   Stage 的 scheduler 崩溃处理，连带杀掉该 stage 上所有在途请求——所以每个消息循环
   都包了 per-request 的 try/except。
3. **后台任务失败**（abort 监听、outbox 排空、接收任务）→
   `add_done_callback(self._on_background_task_done)` 记录 `_background_task_error`，
   主循环退出时重抛。绝不允许静默吞掉。
4. **Coordinator 视角**：`fail_pending_requests` 把所有在途请求置 FAILED 并向各
   stream queue 注入失败 Complete；`shutdown_stages` 广播 Shutdown。

---

## 9. Coordinator：请求状态机

`coordinator.py` 的职责清单：

- **stage 注册表**：`register_stage(name, endpoint)`；
- **请求跟踪**：`_requests: dict[str, RequestInfo]`、`_completion_futures`、
  `_stream_queues: dict[str, asyncio.Queue[CompleteMessage|StreamMessage]]`——
  客户端迭代的就是后者；
- **终态裁决**：`terminal_stages` 或 `terminal_stages_resolver`（Qwen 的
  `resolve_terminal_stages` 按请求元数据返回 `[decode]` 或 `[decode, code2wav]`）。
  多终态 = **全部**终态都回执才算完成（语音请求同时等文本与音频两条链）；
- **准入**：`max_in_flight`（= 生成容量 + 队列容量），超限抛 `QueueFullError`；
- **副本绑定**：`ReplicaTopology` + `RoundRobinBindingPolicy`，每个请求为复制的
  Process 选一个副本实例，绑定随消息传播（`replica_bindings` 字段），
  Stage 端用 `_resolve_target_instance` 把逻辑目标名翻译成物理实例名；
- **admin 聚合**：`AdminOperation(op_id, action, payload, target_stages, timeout_s)`，
  `_AdminPendingOperation` 等所有目标回执，`_aggregate_admin_results` 汇总；
- **abort**：`AbortMessage` 只带 request_id；持强引用的 abort task 保证"本地准入已关 +
  广播能发出去"，即使调用方协程被取消。

---

## 10. 本篇小结（与其他篇的接口）

- Stage 是**无计算 IO 壳**：聚合（wait_for/wait_for_fn/merge_fn）→ 执行（scheduler inbox）→
  路由（get_next + project_payload + stream 通道）。
- 流与 payload 是两套物理通道、一套控制面；`can_accept_stream_before_payload` 是流先行的开关。
- 下一篇（02）进入 Stage 的对偶物：scheduler 侧的两种主循环与 OmniScheduler 的 AR 调度细节。
