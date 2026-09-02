# 10 decode 阶段与流式反分词：StreamingDetokenizeScheduler

> 主角：`components/streaming_detokenizer.py`（318 行）。
> 它是文本链的 terminal 阶段，也是全仓库最清晰地展示"payload / 流 / 信号
> 三通道如何对齐"的样本。

---

## 1. 为什么 decode 不用 SimpleScheduler

`create_decode_executor`（stages.py）直接返回
`create_streaming_detokenize_scheduler(model_path)`。docstring 说明来历：
它**替代了**基于 SimpleScheduler 的一次性 decode——因为 thinker 语音模式下
每 token 都会流过来（`stream_to` 含 decode），一个只处理 `new_request` 的
调度器无法消费流。它的 inbox 契约与 Code2Wav 同构：

- `new_request`：thinker 的终态 payload（`next="decode"` 那份）；
- `stream_chunk`：`StreamItem(data=torch.tensor([token_id]), metadata={"token_id": t})`；
- `stream_done`：`resolve_thinker_stream_done_targets` 的信号。

`Stage.can_accept_stream_before_payload=True` 对此阶段是必需的：
流必然先于 payload（payload 要等 thinker 生成完）。

---

## 2. 主循环与失败隔离

```python
while self._running:
    msg = self.inbox.get(timeout=0.1)
    try:
        if msg.type == "new_request":   self._on_new_request(msg.request_id, msg.data)
        elif msg.type == "stream_chunk": self._on_stream_chunk(msg.request_id, msg.data)
        elif msg.type == "stream_done":  self._on_stream_done(msg.request_id)
    except Exception as exc:
        self.abort(msg.request_id)
        self.outbox.put(OutgoingMessage(request_id, "error", data=exc))
```

注释（重要）：异常若逃出 `start()` 会触发 `Stage._handle_scheduler_crash`，
**杀掉 decode 阶段上所有在途请求**——所以每条消息一个 try/except，
坏请求只坏自己。这是全仓库统一的调度器契约（Simple/Omni/Code2Wav 同款）。

---

## 3. 增量反分词：UTF-8 边界安全

```python
def _on_stream_chunk(self, request_id, item):
    token_id = int(item.data.item()) if hasattr(item.data, "item") else int(item.data)
    s.pending_tokens.append(token_id)
    candidate = tokenizer.decode(s.pending_tokens, skip_special_tokens=True)
    if "�" in candidate:      # U+FFFD：多字节字符被截断
        return                 # 挂起，等下一个 token 补齐
    s.pending_tokens.clear()
    if not candidate: return   # 特殊 token 全部被跳过 → 无可发
    outbox.put(OutgoingMessage(request_id, "stream", target=None,
        data={"text": candidate, "modality": "text", "stage_name": "decode"},
        metadata={"modality": "text"}))
```

细节：

- `target=None` = **terminal 流**，Stage 会直接发 Coordinator（01 篇 §6），
  客户端收到的是 `{"text": delta}`；
- `_finalize` 时若 `pending_tokens` 还有残货（例如 max_tokens 截断在多字节字符中间），
  必须再 decode 一次发出去——注释：否则流式客户端会**永久丢失**这些尾字节，
  而非流式客户端（走整段 decode）却能看到，二者不一致。

---

## 4. 三通道对齐：payload × 流 × stream_done

### 4.1 状态机

```python
@dataclass
class _RequestState:
    pending_tokens: list[int]
    payload: StagePayload | None
    done: bool = False
```

- `_on_stream_done`（无状态行时）：`_done_seen[request_id] = None`。
  两种合法情形：**零 token 生成**（没有任何 chunk 建过状态行）与
  **迟到重复 done**（finalize 已把状态行删掉）。
- `_on_new_request`：payload 到 → 查 `_done_seen`（零 token 情形：done 立即为真）→
  非流式或已 done → 立即 `_finalize`。

### 4.2 _done_seen 的容量治理

```python
_DONE_SEEN_MAX = 10000      # 上限：零 token 竞态 + 迟到 done 的孤儿条目
_DONE_SEEN_EVICT_TO = 5000  # 超限后 FIFO 淘汰到 5000
```

OrderedDict 头进尾出，最老的先淘汰——两个常数配一对，
避免热路径上的反复淘汰抖动。

### 4.3 为什么 done 可以先于 payload

thinker 的 `stream_done_to_fn` 在**生成结束时**就向 decode 发 done 信号，
而终态 payload 还要走 `project_thinker_to_decode` + relay 传输。
zero-token 请求（例如被 stop 条件立即终止）就是 done 先到、payload 后到的
真实案例。`_done_seen` 把这次竞态变成"payload 到时补 finalize"。

---

## 5. 终态 result 的构建（_build_result:220-310）

终态 `OutgoingMessage(type="result", data=payload)` 的 `payload.data` 被
替换成 result dict。构建过程：

1. 从 state 取 `thinker_out`（或 `engine_outputs["thinker"]`）；
2. 调 **merge.decode_events**（05 篇 §4）生成事件列表——
   同一函数保证流式路径与终态文本一致；
3. 取最后一个 `is_final / text_final / final` 事件的 payload 作为顶层字段；
4. **流式请求删掉 `text`**（注释全文值得背）：

   > 流式客户端已经通过逐 token 流收到全文；
   > `Client.completion_stream()` 的直接消费者会拼接每个 chunk 的 "text" 字段，
   > 终态再带全文等于**输出两遍**。
   > "Mirrors the code2wav slim-final contract for audio"——
   > 音频侧（09 篇 final_result_data 的流式分支）是同一契约。

5. 非流式且无 text → `tokenizer.decode(output_ids)` 整段补上；
6. `finish_reason` / `output_token_logprobs` / `weight_version` 透传；
7. **usage**：`prompt_tokens = input_ids.numel()`（兼容张量/列表）、
   `completion_tokens = len(output_ids)`、总数——注意 decode 阶段自己算
   usage，不依赖上游。

---

## 6. abort 语义

`abort(request_id)`：删 `_state` 与 `_done_seen` 条目。迟到的流块/终态
（`request_id` 已不在册）被静默丢弃——由 Stage 层的 `_aborted` 集合先行拦截，
这里只是第二道闸。

---

## 7. 与 thinker 流输出的一处精妙协作

回忆 06 篇：thinker 的 token 流只在 `stream=True` 时发。那么非流式请求的
decode 收到什么？——只有终态 payload。此时：

- `_on_new_request`：非流式 → 立即 finalize → `decode_events` 对完整
  `output_ids` 一次性产出 `text_final` 事件 → result 带 `text`。
- 流式请求：每 token 走 `_on_stream_chunk`，终态 payload 到时
  `decode_events` 会再算一遍（全量 token）——但它的 `text` 被删了，
  只留事件元数据与 usage。

也就是说 **decode 阶段对"流式/非流式"的全部差异处理**最终归结为一行
`result.pop("text", None)`，而文本内容的一致性由 `decode_events` 单点保证。
这是"终态与流态共享一个纯函数"设计红利的直接体现。

---

## 8. 小结

1. decode 是一个**三输入状态机**：流（增量文本）、信号（done）、payload（终态）；
   `_done_seen` 治理"done 先于 payload"竞态，`pending_tokens` 治理 UTF-8 边界。
2. 失败隔离模式（per-request try/except + error outbox）与全仓库调度器契约一致。
3. 流式终态的 slim contract（去 text）与音频侧互为镜像——
   **任何"已流式发送的内容不得在终态重复"**是整个系统对客户端的统一承诺。
4. usage/finish_reason/logprobs 的透传与补算都发生在这一层，
   Coordinator 拿到的 result 已经是可直接序列化的最终形态。

下一篇（11）：支撑这一切的物理层——CommEngine 与张量传输。
