# 07 Thinker-Talker 流式耦合与部分启动：TalkerPrefillBuilder、PendingTextTensorQueue

> 主角：`components/talker_prefill.py`（407 行）、`components/talker_input.py`（273 行）、
> `pending_text_queue.py`（151 行）、`request_builders.py` 的 `_build_talker_request_data`、
> `talker_scheduler.py` 的 partial start 策略（02 篇已述）。

---

## 1. 问题定义：talker 是一条"未完成的请求"

Thinker 每 token 流出两个东西：token_id（→decode）与隐藏态（→talker_ar）。
talker 要在 thinker **还在生成时**就开始出声（TTFA 优化），这带来三个矛盾：

1. talker 的 prefill 需要完整 prompt 投影，但 thinker prompt 的 assistant 部分
   还没生成完；
2. talker 的 decode 需要"下一个文本行"，但文本一行行地在流里后到；
3. talker 的 KV cache 一旦建好就不能变，但 prompt 是增量到达的。

本仓库的解法是三件事的组合：

- **部分启动**：攒够 N 个流块就构建 prefill（N=5，硬下限 3）；
- **prompt 重建**：prefill 时不重放 thinker 的计算，而是**从 checkpoint 直接捞
  embedding 重建投影**（便宜且与 thinker 无关）；
- **未来文本行 FIFO**：prompt 里放不下/还没生成的文本行进
  `PendingTextTensorQueue`，decode 逐行消费。

---

## 2. 流块的到达与暂存

`OmniScheduler._on_stream_chunk`（02 篇）把 thinker 的流块递给
`stream_chunk_handler = prefill_builder.append_text_chunk`。在此之前，
框架层已经把"payload 未到时先到的流块"攒进 Stage 的 StreamQueue 与
scheduler 的 `_pending_stream_ingress`。payload（thinker join 后经
`merge_for_talker` 的早提交投影）到达后，`prefetched_chunks` 与
`prefetched_stream_done` 两个字段成为 talker request_builder 的直接输入
（`request_builders._build_talker_request_data:272-280`）：

```python
thinker_chunks = list(payload.prefetched_chunks)
thinker_done = bool(payload.prefetched_stream_done)
if not thinker_chunks:
    raise RuntimeError("talker request_builder requires prefetched thinker chunks; "
                       "check the partial-start readiness policy or upstream wiring")
```

即：**走到 request_builder 时流块已经由调度策略担保至少 `partial_start_min_chunks` 个**
（02 篇的 `_is_request_build_ready`），这里的 raise 只是最后一道防线。

---

## 3. TalkerPrefillBuilder.build_prompt_prefill 全流程

### 3.1 重建 prompt 三态（`_reconstruct_prompt_states`）

```python
prompt_ids = state.prompt["input_ids"]            # [seq] long, CPU
prompt_embed  = self._load_prompt_token_embeddings(prompt_ids)   # [seq, thinker_hidden]
prompt_hidden = prompt_embed.clone()
# 多模态行覆写：
merge_prompt_modality(prompt_ids, prompt_embed, prompt_hidden,
                      token_id=audio_token_id, features=model_inputs["audio_embeds"])
# 同样处理 image / video
```

`merge_prompt_modality`：`mask = prompt_ids == token_id`，然后
`prompt_embed[mask] = feature_tensor`（搬设备/对 dtype），**`prompt_hidden[mask] = 0.0`**。
为什么 hidden 置零？因为投影是双轨的（见 §3.4）：

- 文本行 → `text_projection(embedding)`
- 多模态行 → `hidden_projection(thinker_hidden_state)`

而 talker prefill 阶段拿不到 thinker 的"prompt 侧隐藏态"（那是 prefill 中间量，
没有持久化），HF 的做法是：prompt 侧多模态行用 encoder embedding 的
`hidden_projection` 近似。把 hidden 置零等于声明"这一行将在投影时走 hidden 轨"。
talker 模型的 `prepare_input_embeds`（08 篇）用同一个 mask 分派两条投影。

### 3.2 从 safetensors 直接捞 embedding（talker_prefill.py:20-59）

```python
_THINKER_EMBED_CANDIDATE_KEYS = ("thinker.model.embed_tokens.weight", "model.embed_tokens.weight")

def load_thinker_embedding_rows(model_path, row_ids):
    shard_path, tensor_name = _resolve_embed_source(model_path)     # 索引文件 → 分片定位，进程内缓存
    handle = _EMBED_HANDLE_CACHE[model_path]                        # safe_open 句柄复用
    tensor_slice = handle.get_slice(tensor_name)
    rows = [tensor_slice[row_id] for row_id in row_ids]             # 逐行 mmap 读取
```

关键设计：**不开整个 embedding 矩阵**（30B 模型的 embedding 是 GB 级），
只 mmap 读需要的行。`_load_prompt_token_embeddings` 再加一层
`_thinker_embed_cache: dict[int, Tensor]`（token_id → 行），
用 `torch.unique(return_inverse=True)` 去重查表后 `index_select` 还原顺序。
prompt 高频 token（im_start/system/user 等特殊 token）几乎全部命中缓存。

### 3.3 chat template 分段（talker_input.segment_chat_template）

按 `<|im_start|>`（151644）切开输入，取其后一个 token 判角色
（system=8948 / user=872 / assistant=77091），产出
`[{"role", "start", "end"}]`（**含 im_start 本身**，与 HF 一致）。
系统段被跳过（talker 不需要 system 文本），多轮里**只有最后一个 assistant 段**
参与 assistant 构建（历史 assistant 轮次属于"user 侧"上下文，走 user_part）。

`build_prefill_input` 还有一个 off-by-one 修正（注释）：assistant 段末尾的
`<|im_end|>` 要剥掉——HF 的 `thinker_embed` 从不包含 EOS 的隐藏态
（`generate()` 在产出 EOS 前就停了），不剥会让 future_text_rows 整体偏移一行。

### 3.4 user_part：双轨投影

```python
result[multimodal_mask] = hidden_projection(thinker_hidden[multimodal_mask])
result[~multimodal_mask] = text_projection(thinker_embed[~multimodal_mask])
```

多模态行此时 hidden=0（被覆写过），所以实际是 `hidden_projection(0 + embedding)`？
——不是：`merge_prompt_modality` 里 `prompt_hidden[mask] = 0.0` 后，
`_reconstruct_prompt_states` 又没有再往 hidden 里放东西，
因此多模态行真正喂给 `hidden_projection` 的是**零向量**。
零向量经 MLP（含 bias）得到的是"投影偏置"，随后与 assistant 侧布局相加。
这与 HF 的对齐方式一致：talker 侧真正携带多模态信息的是
`hidden_projection` 的输出作为"该行的特征"这一约定，
而 prompt 侧多模态信息其实由 **codec 侧 embedding 与后续 audio 特征行**
承担（语音合成关注的是 assistant 文本与说话人，prompt 图像/音频只提供语气上下文）。
读这段代码时不要试图在 prompt 多模态行里找"视觉信息"——它被刻意置零了。

### 3.5 assistant_part：9 行布局（talker_input.build_assistant_part）

这是与 HF `_get_talker_assistant_parts` 逐位对齐的部分：

```
文本轨（text_projection 后，[9, hidden]）：
  [0:3]   assistant 前 3 个 token 的投影
  [3:7]   4 × tts_pad_embed                       ← 占位，稳定布局
  [7]     tts_bos_embed                            ← 语音开始符
  [8]     assistant 第 4 个 token 的投影            ← 还没生成时放零（后来经未来行队列补）
codec 轨（[9, hidden]）：
  [0:3]   3 × zeros
  [3:9]   codec_embedding([nothink, think_bos, think_eos, speaker_id, pad, codec_bos])
最终：input_embeds = text_hidden + codec_hidden
      input_ids   = 9 × tts_pad_token_id（占位 id，位置追踪用）
future_text_rows = projected[4:] + tts_eos_embed       ← 关键！
```

三个理解要点：

1. **为什么是 9 行固定布局**：talker 需要在 prefill 里就看到"语音启动序列"
   （nothink/think 边界/说话人/pad/codec_bos），HF 的布局把 codec 特殊 embedding
   与文本前 3 token 叠加，形成"文本轨承载语义、codec 轨承载声学身份"的加性混合。
2. **`future_text_rows = projected[4:] + tts_eos`**：从第 5 个 assistant token 起的
   所有投影 + 语音 EOS 行，构成"未来文本行队列"的**初始内容**——第一个 decode 步
   消费 `projected[4]`（正好接在 `[8]` 位置的"第 4 个 token"之后）。
3. **partial start 下的不完整**：`projected.shape[0] <= 3` 时第 4 token 槽放零
   （注释：初始 talker 请求可能在 thinker 发出 4 个 token 前构建，槽位保持稳定，
   由未来行队列后续更新）。`include_assistant_eos=False`（thinker 未完）时
   future_text_rows 去掉最后的 tts_eos 行——EOS 只有 thinker 真正结束后才该出现。

### 3.6 增量与终止（append_text_chunk / mark_thinker_done）

```python
def append_text_chunk(self, req_data, chunk):
    if req_data.thinker_chunks_done: return
    if chunk.metadata.get("token_id") == im_end_token_id: return   # 会话边界不入队
    pending_text_queue.append(self.project_assistant_chunk(chunk))
    # project_assistant_chunk：有 token_id → 捞 embedding 行再 text_projection；
    #                          无 token_id → 直接用 chunk.data（已是隐藏态）投影

def mark_thinker_done(self, req_data):
    req_data.thinker_chunks_done = True
    pending_text_queue.append(req_data.tts_eos_embed)   # 补上被扣掉的 EOS 行
```

流块若带 token_id（thinker 侧 metadata 总是带），投影路径与 prefill 完全一致
（同源 embedding + 同一 text_projection），保证 prompt 内外的文本行分布一致。

---

## 4. PendingTextTensorQueue：设备端 FIFO 的精确设计

（`pending_text_queue.py` 全文 151 行，值得整篇读）

### 4.1 数据结构

```python
@dataclass(slots=True)
class PendingTextTensorQueue:
    rows: torch.Tensor | None    # 当前头块
    cursor: int                  # 头块内的读位置
    _chunks: deque[torch.Tensor] # 后续整块
    _pending_rows: int           # 缓存的剩余行数（O(1) len）
```

- `append_rows`：行规格检查（1D→reshape(1,-1)；2D 要求 hidden 维非空），
  隐藏维必须与既有头块一致，然后 `rows.to(device=..., dtype=...)` **就地转到设备**
  并压入 `_chunks`（不拼接！避免 O(n) 拷贝）。
- `popleft`：读 `self.rows[self.cursor]`，cursor+1；**头块耗尽时
  `self.rows = self._chunks.popleft()`**——升块零拷贝，旧头块交给 GC。
- `__getitem__(0)` 走 O(1) 捷径（注释：talker decode 只 peek 第一行，
  热路径绝不 consolidate）；任意下标才 `torch.cat([remaining, *chunks])[idx]`。
- `coerce_pending_text_queue`：None/Queue/Tensor/Iterable 四态归一，
  Queue 会 `copy()`（跨请求共享必须复制，防串扰）。

### 4.2 为什么必须设备端

decode 每步要做 `feedback + text` 的逐行相加（08 篇），若 text 行在 CPU，
每步一次 H2D + 同步。设备端 FIFO 让"peek 下一文本行"变成纯 GPU 读。
这与 talker 模型里 `pending_feedback_queue`（同为设备张量 deque）对称——
两条 FIFO 的行最终在 `_combine_feedback_with_next_text` 里相加
（`row + next_text`），都在 GPU 上完成。

---

## 5. 组装成 SGLang 请求（`_build_talker_request_data`）

```python
sampling_cfg = _resolve_talker_sampling_config(params)
#   max_new_tokens=talker_max_new_tokens(4096), temperature=0.9, top_k=50,
#   top_p=1.0, repetition_penalty=1.05, codec_eos_id=model.config.codec_eos_token_id,
#   suppress_tokens=[codec_vocab-1024 .. codec_vocab) 除 eos 外全部抑制]
#   —— 抑制区间：codec 词表尾部 1024 个是非语音保留位，除 EOS 外必须封死
if sampling_cfg["seed"] is None:
    sampling_cfg["seed"] = xxhash.xxh64_intdigest(request_id) & 0x7FFFFFFF
    # 请求级确定性种子；PyTorch 采样种子必须落在正 int32（MAX_INT32_POSITIVE）

prompt_prefill = prefill_builder.build_prompt_prefill(payload, thinker_chunks, thinker_done=...)
req_data = build_sglang_talker_request(
    thinker_hidden_states=torch.empty(0),        # 空占位：真实嵌入走 talker_input_embeds
    talker_input_embeds=prompt_prefill["input_embeds"],
    talker_input_ids=prompt_prefill["input_ids"],
    input_embeds_are_projected=True,             # 已在 talker 空间，forward 不再投影
    pending_text_queue=pending_text_queue,
    tts_pad_embed=prompt_prefill["tts_pad_embed"],   # 文本行耗尽后的填充行
    thinker_chunks_done=thinker_done,
    talker_model_inputs=prompt_prefill["prompt_model_inputs"],  # M-RoPE 判定用
    seed=...)
req_data.tts_eos_embed = prompt_prefill["tts_eos_embed"]         # mark_thinker_done 要用
```

`build_sglang_talker_request`（request_builders.py:518-680）里还有两件与 M-RoPE 相关的事：

- `talker_can_use_linear_mrope(ids, mm_model_inputs, thinker_config)` →
  可用则 `linear_mrope_positions`，否则全量 CPU 计算（05 篇 §3.4）；
- `talker_multimodal_mask`：assistant/重放行的多模态标记进
  `omni_model_inputs["talker_multimodal_mask"]`，供 forward 的双轨投影（08 篇）。

采样注释里的硬核警告必须复述：**pytorch 采样后端是为了绕开 SGLang 上游缺陷**——
`Sampler.forward` 不把 seed 传给 flashinfer，CUDA graph 下捕获的 RNG 变成
"启动时决定"，约 5% 的 prompt 触发退化循环（#408）。上游修复前不可回退。

---

## 6. 时序总图

```
thinker 每 token：
  OutgoingMessage(stream, hidden, target=talker_ar)
        │ (Stage 路由; payload 未到时先入 StreamQueue)
        ▼
talker_ar Stage ── payload 到达(join 后) ──▶ prefetched_chunks[] ──▶ OmniScheduler
        │                                                        │ _is_request_build_ready?
        │                                                        │  (>=5 块 或 stream_done)
        │                                                        ▼
        │                                    request_builder: TalkerPrefillBuilder
        │                                        ├─ 重建 prompt 嵌入(safetensors 捞行)
        │                                        ├─ 9 行 assistant 布局
        │                                        └─ future_text_rows → PendingTextTensorQueue
        │                                                        │
        │                                          SGLang Req(prefill_input_embeds=投影行)
        ▼                                                        ▼
之后每个流块 ──append_text_chunk──▶ pending_text_queue   prefill → decode 循环(08 篇)
stream_done ──mark_thinker_done──▶ 追加 tts_eos 行
```

---

## 7. 小结

1. partial start 的本质：**把"prompt 不完整"转化为"prompt 完整 + 未来行队列"**，
   KV cache 一次建对，后续只追加 decode 行。
2. prompt 重建选择"从 checkpoint 捞 embedding"而非"缓存 thinker 中间态"：
   与 thinker 的 prefill 策略解耦，且天然支持多请求共享只读权重。
3. 9 行 assistant 布局是 HF 兼容性的锚点；`projected[4:]` 的切分位置与
   `[8]` 槽位的"第 4 token"必须严格对应，错一行整个语音节奏就乱了。
4. PendingTextTensorQueue 的零拷贝升块 + O(1) peek 是 decode 热路径的微观优化，
   但其正确性约束（hidden 维一致、设备/dtype 就地转换）同样严格。
5. EOS 行的"预扣后补"（include_assistant_eos / mark_thinker_done）保证了
   "thinker 未结束就不发语音 EOS"这一语义。

下一篇（08）：talker 的前向、code predictor 与每步解码交接的完整闭环。
