# 08 Talker 模型与解码交接：QwenTalkerModelRunner 的自反馈闭环

> 主角：`talker_model_runner.py`（503 行）、`components/talker.py`（1811 行，本篇取前向/
   采样/code predictor 部分）、`model_runner/base.py` 的 decode 钩子。

---

## 1. talker 前向的两种形态（talker.py forward:1235-1305）

```python
def forward(self, input_ids, positions, forward_batch, input_embeds=None,
            input_embeds_are_projected=False, ...):
    if forward_batch.forward_mode.is_extend():
        self.invalidate_decode_buffers()          # prefill 采样绕过 _sampled_token_ids，
                                                  # 快速路径必须失效（akazaakane 注释）
    if input_embeds is not None and not input_embeds_are_projected:
        input_embeds = self.prepare_input_embeds(...)      # 双轨投影（见下）
    positions = mrope_positions if self._uses_mrope else forward_batch.positions
    hidden = self.model(input_ids, positions, forward_batch, input_embeds=...)
    if extend and input_embeds is not None:
        return self._manual_extend_logits(hidden, forward_batch)   # prefill：只取末位 logits
    logits_output = self._manual_decode_logits(hidden)             # decode：直接 codec_head
    if forward_batch.forward_mode.is_decode():
        sampled = self._sample_decode_tokens(logits_output.next_token_logits, forward_batch)
        self._sampled_token_ids[:bs].copy_(sampled)                # 固定地址缓冲
        self.code_predictor_forward(sampled.unsqueeze(1), hidden.unsqueeze(1))
    return logits_output
```

注意三个"SGLang 旁路"：

1. **手写 logits**：`_manual_extend_logits` 只对每个请求的最后一个位置取
   `codec_head`（`cumsum(extend_seq_lens) - 1` 找末位），绕开通用 LogitsProcessor
   （注释：该路径在投影 prefill 的 extend batch 上会挂）；`_manual_decode_logits`
   则因为 decode 无需 gather，直接过 head。
2. **手写采样**：`_sample_decode_tokens` 自己做 repetition penalty
   （`logits>0 ? logits/p : logits*p`，作用在 `_repetition_mask` 标过的
   历史码位上）、suppress mask（`-inf`），再交 SGLang sampler 或 argmax。
3. **自产自销**：decode 步采样结果不回调度器再进 predictor，而是**在同一前向里**
   直接喂 `code_predictor_forward`，把残差码与求和 embedding 写进固定缓冲
   `_output_codes / _output_embeds`。

### 1.1 双轨投影（prepare_input_embeds:1130-1155）

```python
if thinker_hidden_states is None or is_multimodal_mask is None:
    return self.text_projection(thinker_embeds)          # 全文本轨
if thinker_embeds is None:
    return self.hidden_projection(thinker_hidden_states) # 全隐藏轨
# 混合：mask 分派（07 篇的 mask 语义在此落地）
```

prefill 侧 deepstack（`input_deepstack_embeds` + mask）进 talker 时走的正是这条混合路径；
07 篇的 prompt 重建已把行预投影（`input_embeds_are_projected=True`），forward 跳过投影。

### 1.2 静态缓冲与采样参数分段上载（prepare_decode_buffers:1053-1128）

`Qwen3OmniTalker.__init__` 为 `max_running_requests` 预分配了一整组固定地址缓冲：
`_repetition_mask / _suppress_mask / _repetition_penalties / _sampling_temperatures /
top_ps / top_ks / min_ps / _sampling_seeds / _output_codes / _output_embeds / …`。

`prepare_decode_buffers(requests)`（`before_decode` 钩子调用，08 §2）把每请求的
采样参数写进缓冲，有两个微优化值得复述：

- **pinned staging + 位重解释**：6×B 的 int64 CPU pinned 缓冲，
  行 0-3 用 `.view(torch.float64)` 位重解释放 float 参数（penalty/temperature/top_p/min_p），
  行 4-5 放 int（top_k/seed）——一次 `copy_(non_blocking)` + event 就把全部参数上 GPU，
  省掉 6 次 H2D。注释强调 staging 必须 `device="cpu"`：模型 `__init__` 在 CUDA 默认
  设备上下文里执行，只有 CPU 张量能 pinned。
- **快速路径 `_reuse_decode_buffers`**：同一批 rid 且各请求输出长度恰好 +1
  （= 常规 decode 连续步）时，只把上一步采出的 token 补进 repetition mask
  （`_repetition_mask[rep_rows, _sampled_token_ids[rep_rows]] = True`），其余参数不动。
  `_decode_prep_rids/_decode_prep_out_lens` 是路径有效的证据链；
  `invalidate_decode_buffers` 在 prefill 后强制失效。

---

## 2. QwenTalkerModelRunner：decode 前后的两次交接

### 2.1 before_decode（talker_model_runner.py:57-75）

```python
def before_decode(self, forward_batch, schedule_batch, requests, *, is_lookahead=False):
    if not self._feedback_enabled: return
    if not self._requests_ready_for_decode(requests):
        raise RuntimeError("Talker decode reached model runner without ready feedback/text input")
    self.model.prepare_decode_buffers(requests)
    self._write_feedback_buffers(requests)
```

`_requests_ready_for_decode` 检查每请求的 `_data_has_next_decode_input`：
`pending_feedback_queue` 非空 **且**（`pending_text_queue` 非空 **或**
(`thinker_chunks_done` 且 `tts_pad_embed` 就位)）。02 篇的
`_is_batch_ready_to_run` 用同一谓词把不就绪的 decode 批整个推迟——
这里再断言一次属于"防御性双保险"。

`_write_feedback_buffers` 把每请求的下一步输入写进模型的
`_feedback_buffer[:bs] / _feedback_mask[:bs]`：

```python
for row_idx, sched_req in enumerate(requests):
    combined = self._take_next_decode_input_embed(sched_req=sched_req, device, dtype)
    ...
if len(rows) == batch_size:      # 稠密稳态：切片赋值，避免 per-frame 可分页索引 H2D
    feedback_buffer[:bs] = embeds_stacked; feedback_mask[:bs] = True
else:
    rows_t = torch.tensor(rows, device=...)   # 稀疏才走索引上传
    feedback_buffer[rows_t] = embeds_stacked; feedback_mask[rows_t] = True
```

### 2.2 下一输入的合成（_combine_feedback_with_next_text:437-460）

```python
feedback = peek(pending_feedback_queue)          # 上一步 codec 求和 embedding
combined = feedback                              # 必须存在，否则整体返回 None
next_text = peek(pending_text_queue)
if next_text is None:
    if not data.thinker_chunks_done: return None  # 文本还没来且没说完 → 不可解
    next_text = data.tts_pad_embed                # 说完了 → pad 填充行
return combined + next_text                       # 相加（两个投影空间的和）
```

**`feedback + text` 的加法是 talker 的核心数学**：codec 轨的上一帧求和 embedding
加上文本轨的下一行投影，作为本步输入。这与 prefill 布局的"加性混合"完全同构
（07 篇 9 行布局就是 `text_hidden + codec_hidden`）。两条 FIFO 各弹一行
（`_take_next_decode_input_embed` 里 pop），保持严格同步。

弹出的行同时 `append` 进 `decode_input_embeds` 历史——这是为 **retract（回退重跑）**
准备的：`_generated_prefill_slice` 在 decode 请求被重新 prefill 时从历史重放
已生成的行，缺失则报错（"Cannot replay retracted talker decode tokens"），
绝不静默编造。

### 2.3 post_decode / post_prefill：码帧出口（talker_model_runner.py:77-140）

```python
def post_decode(self, result, forward_batch, schedule_batch, requests):
    batch_size = len(requests)
    result.next_token_ids = self.model._sampled_token_ids[:batch_size].clone()
    self._stage_token_ids(result, result.next_token_ids)
    self._emit_code_chunks_and_feedback(...)
```

`_emit_code_chunks_and_feedback` 是自反馈回路的出口：

```python
codes_snap  = self.model._output_codes[:bs].detach().clone()    # [bs, num_code_groups]
embeds_snap = self.model._output_embeds[:bs].detach().clone()   # [bs, hidden]
for idx, sched_req in enumerate(requests):
    is_streaming = params.get("stream", False)
    self._outbox.put(OutgoingMessage(request_id, "stream",
        data=code_chunk, target="code2wav", metadata={"stream": is_streaming}))
    sched_req.data.pending_feedback_queue.append(feedback_row)
```

**为什么必须 clone**（wenyao 注释）：`_output_codes/_output_embeds` 是固定地址缓冲，
下一帧（可能在 CUDA graph 内）会原地覆写；快照必须是新分配，否则发给 code2wav 的
张量会被后续帧改写。这是"图内写固定地址 + 图外消费"的经典竞态。

`post_prefill` 的差异：layer0 码直接取 `result.next_token_ids`
（prefill 采样结果），talker_hidden 取 `result.logits_output.hidden_states`
（捕获层），走同样的 predictor + emit。注释强调**不要清
`data.prefill_input_embeds`**——decode retract 可能把请求重新排队再 prefill，
而 `Req.input_embeds` 是 None。

---

## 3. Code Predictor：残差 RVQ 的增量生成

### 3.1 结构

`Qwen3OmniMoeTalkerCodePredictor` 是一个小型 decoder LM，带
`num_code_groups` 个独立的 `lm_head` 与逐组 `codec_embedding`。
talker 主干只产**第 0 组**码（layer0 code），其余 `num_code_groups - 1` 组
由 predictor 按序贪心生成（`_sample_code_predictor_token`：argmax，
注释：匹配 HF `generate(do_sample=False)`）。

### 3.2 增量算法（_code_predictor_forward_incremental_eager:1550-1607）

对每个位置 `pos`：

```
输入序列构造（predictor_input[:, 0:2, :]）：
  [0] = talker_hidden[pos]        ← talker 主干的隐藏态
  [1] = embedding(layer0_code)    ← 主干采出的第 0 组码
result_codes[:, :, 0] = layer0_code
pos_summed = layer0_embed          ← 求和 embedding 的起点

cache_len = 0
last_hidden = predictor_forward_one_token(输入[0])   cache_len=1   ← 先吃隐藏态
last_hidden = predictor_forward_one_token(输入[1])   cache_len=2   ← 再吃 layer0 码
for group in 1 .. num_groups-1:
    logits = lm_head[group-1](last_hidden)
    code = argmax(logits)
    result_codes[:, :, group] = code
    new_embed = codec_embedding[group-1](code)
    pos_summed += new_embed                            ← 求和 embedding 累加
    if group < num_groups-1:
        last_hidden = predictor_forward_one_token(new_embed)  cache_len+=1
```

`_predictor_forward_one_token` + `_predictor_cached_self_attention` 是手写的
单 token 前向：**每层一个单回合 KV cache**（`_predictor_k_cache/_v_cache[layer_idx,
:bs, :, cache_len+1, :]`），SDPA 对 cached 前缀做非因果 attention。
KV cache 位置用 `_predictor_positions = arange(num_code_groups+1)`——
predictor 的序列长度上限就是"组数+1"，无需 paged attention。

### 3.3 predictor 的 CUDA 图（_PredictorDecodeGraph, talker.py:52-110）

seq_len==1 且张量在 CUDA 时走图：按 `(bucket_size, code_dtype)` 捕获/复用，
bucket 集合来自 `get_decode_cuda_graph_bs(server_args)`（归一化后保证包含
max_running_requests）。replay 前 `layer0_codes[live:].zero_()` 补零填充。
捕获失败（任意异常）→ 记入 `_predictor_decode_graph_disabled`，永久回 eager。
`_can_use_predictor_decode_graph` 里有一条 `torch.cuda.is_current_stream_capturing()`
守卫——**外层 talker 图捕获期间绝不能再触发内层图捕获**。

---

## 4. MoE 细节：共享专家的统一 all-reduce（talker.py:230-293）

`Qwen3OmniMoeTalkerSparseMoeBlock` 继承 thinker 的 MoE（路由专家部分），
新增 shared expert：

```python
shared_output = self.shared_expert(x)                    # reduce_results=False！
shared_output = shared_output * sigmoid(shared_gate(x))   # 门控
routed_output = self.experts(x, topk_output)
final = routed_output + shared_output                     # TP>1 且未融合时：
if ...: final = tensor_model_parallel_all_reduce(final)   # 只做一次 all-reduce
```

要点：shared expert 的 `down_proj` `reduce_results=False`，与路由专家的
部分和相加后**统一做一次 all-reduce**——省一半通信。还有个顺序性注释：
shared 分支必须在路由专家**之前**消费原始输入，因为 fused MoE 实现会原地改写
`hidden_states`。

DecoderLayer 直接继承 thinker 层、只换 mlp（`Qwen3OmniMoeTalkerDecoderLayer`）；
`Qwen3OmniMoeTalkerTextModel` 另有一套**手工 prefill 前向**
（talker.py:700-760，`_direct_self_attention`：qkv 分裂 + qk norm + rotary + SDPA
`is_causal=True, enable_gqa=...`），用于需要精确对齐 HF 的路径。

---

## 5. 权重加载（load_weights:1740-1810）

- `talker.` 前缀剥离；`thinker./code2wav.` 前缀直接跳过（负负得正：同一份
  checkpoint 被三个引擎分别加载）；
- 三段式映射：stacked（qkv_proj/gate_up_proj 拆分）→ MoE expert 参数
  （`FusedMoE.make_expert_params_mapping`）→ 直接匹配；
- `get_weight_preprocessor(root_config, fp8_scale_inverted=True)` 处理 FP8 scale
  的方向差异。

---

## 6. 小结：一次 talker decode 步的完整时间线

```
OmniScheduler.get_next_batch_to_run
  └─ is_decode_batch_ready? (每请求 feedback+text 就绪)   否 → 回滚 KV 分配，跳过本步
        │ 是
        ▼
before_decode: prepare_decode_buffers(参数分载/复用) + _write_feedback_buffers
  └─ combined = feedback + text   （两条设备端 FIFO 各 pop 一行，写 _feedback_buffer）
        ▼
forward(hidden=..., input_embeds=None)
  ├─ codec_head → logits
  ├─ _sample_decode_tokens（repetition penalty + suppress + sampler/argmax）
  ├─ _sampled_token_ids[:bs] ← sampled          （固定缓冲）
  └─ code_predictor_forward(sampled, hidden)     （图或 eager，残差码 + 求和 embedding）
        ▼
post_decode: result.next_token_ids ← _sampled_token_ids 克隆
  └─ _emit_code_chunks_and_feedback
       ├─ outbox → code2wav (stream, codes 帧快照)
       └─ pending_feedback_queue ← embeds 帧快照   ←→ 下一轮 before_decode
```

闭环成立的三个支点：**就绪谓词**（feedback+text 同步）、**快照克隆**
（对抗固定地址覆写）、**回滚对称性**（02 篇 §3.3 的 KV 回滚）。

下一篇（09）：码帧离开 talker 后，在 code2wav 里如何变成连续的音频。
