# 06 Thinker 引擎与多模态嵌入注入：从 StagePayload 到 SGLang 前向

> 主角：`model_runner/model_worker.py`、`model_runner/base.py`（钩子契约）、
> `model_runner/thinker_model_runner.py`（490 行）、`model_runner/_hidden_capture.py`、
> `models/qwen3_omni/bootstrap.py`（267 行）、`model_runner/prefill_inputs.py`、
> `models/qwen3_omni/thinker_model_runner.py`（prefill sidecar 适配）。

---

## 1. ModelWorker：把"子模型"伪装成完整 LLM

SGLang 的 ModelConfig 天生服务单模型。Qwen 的 thinker/talker 都藏在
`Qwen3OmniMoeForConditionalGeneration` 这一个 HF config 里，于是 `ModelWorker` 做了
**架构覆盖**（model_worker.py:40-121）：

```python
_ARCH_CONFIG_MAP = {
    "Qwen3OmniTalker":         ("talker_config",  "text_config"),
    "Qwen3OmniThinkerForCausalLM": ("thinker_config", "text_config"),
    ...
}
def _apply_arch_override(model_config, arch):
    model_config.hf_config.architectures = [arch]
    sub_cfg = getattr(hf_config, sub_config_attr)          # thinker_config / talker_config
    text_cfg = getattr(sub_cfg, text_config_attr)
    model_config.hf_text_config = text_cfg
    model_config.num_attention_heads / num_kv_heads / hidden_size / num_hidden_layers = ...
```

效果：SGLang 全链路（KV pool 尺寸、CUDA graph、采样器、attention backend）
都按**子模型**的真实形状构建；权重的 `talker.` / `thinker.` 前缀由
`weight_prefix` 参数在加载时剥离（`stages.create_talker_ar_executor_from_config(weight_prefix="talker.")`）。
talker 的 vocab 也在 bootstrap 里被改写成 **codec vocab**
（`model_config.vocab_size = talker_config.text_config.vocab_size`，
注释说明 repetition-penalty orchestrator 会按 vocab_size 定尺寸，用文本词表会错位）。

量化策略链：`_apply_omni_quantization_adapters`（checkpoint 名归一）→
公共策略 → 平台策略。这让 AR 阶段可以原生跑 FP8（见 03 篇 §6）。

---

## 2. bootstrap.py：两个 scheduler 的装配差异

`create_thinker_scheduler` vs `create_talker_scheduler` 的差异表：

| 装配项 | thinker | talker |
|--------|---------|--------|
| arch override | `Qwen3OmniThinkerForCausalLM` | `Qwen3OmniTalker`（prefix `talker.`） |
| capture_hidden_layers | `[0, 24]`（speech 时） | 无 |
| defer cuda graph | speech 时 defer（先装 omni 包装再捕获） | want_cuda_graph 时 defer + 补跑 init |
| prefill input_embeds 槽 | BREAKABLE prefill-graph backend 时 enable | 不开（decode 图 + 手写 logits） |
| output processor | `capture_hidden=True, layers=[0,24]` + `should_emit_hidden` | `capture_hidden=False` |
| model runner | `Qwen3OmniThinkerModelRunner` 或 `ThinkerModelRunner` | `QwenTalkerModelRunner` |
| vocab | 文本 vocab | **codec vocab**（bootstrap 内改写） |
| 额外适配器 | — | `append_text_chunk` / `mark_thinker_done` 两个流处理回调 |

一个容易忽略的注释：**defer 捕获也必须装 omni 包装**
（`bootstrap.py:63-66`）——否则 prefill embeds 的 view 永远不会被应用，
图的 `input_embeds` 槽是否存在将取决于"模型 config 恰好是不是多模态"，
这是典型的隐式耦合。

---

## 3. ModelRunner 钩子契约（base.py）

`ModelRunner`（`model_runner/base.py`，1007 行）把 SGLang 的 model_runner 前向流程
切成九个可选钩子：

```
before_prefill(forward_batch, schedule_batch, requests)
custom_prefill_forward(...) -> 可选接管整个 prefill 前向
sample_before_post_prefill(...) -> bool
post_prefill(result, ...)
before_decode(..., is_lookahead)
sample_before_post_decode(...) -> bool
post_decode(result, ...)
is_decode_batch_ready(schedule_batch) -> bool
execute(scheduler_output) / bind / outbox 注入
```

thinker 只用前半部分；talker 用后半部分（08 篇）。钩子的执行点在
SGLang batch 生命周期的固定位置，OmniScheduler 的 `_run_batch` 驱动它们。

---

## 4. 多模态嵌入注入（thinker_model_runner.py:77-320）

这是 thinker 的灵魂函数 `_inject_multimodal_embeds`。SGLang 的 prefill 是
**chunked** 的（8192/块），一次前向只覆盖 prompt 的一个区间
`[prefix, prefix+extend_len)`。注入的任务：把该区间内的占位符 token 换成
**对应模态 embedding 的连续切片**。

### 4.1 算法骨架

```python
# 1) 全量 embedding 查表（pad 值被 clamp 进合法范围，占位符行的 embedding 会被覆盖，无所谓）
input_embeds = self._embed_tokens(forward_batch.input_ids.clamp(0, num_embeddings-1))

# 2) 每请求的区间游标
for i, req in enumerate(schedule_batch.reqs):
    start = offsets[i]                  # 本 chunk 在整批中的行起点
    prefix = prefix_lens[i]             # 本请求已进 KV 的长度（radix 复用后可能 >0）
    consumed = req._omni_consumed       # 每模态已消费的 embedding 行数（跨 chunk 持久）

    positions = self._req_mm_token_positions(req, pad_values)   # CPU int64，prompt 绝对位置
    for modality in ("image", "video", "audio"):
        rel, offset, n_tokens = self._plan_modality_chunk(
            positions[modality], consumed, modality, prefix, length)
        # rel = 落在 [prefix, prefix+length) 的占位符相对位置
        # offset/consumed[modality] = 该模态 embedding 的读游标
        embeds = omni_inputs.get(f"{modality}_embeds")
        if n_tokens:
            chunk_embeds = embeds[offset : offset + n_tokens]
            scatter_rows.append(rel + start)          # 写入位置
            scatter_srcs.append(chunk_embeds)         # 写入内容
            consumed[modality] = offset + n_tokens
```

要点：

- **游标协议**：`_plan_modality_chunk` 是纯函数（不 mutate），
  `_validate_modality_cursor` 做区间断言（offset 非负、`offset+n ≤ row_count`），
  `_reconstruct_missing_cursor` 处理"radix 前缀隐藏了早前音频行但游标丢失"的场景——
  只有当前缀内缓存行数与 embedding 行数能一一对应时才允许重建，否则拒绝。
- **deepstack 合并**：图像/视频的多尺度残差嵌入要按 prompt 顺序交错
  （图像占位与视频占位可能同段出现）。实现用 `torch.sort(cat([img_pos, vid_pos]))`
  得到 `visual_order`，再用 `slots[visual_order] = arange` 的逆排列把两个模态的
  层切片填进同一个 `joint` 缓冲——全程无设备同步（注释：unique 的模态位置保证
  排序无 tie）。
- **消费完即卸载**：`if req.inflight_middle_chunks == 0: req.omni_model_inputs = None`——
  最后一个 chunk 吃完就把 embedding 引用还给 GC。

### 4.2 scatter 与前向分派

```python
if ds_embeds is None:
    forward_batch.input_embeds = input_embeds      # 普通路径：SGLang 自己处理
    return None
return self._forward_with_omni_embeds(forward_batch, input_embeds, ds_embeds, vis_masks)
```

deepstack 残差是**层间注入**（特定 decoder 层的输出上加视觉残差），
`ForwardBatch` 没有这个概念，所以必须走 custom forward（`_forward_with_omni_embeds`）。
没有 deepstack 时只设置 `forward_batch.input_embeds`，让 SGLang 原生路径接管。

### 4.3 capture hidden 与 CUDA graph 的兼容

`requested_capture_hidden_mode_prefill/decode` 强制返回 `CaptureHiddenMode.NULL`
（注释：thinker 的流式隐藏态靠本地 forward hook 捕获；SGLang 的 LAST 模式会禁用
graph replay）。所以隐藏态捕获改由 `_hidden_capture.py` 的**静态 buffer + pre-hook**：

```python
text_model.register_buffer(f"_omni_aux_hidden_layer_{L}",
    embedding_weight.new_empty((max_tokens, hidden_size)), persistent=False)
text_model.layers[L].register_forward_pre_hook(_layer_input_capture_hook(buffer, max_tokens))
```

- 层 0 的 hook 捕获"第一个 decoder 层的输入" = embedding 输出（这正是 talker 需要的
  `embed` 层）；层 24 捕获第 24 层输入 = 23 层输出（talker 的 `accept_hidden_layer`）。
- hook 里 `layer_input = hidden_states + residual`（pre-hook 拿到的是残差流输入）。
- buffer 是**非持久化**的固定地址显存，graph replay 会自动刷新内容，
  Python 侧零簿记（`StaticAuxHiddenCapture.views(num_rows)` 只做切片）。
- `text_model.layers_to_capture = []`：显式关掉上游的 tuple 返回捕获路径，
  保持内层模型单张量返回。

---

## 5. prefill sidecar：Qwen3OmniThinkerModelRunner（models/…/thinker_model_runner.py）

这个 324 行的类是"共享 prefill sidecar"的 Qwen 适配器，目标：**把纯文本/纯音频
prefill 纳入 SGLang 的 breakable prefill CUDA graph**。

`_classify_prefill` 返回三态：

- `_SIDECAR`（无 mm 输入）→ 直接 `attach_omni_prefill_inputs(input_embeds=embed_tokens(input_ids))`；
- `_SIDECAR, has_audio=True` → 音频也要满足苛刻的资格检查
  （`_audio_inputs_are_supported`：model_inputs 键集合恰好是
  `{audio_embeds, audio_feature_lengths, feature_attention_mask, pad_values}`、
  embeds 形状/维度合法、**只有 audio 一个模态**（`positions["image"].numel() == 0`
  等硬条件）、游标可规划、`inflight_middle_chunks ≥ 0`）→ 注入后走 sidecar 图；
- `_UNSUPPORTED` → 交给父类 eager 多模态路径（deepstack、视频、AIV 都在此列）。

设计哲学写在 docstring 里：**这个类只做"资格审查"，图的准入、bucket 选择、padding、
replay 元数据、eager 回退全部归 SGLang 管**。资格不过就退回 eager，
绝不为了走图而放宽正确性（`custom_prefill_forward` 里若 sidecar 被分类出来却没被
attach，直接 RuntimeError——宁可炸也不静默错）。

---

## 6. 流输出的生产：make_thinker_stream_output_builder

02 篇已列三条规则，这里补全数据形状：

```python
messages.append(OutgoingMessage(request_id, "stream",
    data=torch.tensor([token_id], dtype=torch.long),   # 流传输只收张量，int 要包一层
    target="decode", metadata={"token_id": token_id}))
messages.append(OutgoingMessage(request_id, "stream",
    data=hidden,                                        # [hidden_size] 1D
    target="talker_ar", metadata={"token_id": token_id}))
```

隐藏态的拆解（`_split_dual_layer_hidden`）：
dict 形态 `{"embed"(或 0/"0"): layer0, 其余任意键: layer24}` →
`(embed, layer_hidden)`；talker 侧 `accept_hidden_layer=24` 会消费后者
（`required_aux_hidden_key`，07 篇）。

`should_emit_hidden` 由 output processor 持有（`should_generate_audio_output(stage_payload)`），
从请求元数据 `output_modalities` 读取——**不需要音频的请求一个隐藏态都不发**，
talker 连被通知的代价都不用付。

---

## 7. result_adapter：thinker 输出的归一化

`apply_thinker_result`（request_builders.py:479-505）把 SGLang 的 req_data 变回
`ThinkerOutput{output_ids, step, is_final=True, extra_model_outputs, finish_reason?,
weight_version?, output_token_logprobs?}`，同时写进 `state.thinker_out` 与
`state.engine_outputs["thinker"]`。终态 payload 沿 `next="decode"` 经
`project_thinker_to_decode`（清输入与 extra）到达 decode 阶段——10 篇会看到
decode 如何用"流里已经发过的 token"与"payload 里的终态"双通道对齐。

---

## 8. 小结

1. arch override 是"**用别人的引擎跑我的子模型**"的全部秘密；
2. 注入 = 全量查表 + 每 chunk scatter，游标协议（plan/validate/reconstruct）
   保证 chunked prefill + radix 复用 + 多模态三者的正确性交集；
3. deepstack 迫使部分 prefill 走 custom forward，这是无法用
   `forward_batch.input_embeds` 表达的层间注入；
4. 隐藏态捕获用静态 buffer + pre-hook，与 CUDA graph 共生而非对抗；
5. sidecar 的价值取向：**资格从严，失败即回退**——图加速永远不能买走正确性。

下一篇（07）：这些每步隐藏态如何变成 talker 的输入流，以及部分启动的完整机制。
