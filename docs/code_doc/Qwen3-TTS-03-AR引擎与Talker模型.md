# Qwen3-TTS 深度解析(三):AR 引擎与 Talker 模型

> 本篇覆盖 `sglang_model.py` 的模型结构部分(前 ~1100 行):`Qwen3TTSTalker`、`Qwen3TTSTalkerTextModel`、`Qwen3TTSTalkerDecoderLayer`、三种 prompt 构建、持久缓冲区、权重加载。预测器与 CUDA 图在 04 篇。

---

## 1. 类层次:从检查点到 SGLang 原生模型

```
检查点 config (qwen_tts Qwen3TTSConfig)
  └── root_config (含 talker_config / speaker_encoder_config / tts_*_token_id / tts_model_type)
        └── Qwen3TTSTalker (EntryClass,SGLang 按 model_arch_override 实例化)
              ├── text_projection      ResizeMLP(text_hidden → hidden)     # 文本空间→说话人空间
              ├── model                Qwen3TTSTalkerTextModel              # AR 主干
              │     ├── codec_embedding    Embedding(vocab_size, hidden)    # 层0码本 + 控制token
              │     ├── text_embedding     Embedding(text_vocab, text_hidden) # 原文本 vocab
              │     ├── layers[N]          Qwen3TTSTalkerDecoderLayer       # 复用 omni 注意力
              │     └── norm               RMSNorm
              ├── codec_head           ReplicatedLinear(hidden → vocab)     # 层0 logits
              ├── code_predictor       Qwen3TTSCodePredictor                # 其余 Q-1 层(04 篇)
              ├── speaker_encoder      Qwen3TTSSpeakerEncoder(仅 base)      # x-vector 提取
              └── speech_tokenizer     qwen_tts.Qwen3TTSTokenizer(运行期挂载)
```

`__init__` 第一行的解包:`config.talker_config` 存在则 `root_config=config; config=config.talker_config`——上游 config 是两层结构,SGLang 侧全部操作指向内层 talker 配置。`tts_model_type`("base"/"custom_voice"/"voice_design")从 root_config 读出,决定 speaker_encoder 是否实例化(非 Base 检查点没有该子模块,`self.speaker_encoder = None`)。

**与 Omni talker 的直接血缘**:`Qwen3TTSTalkerDecoderLayer` 内部就是 `Qwen3OmniMoeThinkerTextAttention`(从 `qwen3_omni/components/thinker_model.py` 导入)+ `Qwen3OmniMoeTalkerDenseMLP` + 两个 RMSNorm。Omni 的 talker 层是 MoE(`Qwen3OmniMoeTalkerSparseMoeBlock` + shared expert),TTS 换成 DenseMLP——这是两个模型容量设计的差异:TTS-1.7B 用 dense 就够,Omni talker 用的 MoE 是从更大的共享骨干继承的。`_bind_default_weight_loaders(self)` 同样来自 omni 组件,负责给普通参数绑定默认的 weight_loader(处理 TP 切分逻辑)。

## 2. 解码层与"可断图"的图断点

`Qwen3TTSTalkerDecoderLayer.__init__` 末尾:

```python
_install_breakable_prefill_qk_norm_rope_graph_break(self.self_attn)
```

该函数(`sglang_model.py:52-54`)把 `attention.apply_qk_norm_rope` 包上 `eager_on_graph(True)`——**在 breakable prefill CUDA graph 里,packed QK-norm + RoPE 这一小段强制 eager 执行**,而周边的 QKV 投影、o_proj、MLP 仍被捕获。注释解释原因:捕获这一小块会**破坏 Qwen3-TTS 的 prefill 图重放**(形状随 prompt 变化剧烈的部分保留动态)。这是 SGLang "可断 CUDA 图"能力的典型应用:粒度从整图细化到段,模型作者按算子特性选择捕获边界。解码阶段不经过 breakable-prefill 上下文,行为不变。

`forward`(`sglang_model.py:1005`):

```python
if forward_batch.mrope_positions is not None:
    positions = forward_batch.mrope_positions          # 替换为 3D mrope
hidden_states = self.model(...)
if forward_batch.forward_mode.is_extend():
    last_index = self._extend_last_index(...)          # 每请求取最后一个位置
    hidden_states = hidden_states[last_index]
logits, _ = self.codec_head(hidden_states)
return LogitsProcessorOutput(next_token_logits=logits, hidden_states=hidden_states)
```

三个细节:

1. **`is_mrope_enabled = True` 是一个协议标记**:类注释说明,外层 forward 负责"遇 mrope_positions 则替换",而 prefill CUDA graph runner 捕获的是**内层 text model**(不走这个替换),用该标记维持位置契约。这与 Omni talker 的 `self._uses_mrope` 检查同源,但 Omni 是真 3D M-RoPE(文本/音频/图像各自维度),TTS 的 talker 位置本质是 1D——接受 mrope_positions 只是 SGLang batch 准备的兼容形态。
2. **extend 时取每请求最后位置**:prefill 一次前向多个请求拼接,`torch.cumsum(extend_seq_lens) - 1` 取各行末位——SGLang 标准的 last-token gather,只要 last hidden 参与 logits。
3. **返回的 `hidden_states` 是末位 hidden**(prefill)或全量 hidden(decode,形状 [B,1,H]),它被 runner 的 `_collect_codes` 直接喂给预测器(04 篇)——**hidden state 是层0码之外的第二个条件源**,这是双流设计的核心。

## 3. 输入嵌入的三条通道:`Qwen3TTSTalkerTextModel.forward`

```python
def forward(self, input_ids, positions, forward_batch, input_embeds=None):
    if input_embeds is None:
        if is_decode:
            hidden_states = self._decode_feedback_embedding(input_ids)   # 通道②
        else:
            hidden_states = self._build_input_hidden_states(input_ids)   # 通道①
    else:
        hidden_states = input_embeds                                     # 通道③(prefill 注入)
```

**通道①:prefill 的反馈混合。**

```python
def _build_input_hidden_states(self, input_ids):
    hidden_states = self.codec_embedding(input_ids)
    feedback_mask = self._feedback_mask[:bs]
    return torch.where(feedback_mask.unsqueeze(-1),
                       self._feedback_buffer[:bs].to(dtype), hidden_states)
```

`_feedback_buffer [max_batch_size, hidden]` + `_feedback_mask [max_batch_size]` 是**按 batch 行**的全局缓冲:`feedback_mask[row]=True` 的位置,该行 embedding 被缓冲内容替换。用途:预填充时,若某行需要以"上一说话人输出"作为输入(如 ICL prompt 里的 ref codec 段已求和为单行向量),构建期直接把该向量写进缓冲、开 mask——**避免为混合输入再造一个 embedding 索引协议**。

**通道②:decode 的"行号寻址"嵌入。**

`_decode_feedback_embedding = nn.Embedding(max_batch_size, hidden)`——一个**行号→embedding** 的表!decode 时 `input_ids[row]` 不是 token,而是行号;前向先查表得到该行实际输入 embedding。为什么绕这一圈?**CUDA 图解码要求输入形状与地址固定**:真正的 embedding 查表依赖 input_ids 内容,而 input_ids 每步都变——runner 在 `before_decode` 把真实输入 embedding 写进 `_decode_feedback_embedding` 的对应行,再把行号写进 `input_ids`(`model_runner.py:_write_feedback_buffers` 末尾 `input_ids[:batch_size].copy_(row_ids)`)。图内执行 `embedding(row_ids)` 读到的是**固定地址里已更新的内容**——标准 CUDA graph 输入暂存技巧,与 vLLM/SGLang 用持久 buffer 暂存 logits 同构。**此设计直接继承自 Omni talker**(其 `_decode_feedback_embedding` 同名同构),是两个 talker 最深的共同骨架。

**通道③:prefill 的显式 embedding 注入**——由 runner 通过 `attach_omni_prefill_inputs` 提供 `input_embeds`,模型直接使用,不做 codec_embedding 查表。TTS 的 prefill 全部走这条(prompt 是构建期算好的 embedding 序列)。

## 4. Prompt 组装的完整数学(核心章节)

三种任务的 prompt 都是 embedding 序列,以最复杂的 **voice clone(ICL 模式)** 为例逐段拆。记号:`TE(x)=text_projection(text_embedding(x))`(文本→hidden 空间),`CE(id)=codec_embedding(id)`(codec 空间),特殊向量 `tts_bos/tts_eos/tts_pad` 由 `_build_tts_special_embeds` 从 `tts_*_token_id` 经 TE 得到。

### 4.1 codec 前导(`_build_codec_prefill`)

```
language="auto":  [CE(codec_nothink), CE(think_bos),                CE(think_eos)]
language=L:       [CE(codec_think),   CE(think_bos), CE(L_id),      CE(think_eos)]
```

这是一段"任务声明头":think/nothink 开关 + 语言 id。CustomVoice 且 `spk_is_dialect[voice]` 存在时,即使请求 language="auto" 也会被该音色的方言覆盖(`_resolve_language_id`)。

### 4.2 说话人条件段

```
speaker_embed = voice_clone_prompt["ref_spk_embedding"][0]     # [hidden] x-vector
codec_input = cat(codec_input_0, speaker_embed.view(1,1,-1), CE(pad), CE(codec_bos))
```

x-vector 作为**单个 codec 位置**插入;`CE(pad), CE(codec_bos)` 是生成段起始符。CustomVoice 用 `CE(spk_id_map[voice])` 替代 speaker_embed(表驱动,嵌入即码本里预训练的说话人 id 向量);VoiceDesign 则没有该段。

### 4.3 条件前缀(`_build_conditioned_prompt_prefix`)

```
role_embed = TE(input_id[:, :3])                          # <im_start>system/user 骨干,3 token
prompt_embed = cat([tts_pad.expand(·, Q-2, ·), tts_bos]) + codec_input[:, :-1]
prefix = cat([role_embed, prompt_embed], dim=1)
```

注意加法:**codec embedding + tts_pad 文本嵌入逐位相加**。这是 talker 的双流融合方式——每个位置同时携带"codec 层信息"与"文本层信息",相加即融合。Omni talker 的 prompt(`talker_input.py build_prefill_input`)是同一数学,但它把 thinker 的真实 hidden(`text_projection(thinker_embed)`)加到对应 codec 位置上,而 TTS 用静态的 pad/bos 文本嵌入——**Omni 的文本流是动态生成的 thinker 状态,TTS 的文本流是预置的常量向量**,这是"有无上游 thinker"在 prompt 数学上的直接投影。

### 4.4 ICL 段(`generate_icl_prompt`)——长度对齐三角

```
text_embed  = TE(cat([ref_id[:, 3:-2], text_id[:, 3:-5]]))     # 参考文本 + 待合成文本
text_embed  = cat([text_embed, tts_eos], dim=1)
codec_embed = Σ_k CE/refEmbed(ref_code[:, k])                   # 各层码嵌入求和
codec_embed = cat([CE(codec_bos), codec_embed], dim=1)          # 起始符
```

ref codec 是 [T_ref, Q],各层嵌入**按位求和压成 [T_ref, hidden]**(多码本帧→单向量);然后与文本嵌入做**长度对齐**:

```
if text_len > codec_len:  返回 (text[:codec_len] + codec,  text[codec_len:])   # 多余文本 → trailing
else:                     text 补 tts_pad 到 codec_len;返回 (text+codec, tts_pad)
```

- 前者的余量 `text_embed[:, codec_len:]` 就是 **`trailing_text_hidden`**——流式模式下逐 token 喂给 talker 的"未来文本"队列(§4.6)。
- 返回元组第二项是 trailing;ICL 首段是 `text+codec` 逐位相加。

x_vector_only_mode(ref_code=None)时跳过整段,`trailing_text_hidden` 来自 `_finish_text_prompt`。

### 4.5 文本收尾(`_finish_text_prompt`):流式 vs 非流式分野

**非流式(CustomVoice/VoiceDesign 强制,Base 可选)**:

```
text_all = TE(input_id[:, 3:-5]) + tts_eos
prefill = cat([prefix, text_all + CE(pad)×len, tts_pad + CE(codec_bos)])
trailing = tts_pad          # 空占位
```

全部文本一次性进 prefill;decode 期每步从 trailing 队列拿到的都是 `tts_pad`(恒定向量)。

**流式(Base 默认)**:

```
first_text = TE(input_id[:, 3:4]) + codec_last_embed       # 首个文本 token 与 codec 起始融合
prefill   = cat([prefix, first_text])
trailing  = cat([TE(input_id[:, 4:-5]), tts_eos], dim=1)   # 其余文本 → 队列
```

**为什么流式要这样切?** 首个文本 token 必须与 `codec_bos` 一起出现在 prefill 尾部(模型从该位置开始生成层0码),但后续文本还没被"消耗"——它们排队等每步解码时加到反馈向量上(§4.6)。这使 **prompt 长度与音频生成解耦**:文本多长都不增加 prefill,只增加队列深度。Omni 的流式文本来自 thinker 逐 chunk 投影(`TalkerPrefillBuilder.append_text_chunk`),机制相同但来源动态;TTS 的队列在预处理期一次性装满(`PendingTextTensorQueue.from_tensor(trailing_text_hidden)`,02 篇)。

### 4.6 decode 期的每步输入合成

runner 的 `_write_feedback_buffers`(06 篇)每步调用静态方法 `_take_next_decode_input_embed`(直接从 `QwenTalkerModelRunner` 复用):

```python
combined = feedback                      # 上一步预测器的 ΣQ 层嵌入快照(_output_embeds 行)
         + next_text                     # PendingTextTensorQueue.popleft(),队列空且 thinker 结束则 tts_pad
# 写入 _decode_feedback_embedding[row],input_ids[row] ← row
```

所以每步 talker 的真实输入 = **语音反馈(上一帧全部码层的嵌入和)+ 当前文本嵌入**。反馈闭环:`_output_embeds`(预测器输出,04 篇)→ runner 快照 → `pending_feedback_queue` → 下一步输入。TTS 与 Omni 的差异仅在 next_text 的来源(静态队列 vs thinker 流)。

### 4.7 instruct 前缀

`_apply_instruct_prefix` 把 `TE(wrapper._tokenize_texts("<|im_start|>user\n{instructions}<|im_end|>\n"))` 拼在**最前面**。VoiceDesign 必填(是唯一条件),CustomVoice/Base 可选。注意 instruct id 是 wrapper(`_build_instruct_text`)构建的文本 token 序列再走 text_projection——它占的是文本通道,不受 codec 流影响。

## 5. 持久缓冲区清单(全部预分配,图安全)

`Qwen3TTSTalker.__init__` 尾部按 `max_batch_size=server_args.max_running_requests` 预分配:

| 缓冲区 | 形状 | 用途 |
|---|---|---|
| `_predictor_k/v_cache` | [L_pred, B, kv_heads, Q+1, head_dim] | 预测器每 token 的 KV(04 篇) |
| `_predictor_positions / _position_rows` | [Q+1] / [Q+1, B] | 预测器位置(固定 0..Q,无位置外推) |
| `_sampled_token_ids` | [B] | 图内采样结果暂存 |
| `_output_codes` | [B, Q] | 本步全部码层(含层0) |
| `_output_embeds` | [B, hidden] | 本步 ΣQ 嵌入(=反馈向量) |
| `_predictor_embedding_buffer` | [B, hidden] | fused gather 目标(04 篇) |
| `_sub_*_tensor` 全家 | [B] | 子说话人采样参数暂存(温度/top_p/top_k/种子/do_sample/行号) |
| `_semantic_sampling_seed_tensor` | [B] | 语义层种子(runner 装进 sampling_info) |
| `_decode_feedback_embedding` | Embedding(B, hidden) | 行号→decode 输入(§3 通道②) |

`prepare_decode_buffers(requests)`(`sglang_model.py:1056`)是这些缓冲的**唯一暂存入口**,每次 decode 前由 runner 调用:

- **暂存去重**:以 `(request_id, prep_epoch)` 列表与上次比对,相同则整体跳过("每步静态值,批组成不变就复用")。epoch 机制:`data._qwen3_tts_prep_epoch` 首次分配时递增——request_id 可能被不同请求生命周期复用,id+epoch 才是真实身份。**对比 Omni** 的 `_reuse_decode_buffers`:Omni 用 `(rid, output_ids 长度恰 +1)` 判定可增量复用,并增量置位重复掩码;TTS 的参数完全静态,可整批跳过——同为暂存缓存,失效策略随参数可变性设计。
- **greedy 行的规范化**:`do_sample=False` 的行 temperature→1.0、top_p→1.0、top_k→1,注释说明动机:"greedy 行原 top_k 可能是 0 或 -1,否则会落入全排序分支"(05 篇全排序是慢路径,强制 top_k=1 走快路径且 argmax 数学等价)。
- **top-k 阶梯量化**:`_quantize_predictor_top_k(max_bounded_top_k)` 把批内最大 top_k 量化到 `(4,8,16,32,50,64,128,256,512,1024)` 梯级——**CUDA 图键只认梯级宽度**,per-row 掩码保证真实 k 生效(04/05 篇)。
- 采样参数经 CPU 中转张量一次 H2D;`sub_positions = semantic_pos × (Q-1) + layer_idx + 1`——子说话人采样的位置 = 语义位置 × 层数 + 层内偏移,**同一步内各层码用不同 Gumbel 噪声、跨步语义对齐**(05 篇哈希键)。

## 6. 权重加载

`load_weights`(`sglang_model.py:1759`):

```python
for name, loaded_weight in weights:
    if name.startswith("talker."):          target = name[7:]
    elif name.startswith("speaker_encoder."): target = name
    else: continue                           # 其余(text toaster 等)跳过
```

上游检查点把 talker 包在 `talker.` 前缀下;`speaker_encoder.` 整段透传(它是 qwen_tts 的模块,结构不变)。堆叠映射表处理 SGLang 融合参数:

```python
(".qkv_proj", ".q_proj", "q"), (".qkv_proj", ".k_proj", "k"), (".qkv_proj", ".v_proj", "v"),
("gate_up_proj", "gate_proj", 0), ("gate_up_proj", "up_proj", 1)
```

命中则调用 `param.weight_loader(param, loaded_weight, shard_id)` 走 SGLang 标准 TP 装载;否则按 named_parameters 缓存字典(`_cached_params_dict`,`__init__` 时构建,免得每请求遍历)直拷或走默认 loader。**预测器权重同样在这个循环里装载**(它的 `lm_head`/`codec_embedding`/`layers` 都带 `talker.` 前缀),不需要单独路径。

## 7. 与 Qwen3-Omni talker 的结构对照

| | Qwen3-TTS `Qwen3TTSTalker` | Omni `Qwen3OmniTalker` |
|---|---|---|
| 层实现 | DenseMLP + omni 注意力 | MoE(`SparseMoeBlock`+shared expert)+ omni 注意力 |
| 文本→talker 投影 | 单一 `text_projection` | `text_projection` + `hidden_projection` 双投影:文本位置走 text_projection,**多模态位置走 hidden_projection(thinker 中间层 hidden)**(deepstack) |
| prompt 构建 | 预处理期一次性(`build_*_inputs`,数学见 §4) | `TalkerPrefillBuilder.build_prompt_prefill`:重建 prompt embedding(直接从 safetensors 读 thinker embed 行!) + thinker chunk 拼接 |
| 文本来源 | `PendingTextTensorQueue`(预处理装满) | `PendingTextTensorQueue`(thinker 流式 append,`append_text_chunk`;im_end 截断) |
| decode 输入 | 行号寻址 `_decode_feedback_embedding` + 反馈/文本相加 | 完全同构 |
| 采样位置 | SGLang sampler(runner `_sample_next_token_ids`)+ 预测器内自研采样 | **模型 forward 内部** `_sample_decode_tokens`:克隆 logits → 自维护重复/抑制掩码 → 静态 `SamplingBatchInfo` → `self._sampler`(SGLang sampler 类) |
| 采样图安全策略 | `is_all_greedy=False, need_top_p/k=True` 恒定 → 图内分支固定 | 同一份 `SamplingBatchInfo` 构造(两处代码注释几乎相同) |
| 预测器采样 | seeded(自定义内核) | 纯 argmax(`_sample_code_predictor_token`) |
| speaker 条件 | x-vector / spk_id 表 / instruct | 固定 speaker map(`resolve_speaker_id` 缺省 "Ethan") |

关键洞察:**Omni talker 把采样搬进 forward 是为了 CUDA 图完整性**(capture 时 sample 一并在图内,`_sampled_token_ids` 持久缓冲即图输出);TTS 的主采样留在 SGLang sampler(需要 per-request seed 语义与 SGLang 的 penalizer 生态),把预测器采样单独图化。两者的采样位置选择相反,驱动力都是"哪些部分需要进图",而答案因模型而异——这是 CUDA 图工程里"按模型裁剪捕获边界"的两个实例。

下一篇:预测器逐层生成 + `_PredictorDecodeGraph` 的签名/桶/捕获/重放全链路。
