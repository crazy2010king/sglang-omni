# Qwen3-TTS 深度解析(四):CodePredictor 与 CUDA 图加速

> 本篇覆盖 `sglang_model.py` 的 `Qwen3TTSCodePredictor`、`_code_predictor_forward_incremental`、`_PredictorDecodeGraph`,以及 Omni `components/talker.py` 中对应物的异同。

---

## 1. 为什么需要 CodePredictor

Qwen3-TTS 的码本有 Q 层(`num_code_groups`)。主 AR 骨干每步只产**层0(语义层)**的 token;层 1..Q-1(声学细节层)由一个**独立小 transformer(Qwen3TTSCodePredictor)在层0 条件下逐层自回归**。这意味着每个 12Hz 音频帧,推理实际执行:1 次骨干前向 + Q-1 次预测器前向 + Q-1 次采样。若预测器走 host 驱动的普通路径,Q=12 时每帧 ~22 次 kernel launch 链,host 开销吞掉收益——**预测器链整体 CUDA 图化是本模型在 SGLang 里能打的核心优化**。

结构(`Qwen3TTSCodePredictor.__init__`):

```python
self.model.codec_embedding = ModuleList([Embedding(cp_vocab, hidden) for _ in range(Q-1)])  # 每层独立嵌入
self.model.layers          = ModuleList([Qwen3TTSTalkerDecoderLayer(cp_config, ...)])       # 小骨干
self.model.norm            = RMSNorm(cp_hidden)
self.lm_head               = ModuleList([ReplicatedLinear(cp_hidden, cp_vocab) for _ in range(Q-1)])  # 每层独立头
if cp_hidden != hidden:
    self.small_to_mtp_projection = nn.Linear(hidden, cp_hidden, bias=True)   # hidden_size 适配
```

- `cp_config = config.code_predictor_config`:预测器更小(hidden/vocab/层数都独立),`cp_vocab=2048`(05 篇 fused kernel 的硬前提之一就是 `logits.shape[1]==2048`)。
- `small_to_mtp_projection` 命名源自 MTP(multi-token prediction)惯例:骨干 hidden → 预测器 hidden 的线性适配。
- **每层一个嵌入表 + 一个 LM 头**(不是共享),因为各声学层是不同的分布。

## 2. 增量前向:`_code_predictor_forward_incremental`(eager 参考实现)

这是整个模型的"第二心跳"。逐 token 分解(设批 B,序列长 seq_len):

```
对每个位置 pos:
  layer0_embed   = codec_embedding(layer0_code)            # 层0码嵌入 [B,1,H]
  talker_pred    = project_input(talker_hidden[:, pos])     # 骨干 hidden → 预测器空间
  pos_codes[:,0] = layer0_code;  pos_summed += layer0_embed # 层0 直接进结果与累加和

  _predictor_forward_one_token(talker_pred, cache_len=0)    # token A:骨干 hidden
  _predictor_forward_one_token(layer0_pred, cache_len=1)    # token B:层0嵌入
  last_hidden = (B 的输出)

  for layer_idx in 0..Q-2:                                  # 预测层1..Q-1
      logits = lm_head[layer_idx](last_hidden)
      next_code = _sample_subtalker_token(logits, layer_idx, ...)
      pos_codes[:, layer_idx+1] = next_code
      new_embed = codec_embedding[layer_idx](next_code)
      pos_summed += new_embed                                # 累加 ΣQ 嵌入(反馈向量)
      if layer_idx < Q-2:
          last_hidden = _predictor_forward_one_token(project(new_embed), cache_len+=1)
      # 最后一层无需再前向——没有下一层要预测
```

**三个容易误读的点:**

1. **token 序列 = [talker_hidden, layer0_embed, embed(c1), embed(c2), ...]**:预测器把骨干 hidden 当作第一个输入 token,层0 嵌入是第二个,之后每个已采样码的嵌入是下一个输入——这是一个**单步内的小自回归**,KV cache 长度最多 `Q+1`(`predictor_len = config.num_code_groups + 1`,正好解释 §5 缓冲形状)。
2. **`summed_embeddings`(ΣQ 嵌入)就是下一帧 talker 的输入反馈**(`_output_embeds`),与层0码一起构成"语音反馈闭环"。03 篇 §4.6 的 `pending_feedback_queue` 来源就是它。
3. **`cache_len` 计数在每个 token 前向后自增**,且**序列内位置固定 0..Q**(`_predictor_positions`)——预测器内没有位置外推,因为序列长度恒定 ≤ Q+1。`_normalize_semantic_positions` 把外部传来的语义位置展成 [B, seq_len],只用于**采样的 Gumbel 键**,不影响注意力位置。

`_predictor_forward_one_token`(`sglang_model.py:1663`)是手工展开的 decoder 层循环:

```python
positions = self._predictor_position_rows[cache_len, :batch_size]   # [B] 行切片,免 H2D
for layer in layers:
    normed = layer.input_layernorm(hidden.reshape(-1, H))           # RMSNorm(带 residual 分支)
    attn   = self._predictor_cached_self_attention(layer_idx, ...)
    hidden = self._predictor_o_proj_add_residual(o_proj, attn, residual)   # ← H100 特化
    normed = layer.post_attention_layernorm(...)
    hidden = residual + mlp(normed)
return norm(hidden)
```

`_predictor_cached_self_attention`(`sglang_model.py:1702`)——**预测器的 KV cache 不走 SGLang 的 paged KV pool,而是自管的稠密张量**:

```python
qkv = qkv_proj(flat_hidden); q,k,v = split([q_size, kv_size, kv_size])
q, k = apply_qk_norm(q, k, ...)                     # 复用 omni 的 qk norm
q, k = rotary_emb(positions, q, k)                  # 固定位置
layer_k_cache[layer_idx, :B, :, cache_len:cache_len+1, :].copy_(k)   # 写自管 cache
attn = SDPA(q, k_cache[:, :cache_len+1], v_cache[:, :cache_len+1], is_causal=False, enable_gqa=True)
```

细节:`is_causal=False`——因果性由 cache 切片天然保证(当前 token 只看见前缀);`enable_gqa` 当 kv 头数 < q 头数时在 kernel 内广播。自管 cache 的原因:预测器序列与骨干序列是两个不同的"时间轴",塞进 SGLang paged pool 需要为它单独管理页表,得不偿失;稠密 [L,B,kv_heads,Q+1,head_dim] 也就几 MB。

`_predictor_o_proj_add_residual`(`sglang_model.py:1620`)是**为图捕获定制的 H100 融合**:

```python
use_fused_addmm = (is_cuda and capturing and not grad and capability==(9,0)
                   and UnquantizedLinearMethod and tp_size==1 and no bias
                   and bf16 + contiguous + shape 相容 全部满足)
if use_fused_addmm:
    residual_2d = residual.reshape(B, H_out)
    torch.addmm(residual_2d, attn_input, weight.t(), out=residual_2d)   # 原地,residual 即 beta 项
```

注释指出:**调用方在返回后立即替换 residual 变量,复用其存储作为 addmm 的输出,连 beta 项的 copy 都省了**。条件检查近乎偏执(量化方法/tp/bias/dtype/连续性/形状逐项断言),任何不满足就走 eager fallback——性能特化的标准姿势:快路径激进,守门人严格。

## 3. `_PredictorDecodeGraph`:整链图化

### 3.1 捕获(`_capture`,`sglang_model.py:97-139`)

```python
with model._predictor_graph_capture_state(bucket_size, signature):     # ① 暂存并改写全局采样状态
    warmup_stream = torch.cuda.Stream(...)
    with torch.cuda.stream(warmup_stream):
        for _ in range(2):                                              # ② 2 次 warmup(边流)
            model._code_predictor_forward_incremental(..., for_capture=True)
    capture_stream = torch.cuda.Stream(...)
    with torch.cuda.graph(self.graph,
                          pool=model._predictor_graph_memory_pool(),    # ③ 共享内存池
                          stream=capture_stream,
                          capture_error_mode="thread_local"):
        self.result_codes, self.summed_embeddings = model._code_predictor_forward_incremental(
            ..., for_capture=True)
```

- **① `_predictor_graph_capture_state`**:捕获时把 `_sub_batch_size/_sub_sample_count/_sub_sampled_max_top_k/...` 全局暂存值替换成签名对应的值,finally 恢复。因为图内的采样分支读取这些宿主标志——**捕获那一刻决定图内的控制流**,这就是"签名钉住 host 分支"的实现。
- **② warmup 在独立边流**:两次 eager 执行把 lazy init(cudnn/cublas 句柄、autotune)在捕获前做完——CUDA 图捕获期间不允许有首次分配/编译这类"非法"操作。
- **③ `graph_pool_handle()` 全部图共享一个池**:注释强调"per-graph 私有池会按 key 数保留中间张量,随多样性线性膨胀"。共享池的前提是不同图不同时重放(事实:同一模型同一时刻只有一个预测器前向)。
- **`for_capture=True`** 让 `_sample_subtalker_token` 走 graph-safe 分支、让 embedding gather 走 fused kernel(见下)。

### 3.2 输入输出契约

```python
self.layer0_codes      = zeros(bucket, 1, long)      # 静态输入
self.talker_hidden     = zeros(bucket, 1, hidden)    # 静态输入
self.semantic_positions= zeros(bucket, long)         # 静态输入(采样键)
self.result_codes      # 图输出:view of _output_codes
self.summed_embeddings # 图输出:view of _output_embeds
```

`replay()`(`sglang_model.py:144`):

```python
self.layer0_codes[:live].copy_(layer0_codes)          # D2D 拷入静态缓冲
self.talker_hidden[:live].copy_(talker_hidden)
self.semantic_positions[:live].copy_(...)
if live < bucket: 尾部清零                              # 死行归零,防止上一步残留污染 Σ
self.graph.replay()
return result_codes[:live], summed_embeddings[:live]   # 返回 live 前缀视图
```

死行清零不是洁癖:`summed_embeddings` 是全缓冲累加,若上一步 live=3 残留、本步 live=1,死行虽然不返回,但**它们的累加不发生的前提是输入为 0**(embedding(0) 非零,故输入码也必须清零)。

### 3.3 签名系统:为什么要按"采样形态"分图

`_predictor_graph_signature`(`sglang_model.py:1102`):

```python
if not self._sub_has_sampled_rows:   return ("argmax", 0, False, False)   # 全批 greedy
return ("sampled", max_top_k, has_top_p, has_unbounded_top_k)             # 有采样行
```

四元组**完整钉死图内控制流**:(argmax vs Gumbel 采样)、(top-k 候选宽度——决定 kernel 分支)、(是否需要 top-p 截断)、(top-k 是否触顶需全排序)。任何进入图的选择分支必须是签名的一部分,否则重放时分支结果与 eager 不一致。**Omni 的 `_PredictorDecodeGraph` 键只有 `(bucket, code_dtype)`**——因为 Omni 预测器只 argmax,host 分支只有一种,TTS 多出来的维度全部来自 seeded sampling。这是"确定性采样需求倒逼图键复杂化"的活标本。

桶(batch bucket)与键:

```python
_PREDICTOR_GRAPH_MAX_KEYS = 32;  _PREDICTOR_GRAPH_MAX_FAILURES = 8
_PREDICTOR_TOP_K_LADDER = (4, 8, 16, 32, 50, 64, 128, 256, 512, 1024)
batch 桶 = get_decode_cuda_graph_bs(server_args) or (1,2,4,8,12,16[,max])   # 对齐骨干默认捕获表
键 = (bucket, *signature)
```

50 在阶梯里是特意的:1.7B checkpoint 默认 top_k=50,**让最常见签名的 kernel 宽度与之前完全一致**(不因量化到 64 而回退)。

### 3.4 降级矩阵(生产级容错)

`_predictor_forward_graphed`(`sglang_model.py:1248`)的通过条件依次检查:env 开关(`SGLANG_OMNI_QTTS_PREDICTOR_GRAPH`,默认开)、`disable_cuda_graph`、`tp_size==1`(注释:TP 会把 collective 录进图,图化链路只在单卡验证过)、seq_len==1(decode only)、dtype/设备、`batch_size == self._sub_batch_size`(批组成刚暂存过)、`torch.cuda.is_current_stream_capturing()`(防图套图)、签名非 None、桶存在、键不在禁用集。之后:

- **容量兜底**:键数达 32 → 首次 warning + eager(计数,只警告一次);
- **捕获失败**:单键禁用 + 计数,达 8 次 → **全局禁用预测器图** + warning;
- `_predictor_graph_enabled` 初始 None(注释:"bootstrap 把捕获推迟到 init 之后,此处不决断")——避免构造期捕获与 colocated stage 的请求期 GPU 工作重叠。

每次捕获成功都 `logger.info("Captured ... key=%s")`——图键集合是运维上最值得盯的日志。

### 3.5 图内采样的"无分支"化

图内不能有依赖数据的 host 分支,`_sample_subtalker_token_graph_safe`(`sglang_model.py:1522`)的策略是**全采样再选择**:

```python
sampled_tokens = _sample_subtalker_token_seeded(全批 logits, ...)
argmax_tokens  = torch.argmax(logits, dim=-1)
return torch.where(self._sub_do_sample_tensor[:B], sampled_tokens, argmax_tokens)
```

greedy 行也走一遍 seeded 采样(浪费一点算力)然后 `torch.where` 丢弃——eager 路径则相反(`_sample_subtalker_token`:仅对 sampled 行 index_select 采样,argmax 行直接填),省算力但分支多。**eager 省算力、图内省分支**,同一个语义两套实现,由 `for_capture` 一参数切换。`torch.where` 的对偶形式在 Omni 里不存在(它没有 per-row do_sample)。

## 4. 两个 fused kernel 的接线点

**`gather_codec_embedding_and_add`(`predictor_kernels.py`)**:在图捕获期,把"查嵌入 + 累加到 Σ"合一次 launch:

```python
fused_embedding = (use_fused_embedding     # = 捕获中 + buffer 存在
                   and gather_codec_embedding_and_add(next_code, codec_weight,
                                                      embedding_buffer, pos_summed))
if fused_embedding: new_embed = embedding_buffer.unsqueeze(1)      # 结果已在两个缓冲里
else:               new_embed = codec_embedding(next_code); pos_summed.add_(...)
```

kernel 本体(`_gather_codec_embedding_and_add_kernel`):grid=(B, ceil(H/256)),每 program 读 `token_id` → 拷权重行到 `gathered` → `accumulated += values`。守门检查(22 项)包括 **`_contiguous_storage_ranges_overlap`:三块缓冲的物理地址区间两两不重叠**——Triton kernel 不做别名保护,由调用方物理排除。eager fallback 保留完整语义。注释犀利:"Capture P2 into a graph; a standalone Triton launch loses to ATen"——**这个 kernel 只有在图里(消除 launch 间隙)才赢**,eager 下 ATen 两连击反而更快。

**采样 fused kernel**(`sample_from_logits_with_seed_top_k_top_p`)在 eager 采样路径里只于"捕获中"尝试(`sglang_model.py:1590`):"raw-logit 融合只在预测器 CUDA 图里有 launch 数优势;eager 路径保留成熟 ATen 序列(含全部形状覆盖)"。细节在 05 篇。

## 5. 与 Omni 预测器的逐项对比

Omni 的对应物:`components/talker.py` `_PredictorDecodeGraph`(L58-152)+ `_code_predictor_forward_incremental`(L1413)+ eager 版(L1539)。

| 维度 | Qwen3-TTS | Qwen3-Omni |
|---|---|---|
| 图键 | `(bucket, argmax/sampled, max_top_k(梯级), has_top_p, has_unbounded)` | `(bucket, code_dtype)` |
| 图内采样 | seeded Gumbel(fused Triton,若契约满足)或 where-mix | **纯 argmax**(注释:匹配 HF `do_sample=False`) |
| 输入组织 | 逐 token 自管 KV(`_predictor_k/v_cache`,`cache_len` 渐增,token 嵌入即输入) | **预分配定宽输入缓冲** `predictor_input[:, 0]=hidden, [:,1]=layer0, [:,2..]=new embeds`,`_predictor_forward_one_token` 仍用 KV cache(两版相同) |
| 输出缓冲 | `_output_codes/_output_embeds` 复用 + 死行清零 | 同构(`_output_codes[:bs].zero_()`) |
| 内存池 | `graph_pool_handle()` 全键共享 | 每图默认独立(未传 pool)——Omni 键少(dtype × bucket),池膨胀风险低 |
| 位置 | 固定 0..Q(`_predictor_position_rows[cache_len]` 行切片) | `_predictor_positions[cache_len:cache_len+1].repeat(batch)`(每行 repeat) |
| 捕获保护 | 容量上限/失败计数/全局禁用/env 开关/TP 拒绝/签名钉状态 | 失败禁用该键 + warning(无计数、无 env、无 TP 检查) |
| fused kernel | gather+add、raw-logit 采样、H100 addmm | 无(SDPA+ATen 全程) |
| 预测器隐藏维度适配 | `small_to_mtp_projection`(cp_hidden≠hidden 时) | 同名同构 |

为什么 Omni 敢更简?**Omni 的预测器采样无随机性**(argmax 无分支敏感性),而 TTS 要在图内复现"与 eager 逐位一致"的随机采样,签名/梯级/graph-safe 分支全是为这一件事服务的。**CUDA 图的复杂度正比于图内控制流的多样性,而控制流多样性正比于采样语义的丰富度。**

## 6. 帧级时序全景(把 03/04 篇串起来)

```
第 t 帧(每 ~83ms @12Hz):
  ┌── 骨干 decode(在 SGLang 图或 eager)
  │    输入: _decode_feedback_embedding[row] = Σ_{t-1}Q嵌入 + next_text
  │    输出: hidden_state[t] + 层0 logits
  ├── SGLang sample(semantic seed;重复惩罚=原生 penalizer;抑制尾 1024)
  │    输出: layer0_code[t]
  └── 预测器链(本篇,CUDA 图重放或 eager)
       输入: talker_hidden[t], layer0_code[t], semantic_pos
       内部: Q-1 层循环(自管 KV + 每层独立头 + seeded 采样)
       输出: _output_codes[t] = [c0..c_{Q-1}]  → runner 快照 → vocoder 流块
             _output_embeds[t] = ΣQ嵌入          → runner 快照 → 下一步反馈
```

下一篇逐位拆解 `sampled` 分支内部的采样内核——全系列数值最硬的部分。
