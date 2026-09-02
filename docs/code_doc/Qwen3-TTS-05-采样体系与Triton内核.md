# Qwen3-TTS 深度解析(五):采样体系与 Triton 确定性内核

> 本篇覆盖 `sampling_kernels.py`(576 行)与 `sglang_model.py` 的 `_sample_subtalker_token_seeded`、`model_runner.py` 的种子安装。核心问题:**给定 seed,如何在 GPU 上让采样结果跨设备、跨 torch 版本、与 eager 路径逐位一致?**

---

## 1. 采样语义栈全景

Qwen3-TTS 有两套采样,种子体系把它们串起来:

```
公开请求 seed
   ├── derive "semantic"   → SamplingParams.sampling_seed → SGLang sampler
   │                          (runner._install_semantic_sampling_seeds 每步装进 sampling_info)
   └── derive "subtalker"  → _sub_sampling_seed_tensor
                              → 预测器每层采样(seed + position × (Q-1) + layer_idx + 1)
```

子说话人采样路径(`_sample_subtalker_token_seeded`,`sglang_model.py:1560`)是三级 fallback:

```
① 捕获中 + 契约满足 → sample_from_logits_with_seed_top_k_top_p    (raw-logits 全融合 Triton)
② 否则 ATen 精化:
     scores = logits/温度 → topk(或全排序) → softmax → top-p 截断 → log
     → sample_from_sorted_logprobs_with_seed_small_k                 (sorted-logprobs Triton)
③ 内核返回 None(无 Triton/超宽/非 CUDA) → _sample_seeded_categorical
     = SGLang multinomial_with_seed(logprobs, seeds, positions)      (ATen 参考)
```

**三级必须产出相同分布且同 seed 同结果**——③ 是 SGLang 的参考实现,②/① 是它的性能等价物。下面自底向上拆。

## 2. 随机源:Murmur3 哈希 Gumbel 采样

SGLang 的 `multinomial_with_seed` 语义:**不做 RNG 状态机,而是从 (seed, position) 派生每个候选的确定性 Gumbel 噪声,取 `argmax(logprob + gumbel)`**。Triton 内核逐位复刻(`sampling_kernels.py:33-63`):

```python
h = 0
h = murmur3_mix(h, seed & 0xFFFFFFFF)        # seed 低 32 位
h = murmur3_mix(h, seed >> 32)               # seed 高 32 位
h = murmur3_mix(h, position)                 # 位置(跨步去相关)
h = murmur3_mix(h, column)                   # 候选列号(同位内去相关)
h ^= 16;  h = fmix32(h)                      # 终化
gumbel = gumbel_from_hash(h)
```

`murmur3_mix` 是标准 MurmurHash3 32 位混合(`0xCC9E2D51`/`rotl15`/`0x1B873593`/`rotl13`/`5+0xE6546B64`),`fmix32` 是 avalanche 终化(`0x85EBCA6B`/`0xC2B2AE35`)。Gumbel 变换最精妙的三个 clamp:

```python
u = h.to(f64) / 4294967295.0          # → (0,1]
log_u = log(u)
log_u = max(log_u, -1.7976931348623157e308)   # finfo.min:  h==0 → u 极小,log 不炸
log_u = min(log_u, -2.3283064365386963e-10)   # -(2^-32):  u==1 → log_u==0,log(0) 防炸
return -log(-log_u)                    # Gumbel(0,1)
```

注释点明顺序:**"Clamp after log so hash 0 becomes finfo.min, not a tiny positive u"**——先 log 再 clamp,与 SGLang 的 `x.log_().clamp_(min=finfo.min, max=-(2**-32)).neg_().log_().neg_()` 逐位一致。任何顺序或精度差异(比如 f32 计算)都会让同 seed 的采样在 Triton 与 ATen 之间漂移——而 codec id 漂移一个,音频就不同。**确定性的单位是"采样结果位一致",不是"分布一致"。**

采样决策(两内核同式):

```python
scores = weights + gumbel
max_score = max(scores)
rank = min(where(scores == max_score, col, N))     # 并列时取最小列号 → 与 torch argmax 并列语义一致
token = sorted_idx[row, rank]
```

## 3. 快路径②:`sample_from_sorted_logprobs_with_seed_small_k`

输入:**已排序的 logprobs + sorted_idx**([B, K],K≤1024)。这是 ATen 精化路径的末端——前面 top-k/top-p 已由 ATen 完成,内核只做 Gumbel-argmax:

```python
guard: Triton 可用 + 全 CUDA + 2D + idx 形状一致 + K∈(0,1024] + 种子/位置长度=批大小
block = next_pow2(K)
grid=(B,)  每行一个 program
```

block 上界 1024:单 program 一行,寄存器装得下;超过则返回 None 落回 ③。此内核无 torch 版本耦合问题(输入已排好),所以守门很薄。

## 4. 快路径①:raw-logit 全融合 + **torch.topk 位级对齐**(全系列最硬核)

`sample_from_logits_with_seed_top_k_top_p` 跳过 ATen,直接从 bf16 raw logits 一步到 token。守门契约(`sampling_kernels.py:511-545`)**逐项列出**:

```
max_top_k ∈ {4,8,16,32,50,64,128,256,512,1024}(梯级,与图签名同源)
logits.shape[1] == 2048 且 bf16 且 contiguous     ← 预测器词表硬编码
temperatures f32 / top_ks long / top_ps f32 / seeds long / positions long,全部 1D、连续、同设备
tl.gather 可用(Triton 版本)
```

任何一项不满足返回 None(注释:"对任何未验证的形状/布局刻意返回 None,生产参考实现始终是 fallback")。

### 4.1 为什么必须对齐 torch.topk 的不稳定排序

seeded 采样对**并列分数的排序次序敏感**:`scores==max` 的并列候选取最小列,而"最小列"取决于 topk 输出的排列。若 Triton 用普通字典序 sort,与 `torch.topk` 在并列/阈值处的次序不同 → 同 seed 不同 token。kernel 头部注释把这钉死了:

> torch==2.13.0 下 `torch.topk(sorted=True)` k≤32 时:先按源索引序收集大于阈值的项,再按源索引序追加等于阈值的项,最后过一个**不稳定的 32 项 bitonic 网络**。等价行为对 seeded sampler 是可观测的,常规词典序 sort 不是兼容替代。CUDA parity 测试把这个网络与 torch.topk 对拍,**torch 升级若改变该行为会在测试阶段失败**。

### 4.2 打包键技巧:用一个 uint64 同时表示"分数序 + 索引序"

bf16→f32 分数转 uint32 位图后做**符号翻转变换**(大端浮点全序技巧):

```python
ordered = bits ^ (bits>>31 ? 0xFFFFFFFF : 0x80000000)   # 负数全取反,正数只翻符号位 → 无符号序=浮点序
packed  = (ordered.to(u64) << 32) | (0xFFFFFFFF - vocab_index)
top_packed = tl.topk(packed, k=block_k)
```

低 32 位存 `MAX - index`:**分数相同 → packed 大者索引小 → tl.topk 降序给出"分数降序、同分索引升序"**——正好是 gather 之后想要的目标序。解包反向 XOR 恢复分数位与 token id。

### 4.3 k≤32 的 bitonic 网络复刻

`_bitonic_sort_selected_32_desc`(`sampling_kernels.py:115`)展开 15 轮 compare-exchange(stride 序列 1;2,1;4,2,1;8,4,2,1;16,8,4,2,1),每轮 `_bitonic_compare_selected_32_desc`:

```python
is_right = (offset & stride) != 0
left  = where(is_right, partner, self)
right = where(is_right, self, partner)
direction = final_round ? 0 : ((thread_id & (network_size/2)) != 0)
swap = ((left > right) & left_valid) | (right_valid == 0)
take_partner = (swap == direction)
```

`final_round` 区分:最后一轮是纯降序整理(direction 恒 false);前几轮是 bitonic 归并方向模式。**valid 位参与比较**(`right_valid==0` 强制交换)——把无效槽(超出 max_top_k)沉到尾部。网络形状是 PyTorch CUDA `SmallBitonicSort` 的逐轮复刻,注释明确 parity 测试契约:"torch 升级时先红测试再改这里"。

### 4.4 阈值收集(gather)阶段的次序复刻

k≤32 时 PyTorch 先 gather 再 sort;gather 本身有序:大于阈值的按源索引序在前,等于阈值的按源索引序在后——**且"大于/等于"用位图比较而非浮点比较**(注释:"CUDA 阈值收集比较浮点位序表示,尤其 +0 先于 -0;数值比较会坍缩这一区别并可能改变 seeded codec id"):

```python
greater = ordered_bits > threshold_bits         # 位比较!
equal   = ordered_bits == threshold_bits
gather_key = (where(greater,0, where(equal,1, MAX)).to(u64) << 32) | src_index
gathered = tl.topk(MAX_KEY - gather_key, k=32)  # tl.topk 只降序 → 取补变升序
```

又一次"补码反转"把 topk 的降序变成需要的源索引升序。之后 gather 分数、过 bitonic 网络,才是干净的 `sorted_scores/sorted_token_ids`。

### 4.5 采样收尾

```python
keep_top_k = rank < per_row_top_k
masked = where(keep, sorted_scores, -inf); probs = softmax(masked)
if has_top_p: cdf 截断(remove &= rank!=0;remove &= active_top_p)     # 首位永不删
logprobs = where(keep & ~removed, log(probs), -inf)
gumbel = murmur(seed, pos, rank)…                                      # 与 §2 同一哈希
sampled_rank = min(where(scores==max, rank, K))
token = max(where(rank==sampled_rank, sorted_token_ids, 0))            # 用 max 抽取选中槽
```

注意两处位运算巧思:`max(where(...))` 从寄存器向量中抽取单元素(避免 dynamic indexing);`rank` 维度上 Gumbel 键用 rank 而非 token id——与 ② 路径的"sorted logprobs 列号"一致,保证 ①/② 同 seed 同结果。

### 4.6 梯级与块宽

`_fused_raw_logit_block_k`:k≤32 → 固定 32(所有这类宽度共用 PyTorch 的 32 网络);>32 → next_pow2(k)。块宽是编译期常量(`block_k: tl.constexpr`),所以它必须进图签名——**签名里的 max_top_k 实际上冻结了 kernel 的特化版本**。这就是 04 篇梯级量化的最终目的:把"任意 top_k"压缩到有限个 kernel 实例,图键数才有界。

## 5. ATen 精化路径:top-k/top-p 的双分支

`_sample_subtalker_token_seeded` 的 ATen 分支(`sglang_model.py:1600`):

```python
if max_top_k>0 and < vocab and not unbounded:            # 有界:topk 宽度=梯级
    sorted_scores, sorted_idx = torch.topk(scores, max_top_k)
    keep = rank < per_row_top_k                            # per-row 掩码截断到真实 k
else:                                                    # 无界:全排序
    sort 全 vocab;keep = (top_k<=0)|(top_k>=vocab)|(rank<top_k)
softmax → top-p(cdf - p >= top_p 且 p∈(0,1),首位保留)→ 二次掩码 → log
→ sample_from_sorted_logprobs_with_seed_small_k
→ fallback:_sample_seeded_categorical(sorted_logprobs, seeds, positions)
           = multinomial_with_seed(...).view(-1) → sorted_idx.gather(1, rank)
```

两个工程点:

- **梯级宽度 + per-row 掩码**:批内 top-k=7 和 50 都按 50 宽 topk,再各自掩到 7/50。代价是多排了 43 个候选,收益是 topk 调用形状统一(且与图签名一致)。
- `_sample_subtalker_token` eager 入口的行选择(`sglang_model.py:1490`):全 greedy → 直接 argmax;部分采样 → `index_select` 采样行、argmax 填其余(**scatter 回原行号** `tokens[sampled_rows] = ...`);全采样 → 整批走 seeded。三种形态由 `prepare_decode_buffers` 统计出的 `_sub_sample_count/_sub_has_sampled_rows` 决定——host 分支全部在图外完成。

## 6. 语义层(层0)的种子接线

`Qwen3TTSModelRunner._sample_next_token_ids`:

```python
def _sample_next_token_ids(self, logits_output, forward_batch, schedule_batch, requests):
    self._install_semantic_sampling_seeds(forward_batch, requests)
    return super()._sample_next_token_ids(...)
```

```python
forward_batch.sampling_info.sampling_seed = self.model._semantic_sampling_seed_tensor[:batch_size]
```

基类 `_install_sampling_seeds` 开头有让路逻辑:"或当子类已安装自己的(Qwen3-TTS)"——`sampling_seed is not None` 即跳过。**注意种子的设备张量化**:runner 每步把 [B] 张量直接赋给 `sampling_info.sampling_seed`(不是逐行 list),SGLang sampler 检测到非 None 即路由 `multinomial_with_seed`。基类注释同时警告:混合 seeded/unseeded 批中,unseeded 行拿 rid 派生的 rank 共享种子(而非随机)——保证 TP 各 rank 同步;TTS 通过在 `prepare_decode_buffers` 阶段为无 seed 请求派生随机种子(`_new_qwen3_tts_sampling_seed()`,02 篇)避免了混合情形。

重复惩罚的分工(06 篇详述):层0 由 **SGLang 原生 `BatchedRepetitionPenalizer`** 在 `prepare_for_decode` 增量维护、`ModelRunner.sample` 统一应用——TTS runner 的 `_apply_repetition_penalty` 刻意 no-op。**只有预测器层没有 SGLang 采样状态机,才需要这里整套自研内核。**

## 7. 与 Omni 采样体系的对照

| | Qwen3-TTS | Qwen3-Omni |
|---|---|---|
| 层0 采样 | SGLang sampler + per-request semantic seed | talker **模型内** `_sample_decode_tokens` → SGLang `Sampler` 类 + 静态 `SamplingBatchInfo`(seeds 来自暂存张量,未 seed 用 rid 派生) |
| logits 预处理 | no-op 惩罚(原生 penalizer)+ 尾 1024 抑制(runner,host 侧切片) | 图内 `torch.where` 惩罚 + `_suppress_mask`(缓冲暂存,04/06 篇) |
| 预测器采样 | seeded 全家桶(§1 三级) | 纯 argmax 一行 |
| Triton 内核 | 2 个采样内核 + parity 测试网络 | 无 |
| 图键敏感度 | 采样签名进键 | dtype 进键 |
| 静态采样信息 | 不需要(sampler 在图外) | `is_all_greedy=False, need_top_p/k=True` 恒真——**图内分支恒定**的注释两个模型几乎逐字相同 |

一句话总结差异根源:**Omni 的确定性要求止步于"图内分支固定",TTS 的确定性要求上升到"跨路径位一致"**——后者是 voice-clone 场景的业务需求(同样 seed 必须复现同样的音色细节),前者只是 CUDA 图的技术约束。

下一篇:runner 钩子如何把这些模块按步缝合,以及 retraction 恢复惩罚历史的来龙去脉。
