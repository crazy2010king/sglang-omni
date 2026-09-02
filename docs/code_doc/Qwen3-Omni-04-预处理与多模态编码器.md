# 04 预处理与多模态编码器：Preprocessor、双塔、批量与缓存

> 主角：`components/preprocessor.py`（728 行）、`components/image_encoder.py`（202 行）、
> `components/audio_encoder.py`（251 行）、`components/audio_layer_graph.py`（273 行）、
> `stages.py` 的批量/缓存部分（1216 行的一半）。

---

## 1. Preprocessor：CPU 上的"请求整形"

`Qwen3OmniPreprocessor` 在 `preprocessing` 阶段以 `compute_fn = await preprocessor(payload)`
运行（异步：媒体解码/下载可让出）。输出写进 `Qwen3OmniPipelineState`：

| 字段 | 内容 |
|------|------|
| `prompt` | `PromptInputs{input_ids, attention_mask, prompt_text}` |
| `mm_inputs` | 每模态元数据：`image_grid_thw` / `feature_attention_mask`+`audio_feature_lengths` / `video_grid_thw`+`video_second_per_grid`+`use_audio_in_video` |
| `encoder_inputs` | `{"image_encoder": {...张量+cache_key+_active}, "audio_encoder": {...}}` |
| `stream_state` | 贯穿全流的流状态（如 AIV 切分游标） |

### 1.1 关键机制

- **chat template 与 HF processor**：`ensure_chat_template`（必要时回退到
  `Qwen/Qwen3-Omni-30B-A3B-Instruct` 的模板）+ HF `Qwen3OmniMoeProcessor`。
  transformers 5.x 把 `image_token` 等特殊 token 写到 tokenizer_config 顶层，
  4.x 期待 `extra_special_tokens` 子字典——`_extra_special_tokens_compat` 做了迁移补丁。
- **预分词直通**：`_is_pretokenized_prompt`——Miles RL rollout 发来纯 int 列表时
  绕过模板与 processor，保证训练/rollout token 严格一致。
- **媒体缓存键**：`compute_{image,audio,video}_cache_key`（内容哈希）+
  `_contextualize_cache_key(base, fps=..., frames=...)`（处理参数进键）。
  cache_key 随 `encoder_inputs` 一路带到 encoder（LRU）与 merge（pad 值哈希，05 篇）。
- **长度预检**：`validate_prompt_seq_len`：`prompt_len >= max_seq_len` 或
  `prompt_len + max_new_tokens >= max_seq_len` 都直接 ValueError，
  拒绝发生在最便宜的 CPU 阶段。

---

## 2. Image Encoder：Vision Tower + PatchEmbed 优化

### 2.1 PatchEmbed：Conv3d → Linear（image_encoder.py:26-79）

HF 的 `patch_embed.proj` 是 `Conv3d(patch, hidden, kernel=stride=(2,2,2))`。
当 **kernel_size == stride 且 padding=0、dilation=1、groups=1** 时，卷积核不滑窗——
每个输出位置只看一个不重叠的 patch，数学上等价于 `reshape + Linear`：

```python
linear = nn.Linear(in_channels*t*h*w, embed_dim, bias=True, ...)
linear.weight.copy_(conv.weight.view(embed_dim, -1))
linear.bias.copy_(conv.bias)
patch_embed.forward = MethodType(_patch_embed_forward, patch_embed)
```

`_patch_embed_forward` 就一句：`self.linear(hidden_states.to(dtype=self.linear.weight.dtype))`。
收益 7~15×（Conv3d 的 im2col/布局开销在小核情形全是税）。三个安全阀：
kernel≠stride 跳过、非平凡 padding/dilation/groups 跳过、仅做一次（权重已 copy，原 conv 被 del）。

### 2.2 forward 输出契约

`Qwen3OmniImageEncoder.forward` 返回 dict（`stages.py` 与 merge 直接消费）：

```python
{"image_embeds", "image_grid_thw", "image_token_counts",
 "deepstack_visual_embeds_image",          # 多尺度残差嵌入（层列表）
 "video_embeds", "video_grid_thw", "video_token_counts",
 "deepstack_visual_embeds_video"}
```

`_unpack_visual_output` 兼容 transformers 新旧两代返回
（属性 `pooler_output/deepstack_features` 或 tuple）。
`spatial_merge_size/out_hidden_size/deepstack_layers/visual_dtype_bytes`
四个标量被缓存为实例属性，专门供批量成本函数零开销读取。

---

## 3. Audio Encoder：打包 + 段共享 + 层图

### 3.1 特征打包（audio_encoder.py:44-60）

HF 音频塔吃 `[mel, sum(len)]` 的**拼接**布局而非 batch 布局。
`pack_padded_audio_features` 走快路径的前提是 mask 是**前缀 mask**：
`torch.equal(mask, steps < lengths)` 成立时直接
`torch.cat([row[:, :length] ...], dim=-1)`；否则回退到布尔索引的 gather 路径
（注释点明：内部有洞就必须 gather）。

### 3.2 `_SegmentSplits`：消除 32 次 D2H 同步

变长音频以 `cu_seqlens` 段送入 attention。**原版 HF attention 每层都要从
`cu_seqlens` 做 device→host 拷贝算段长**——32 层 × 1 次同步，既慢又让层图无法捕获。
修复（audio_encoder.py:63-118）：

1. `_SegmentSplits` 是个单槽容器，`Qwen3OmniAudioEncoder.forward` 在调塔前
   一次性算好 `splits = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()`；
2. monkey-patch 每层 attention 的 forward 为 `_forward_with_shared_segments`：
   先校验 `sum(splits) == hidden_states.shape[0]`（不匹配就回退原实现——
   **过期 split 会静默损坏 attention，宁可信不过就不信**），然后
   `torch.split` 把 Q/K/V 按段切开逐段 attention 再 `torch.cat`；
3. `_share_segment_splits` 在构造时给每层注入 `_omni_segment_splits` 与
   `_omni_unshared_forward`（原 forward 的引用）。

### 3.3 层间 CUDA 图（audio_layer_graph.py）

`_GraphedLayerStack` 用"**把整层列表伪装成一个模块**"的技巧让塔的
`for layer in layers` 循环只执行一次 Python 迭代：

```python
class _GraphedLayerStack(nn.Module):
    def __iter__(self): yield self          # 塔以为只有"一层"
    def __len__(self): return 1
    def forward(self, hidden_states, cu_seqlens, **kwargs):
        replayed = self._runner.maybe_replay(hidden_states, cu_seqlens, segments)
        if replayed is not None: return (replayed,)
        for layer in self._layers: ...      # 图 miss 时逐层 eager
```

窗口大小：`chunk_tokens = downsample(n_window*2)`，
`window = chunk_tokens * (n_window_infer // (n_window*2))`——即推理窗口
（`n_window_infer` 个 mel 窗）对齐到捕获粒度。`runner.capture_all()` 失败
（`has_graphs=False`）则**整体留在 eager**，只是打 warning。

---

## 4. 编码器批量：同批融合与去重（stages.py）

`_batch_image_encoder_payloads` / `_batch_audio_encoder_payloads` 是对称的两套五段式流程：

```
① 分流：skip_result 的 payload 单独走单条路径
② 查缓存：cache_key 命中 → 直接 apply_encoder_result（不再计算）
③ 批量资格：_image_request_is_batchable（四个输入键都必须是纯 Tensor，
   非张量如 list[Image] 不可批）；_audio_request_is_batchable（input_features 是 Tensor）
④ 同批去重：active_cache_keys 集合 → 同键后到者挂进 duplicate_waiters，
   由"leader"的结果直接分发（一次计算服务 N 个请求）
⑤ 真批量：
   图像：torch.cat(pixel_values/grid_thw)（图与视频分开 cat），一次 forward，
         再按 image_rows/video_rows + token 计数游标切回各请求
   音频：_normalize_audio_request_tensors 统一（features 升维、lengths 从 mask 求和、
         mask 从 lengths 重建），pad 到 max_time 后 cat，forward 后按
         audio_output_lengths 的 token 游标切分
```

切分正确性的关键：图像/视频 token 数 = `grid.prod(-1) // merge²`，输出按
`[token_cursor, token_cursor + 该请求 token 总数)` 切 `embeds`，
按 `[row_cursor, row_end)` 切 grid/counts。两个游标独立推进。

批量成本上限见 02 篇 §2.2。缓存是 CPU LRU
`StageOutputCache(max_size=64, max_bytes=4GiB, cache_device="cpu")`，
trace 由 `SGLANG_OMNI_TRACE_ENCODER_CACHE=1` 打开，日志形如
`encoder_cache stage=image_encoder action=hit/miss/store/dedup_same_batch req=... key=... input_bytes=... output_bytes=...`。

> 为什么要 CPU 缓存：多轮对话里同一张图会被反复请求；encoder 输出是
> "token 数 × hidden × dtype × (1+deepstack层数)" 的大张量，放 GPU 会挤占 KV cache。

---

## 5. Encoder 结果如何流向下游

`apply_encoder_result(state, stage_name, result)`（request_builders.py:190-200）把结果
**同时**写 `state.encoder_outs[stage_name]` 与 `state.engine_outputs[stage_name]`
（后者给终态回执/调试用）。随后 stage 出边投影：

- → join 目标：`project_encoder_to_mm_aggregate` 校验 `encoder_outs` **必须单键**
  （`_single_encoder_stage_name`），否则 ValueError——投影函数同时也是不变量断言；
- → talker_ar：剔除 `deepstack_visual_embeds_{image,video}`（注释：
  talker prefill 只用 image/video/audio embeds，从不吃 deepstack）。

encoder 阶段还包了 profiler 事件（`encoder_start/encoder_end`，metadata 带
modality 与 batch_size），这是 codebase 里粒度最细的可观测点之一。

---

## 6. 小结与踩坑清单

1. PatchEmbed 优化**有条件**：kernel==stride 且无 padding/dilation/groups；
   条件不满足时代码选择"不优化"而不是"错误优化"。
2. 音频段共享的前提是 `sum(splits)==行数`，校验失败自动回退——性能优化必须带
   等价性保险丝。
3. 批量切分依赖两个独立游标（row / token），任何一处 off-by-one 都会让
   embedding 串请求——这也是为什么 merge 端还有 `_single_encoder_stage_name` 断言。
4. 缓存键必须把处理参数（fps/frames/pixels）一起哈希，否则调参后命中旧结果。
5. `_skip`/`_active` 标记是"这个 encoder 分支本次请求是否真实存在"的唯一事实来源，
   被 wait_for_fn、路由函数、投影函数三处共享。

下一篇（05）进入 encoder 输出的消费者：merge_for_thinker 与 M-RoPE。
