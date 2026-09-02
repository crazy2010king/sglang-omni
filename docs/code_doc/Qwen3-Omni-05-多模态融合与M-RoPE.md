# 05 多模态融合与 M-RoPE：merge_for_thinker、pad 值哈希、三轴位置编码

> 主角：`models/qwen3_omni/merge.py`（293 行）、`models/qwen3_omni/mrope_positions.py`（331 行）、
> `request_builders.py` 中 `build_sglang_thinker_request` 的 pad 值与 mm_positions 部分。

---

## 1. merge_for_thinker：把三路 payload 折成一份数学输入

### 1.1 融合算法（merge.py:35-63）

```python
def merge_for_thinker(payloads: dict[str, StagePayload]) -> StagePayload:
    base = payloads.get("preprocessing") or next(iter(payloads.values()))
    state = Qwen3OmniPipelineState.from_dict(base.data)
    encoder_outs = dict(state.encoder_outs)               # base 自带的部分
    for stage_name, payload in payloads.items():          # 其它源（两塔）带来的部分
        stage_state = Qwen3OmniPipelineState.from_dict(payload.data)
        if stage_name in stage_state.encoder_outs:
            encoder_outs[stage_name] = stage_state.encoder_outs[stage_name]
        elif stage_name in stage_state.engine_outputs:
            encoder_outs[stage_name] = stage_state.engine_outputs[stage_name]
    state.thinker_inputs = build_thinker_inputs(state, encoder_outs)
    state.encoder_inputs = {}
    _prune_preprocessing_for_thinker(state, encoder_outs)
    state.encoder_outs = {}                               # 已消费，立即丢弃
    base.data = state.to_dict()
    return base
```

三个值得注意的取舍：

1. **消费即销毁**：`encoder_outs` 折进 `thinker_inputs` 后被清空，注释写明
   "两处都留会把多模态张量双倍发给 thinker"。这个阶段的 IPC 是 payload 尺寸的
   大头（图像 embedding 可达数万 token），省一半就是省一半。
2. **`_prune_preprocessing_for_thinker`**：mm_inputs 被重写成"只剩下游还要用的元数据"——
   grid 张量、音频长度、`video_second_per_grid`、`use_audio_in_video`
   （且统一 cast dtype：grid/lengths→long，second_per_grid→float）。原始像素早已不在。
3. **fallback 顺序**：`image_out.get("image_grid_thw")` 优先取 encoder 输出，
   缺失才退回 preprocessing 的 mm_inputs——encoder 可能被缓存命中跳过，但 grid 恒在。

### 1.2 build_thinker_inputs：键的组装规则（merge.py:66-155）

产出的 `model_inputs` 只包含**非空**项（`_non_empty` 对张量判 `numel>0`）：

```
image_embeds / video_embeds
image_deepstack_visual_embeds + video_deepstack_visual_embeds   # 同时存在才用双键
deepstack_visual_embeds                                          # 单模态时退化的统一键
audio_embeds
image_grid_thw / video_grid_thw          # cast long
feature_attention_mask / audio_feature_lengths   # cast long
video_second_per_grid                    # cast float
use_audio_in_video                       # 仅当 True
```

`video_out = image_out`：视频与图像共享 Vision Tower（同一 encoder 阶段产出两键）。

### 1.3 media_cache_keys：pad 值哈希的种子

```python
media_cache_keys["image"] = f"image:{image_ck}"
media_cache_keys["video"] = f"video:{image_ck}"   # 图像与视频共用 encoder cache_key！
media_cache_keys["audio"] = f"audio:{audio_ck}"
```

注释（Xuesong）点破关键：**图像与视频共享同一个 encoder cache_key**（同一塔产出），
如果不加前缀，两者的 pad 值哈希会碰撞（见下节），radix 前缀会互相污染。

---

## 2. pad 值替换：radix cache 的多模态安全阀（request_builders.py:305-360）

`build_sglang_thinker_request` 在把 `input_ids` 交给 SGLang 之前做替换：

```python
for modality, orig_token_id in [("image", image_token_id), ("video", video_token_id), ("audio", audio_token_id)]:
    cache_key = media_cache_keys.get(modality)
    if cache_key is None: continue
    h = xxhash.xxh3_64(cache_key.encode()).intdigest()
    pad_val = vocab_size + h % (1 << 62)
    pad_values[modality] = pad_val
    input_ids[input_ids == orig_token_id] = pad_val      # clone 后原地替换
```

为什么要替换？SGLang 的 radix/prefix cache 按 token 序列共享前缀。
两个不同请求若都含"50 个 image_token_id 占位"，调度器会误以为这段前缀可复用——
但两段背后是不同的像素。替换后每段占位符变成**媒体内容哈希专属的假 token id**，
不同媒体自动不共享，同媒体天然共享。`vocab_size + h % 2^62` 保证：
(a) 不与真实 token 冲突；(b) 确定性（同键同值，跨请求/跨进程一致）；
(c) 62 位掩码避免 Python int 溢出到 SGLang 不接受的范围。

配套动作：

- `model_inputs["pad_values"]` 随请求带给 model runner（注入时用 pad 值反查占位位置）；
- `req._omni_mm_positions[modality] = (input_ids == match_id).nonzero(...)` 在 CPU 上
  **预计算**每个模态占位符的 prompt 绝对位置——注释（chenrui）：现在记下来，
  merge 时就不用同步 GPU mask、也不用遍历；
- `attention_mask` 从 model_inputs 里移除（SGLang 自己管理）。

---

## 3. M-RoPE：三轴位置编码的向量化实现

### 3.1 语义

Qwen3-Omni 的 RoPE 有三条轴：**时间 t / 高度 h / 宽度 w**（MRotaryEmbedding 按
3×head_dim/3 分段应用）。位置张量形状 `[3, batch, seq]`。规则：

- 纯文本段：三轴相同的 `arange`；
- 图像段：`t` 轴按帧索引 × `position_id_per_seconds`，`h/w` 轴按 patch 网格
  （`grid_h/merge, grid_w/merge`），meshgrid 展平后整体加段起点；
- 视频段：同上但 `t` 轴用 `arange(grid_t) * second_per_grid * pps`；
- 音频段：三轴相同 `arange`，长度 = `_feat_extract_output_lengths(input_len)`；
- **audio-in-video（AIV）**：音频与视频按"时间轴 id"归并排序（`np.lexsort`，
  并列时视频优先），共享同一段起始位置；
- `mrope_position_delta = max_pos + 1 - seq_len`：decode 阶段
  `位置 = seq_len + delta - 1`，delta 让多模态序列的 decode 位置接得上 prefill 末端。

### 3.2 为什么是 numpy 而不是 torch（mrope_positions.py:70-110）

`_linear_pos_ids / _vision_pos_ids / _vision_t_index / _merge_audio_in_video` 全是 numpy，
且注释强调 **FP32 求值顺序必须与 sglang 参考实现逐位一致**：

```python
# (t * second_per_grid) * pps：left-assoc；先算 sec*pps 会改变 FP32 舍入
return (t * np.float32(second_per_grid)) * np.float32(position_id_per_seconds)
```

"bit-identical to sglang HF port"是函数 docstring 的第一句话。位置编码差 1 个 ulp
可能让 attention 的 rotary 相位整体漂移，输出分布可感知地变化——
这里工程上选择了"宁可用 numpy 循环也要逐位一致"。

### 3.3 主循环结构（get_rope_index_qwen3_omni_vectorized:139-296）

按 `multimodal_nums`（图像数+视频数+音频数，AIV 时视频与音频合并计数）迭代"多模态块"：

```
st_idx = 上一块最大位置 + 1
min_ed = min(下一个 vision_start, 下一个 audio_start)   # 谁先来谁定义文本段长度
blocks.append(linear(text_len, st_idx))                 # 文本段
blocks.append(linear(bos_len, ...))                     # BOS：vision/audio 相邻时各占 2（bos+audio_start）
分派四种块：
  音频段    → linear(feat_len)
  图像段    → vision_pos_ids(t=arange*1.0*pps, h, w)
  视频段    → vision_pos_ids(t=arange*sec*pps, h, w)
  AIV 段    → _merge_audio_in_video(video_pos, audio_pos)（时间轴排序，视频并列优先）
BOS 特例：AIV 时 next_st 取最后一列最大值（oracle 是逐列追加）
尾部剩余文本 → linear(剩余长度)
```

纯文本快速路径（无 grid 时）：直接 `arange` 广播 + delta，零开销。
所有 grid/长度张量在进入前被 `.cpu()`（函数内部建的就是 CPU numpy）；
调用方 `_compute_mrope_positions`（request_builders.py:418-475）负责先把
input_ids 与各 grid 搬到 CPU 并 squeeze 成 `[3, seq]`。

### 3.4 talker 的线性捷径（mrope_positions.py:20-48）

`linear_mrope_positions(seq_len)`：`arange` 广播三轴 + delta=0。
`talker_can_use_linear_mrope` 判定可用性：**无 grid，或（有 grid 但 input_ids 里
没有 vision_start / audio_start）**。注释（guozhihao）解释了为什么有 grid 还可能线性：
talker decode 的位置 = `seq_len + delta - 1`，只要不会发出多模态段，
arange+0 与完整 M-RoPE 数学等价；一旦存在多模态段就必须走全量计算
（否则位置分叉，#1149 Part B）。talker 的 prefill 重放 thinker prompt 时
（07 篇），这个判定让纯文本+音频输入免掉整段 CPU 网格计算。

---

## 4. decode_events：文本流的增量语义（merge.py:214-293）

`decode_events` 是"token 序列 → 文本事件流"的纯函数，被 decode 阶段与终态
result 构建共用：

- `is_final=True`：整段解码（剔除 EOS）→ `text_final`；
- 每 token：追加进 `stream_state.token_ids` → `tokenizer.decode` 全量 →
  出现 `U+FFFD`（多字节字符被截断）→ **不发**，等下一个 token；
- 增量 = `decoded[len(emitted_text):]`，空增量不发；
- EOS 单独处理：直接以已累积文本发 `text_final`。

`stream_state` 挂在 `Qwen3OmniPipelineState` 里随 payload 流动，因此
decode 阶段（10 篇）与最终 result 的文本**由同一函数保证一致**。

---

## 5. 小结

1. merge 是"**减法**"的艺术：折入 thinker_inputs、清空 encoder_outs、
   削减 mm_inputs——每一次投影都在给 IPC 减肥。
2. pad 值哈希把"多模态前缀共享"这个正确性问题转化为**确定性的 id 重写**，
   并用 `_omni_mm_positions` 把代价一次性预付。
3. M-RoPE 实现的底线是**逐位对齐参考实现**：numpy、固定求值顺序、
   AIV 归并的 tie-break 都精确到注释。
4. delta 的存在让 prefill/decode 位置无缝衔接；线性捷径的守卫条件
   （无多模态段）必须与 decode 位置的数学定义严格对应。

下一篇（06）：这份精心构造的请求如何进入 SGLang 引擎并被逐层注入。
