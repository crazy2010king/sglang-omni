# Qwen3-TTS 深度解析(一):总体架构与全景导览

> 系列导航:本篇先建立整体框架,后续 7 篇逐一深入。
> 02 流水线与请求构建 · 03 AR 引擎与 Talker 模型 · 04 CodePredictor 与 CUDA 图 · 05 采样体系与 Triton 内核 · 06 ModelRunner 执行流水线 · 07 流式声码器与增量编解码 · 08 与 Qwen3-Omni 深度对比

---

## 0. 这篇文档讲什么,以及与旧文档 `qwen3-tts.md` 的关系

`qwen3-tts.md` 是一份概览(约 7.7KB),给出了正确的骨架:**三阶段流水线、三种任务模式、预测器 CUDA 图、双种子采样**。但它有两处已经和当前代码脱节:

| 旧文档的说法 | 当前代码的真实情况 |
|---|---|
| "Qwen3TTSModelRunner 维护设备驻留掩码(`_shape_masks`),用 `_mask_fingerprint` 判断掩码是否需要重建"(`model_runner.py` L119-200) | **已重构**。当前 `model_runner.py` 里既没有 `_shape_masks` 也没有 `_mask_fingerprint`;重复惩罚**完全让位给 SGLang 原生的 `BatchedRepetitionPenalizer`**(runner 明确 no-op),新增的复杂度转移到了 **retraction(回收重算)时如何恢复惩罚历史**(`_restore_repetition_penalty_history`) |
| 未提及 | 预处理 → AR 引擎之间通过**模块级注册表 `_PREPARED_REQUESTS` 进程内交接**重量级预处理产物;`input_ids` 不是真 token,而是**对逐行 embedding 做 blake2b 哈希得到的伪 token id**,用于 radix cache 前缀复用 |

本系列以当前代码为准,同时标注这类"文档漂移"。

---

## 1. 一句话定位

**Qwen3-TTS 是一个离散多码本 TTS 模型(12Hz 帧率,24kHz 输出),SGLang-Omni 把它接成一个 3 阶段流水线:**

```
preprocessing(CPU/GPU 预处理) → tts_engine(SGLang AR 骨干 + code predictor) → vocoder(流式声码器)
```

模型权重来自官方 `qwen-tts==0.1.1` 包(0.6B / 1.7B 两个 Base 检查点),但**推理引擎不是 transformers generate,而是把 `Qwen3TTSTalker` 重写为 SGLang 原生模型**(`sglang_model.py`,1858 行,全模块最大的单文件),获得 SGLang 的 KV cache 管理、radix cache、CUDA graph、连续批处理等全部能力。

与 Qwen3-Omni 的根本差别一句话:**Omni 是"理解 + 说话"的多模态大模型(thinker 生成文本的同时流式喂给 talker);TTS 是纯 TTS,没有上游 thinker,AR 骨干自己就是"说话者",条件信息(参考音频/音色/指令)在预处理阶段一次性烧进 prompt embedding 里。**

---

## 2. 模块地图(全部 7180 行,按职责分层)

```
sglang_omni/models/qwen3_tts/
├── config.py            121 行  PipelineConfig:3 阶段拓扑、确定性推理开关、模型路径嗅探
├── stages.py            203 行  3 个 stage 工厂 + qwen-tts 兼容补丁挂载点 + HF config 注册
├── compat.py            132 行  qwen-tts 0.1.1 × Transformers 5.12 兼容垫片(纯猴子补丁)
├── engine_builder.py    135 行  TtsEngineBuilder:引擎参数、模型挂载、runner/adapters 装配
├── payload_types.py      35 行  Qwen3TTSState:每请求声明式状态(wire 编解码声明)
├── request_builders.py 1398 行  ★ 任务解析、prompt 构建、参考音频编码缓存、请求适配器
├── model_runner.py      370 行  ★ Qwen3TTSModelRunner:阶段钩子 + code 收集 + logit 整形
├── sglang_model.py     1857 行  ★★ Qwen3TTSTalker:SGLang 原生模型 + 预测器 CUDA 图
├── sampling_kernels.py  576 行  ★ 确定性采样 Triton 内核(murmur3+Gumbel、bitonic parity)
├── predictor_kernels.py 143 行  fused gather-embedding+accumulate 内核(仅图捕获期)
├── incremental_codec.py 471 行  有状态增量 codec 解码器(卷积历史+transformer KV)
└── streaming_vocoder.py 1722 行  ★★ 流式声码器调度器:双异步 worker、pinned 传输、CUDA 图
```

★ 标记的五个文件是理解本实现的硬核所在;`sglang_model.py` 与 `streaming_vocoder.py` 是两座大山。

## 3. 三阶段流水线(来自 `config.py` 的权威定义)

`Qwen3TTSPipelineConfig.stages`(`config.py:44-68`):

```
┌───────────────┐    payload(data=state字典+prepared标记)   ┌──────────────┐
│ preprocessing │ ─────────────────────────────────────────▶ │  tts_engine   │
│ process=pipeline│  只带轻量状态字典,重产物走进程内注册表     │ (EngineStage) │
└───────────────┘                                            └──────┬───────┘
                                                    stream_to=["vocoder"]│
                                                          逐帧 code chunk │ + 终帧 result
                                                                          ▼
                                                              ┌──────────────────┐
                                                              │     vocoder      │
                                                              │ terminal=True    │
                                                              │ can_accept_stream │
                                                              │ _before_payload   │
                                                              └──────────────────┘
```

关键事实:

- **三个 stage 都声明 `process="pipeline"`,都在 `gpu=0`**——单进程单卡部署。这不是简化,而是一个刻意的架构选择:预处理需要访问 AR 引擎进程里加载好的 `model.speech_tokenizer` / speaker encoder,`process_local_edges()`(`config.py:24-27`)显式声明了 `(preprocessing, tts_engine)` 是同进程边,引擎构建时通过 `set_qwen3_tts_preprocessing_context()`(`request_builders.py:110`)把模型对象注册到模块级全局。
- `tts_engine` 是 `EngineStageConfig`——唯一由 SGLang 引擎驱动的 stage;`stream_to=["vocoder"]` 使 code 帧**不等整句生成完**就流向声码器(流式 TTS 的前提)。
- `vocoder` 是 terminal 且 `can_accept_stream_before_payload=True`——它可以在正式 result payload 到达前先消费 stream chunk(与 `tts_engine` 的流式输出形成协议配合)。

**对比 Omni**:Omni 的 speech 拓扑是 **7 个 stage**(`preprocessing → image_encoder/audio_encoder(并行)→ thinker → decode + talker_ar → code2wav`),有 DAG 的 fan-out(fan-in)、`wait_for`/`merge_fn`/`route_fn`、跨进程 payload 投影(`project_payload`);而 TTS 是**纯线性链**,没有 join 语义。拓扑复杂度的差别直接反映了任务差别:Omni 必须把文本理解与语音生成做成两条并行的消费流,TTS 只有一条。

## 4. 一个请求的完整生命周期(端到端数据流)

以 voice-clone 请求(Base 任务)为例,标注每一步涉及的文件与函数:

```
HTTP /v1/audio/speech
  │
  ▼
① preprocessing 线程池(ThreadedSimpleScheduler, max_concurrency=8)
   preprocess_qwen3_tts_payload()                      request_builders.py:832
   ├─ build_qwen3_tts_state()         解析 task_type/voice/ref_audio/seed → Qwen3TTSState
   ├─ _validate_qwen3_tts_model_task() 检查点类型 × 任务类型交叉校验(硬失败)
   ├─ _prepare_qwen3_tts_base_request()
   │    ├─ 参考音频 → voice_clone_prompt(ref_code + ref_spk_embedding)
   │    │    · 上传音色:SpeakerCacheKey 命中 speaker 缓存
   │    │    · 临时音频:ReferenceEncodeService(256 项/64MB LRU,按内容哈希键控)
   │    │      └─ _Qwen3TTSRefCodeBatcher:专用线程 + 专用 CUDA stream 批量编码
   │    ├─ wrapper._tokenize_texts(...) 分词
   │    └─ model.build_voice_clone_inputs(...)  ★ 组装 prompt embedding(03 篇详解)
   ├─ input_ids_list = blake2b(每行 embedding)    ★ 伪 token id,radix cache 键
   └─ _PREPARED_REQUESTS[request_id] = Qwen3TTSPreparedRequest   ★ 进程内交接
  │
  ▼  StagePayload.data = 状态字典 + {"_qwen3_tts_prepared_request": rid}
② tts_engine(OmniScheduler + Qwen3TTSModelRunner + Qwen3TTSTalker)
   build_sglang_qwen3_tts_request()                    request_builders.py:858
   ├─ pop_prepared_qwen3_tts_request() 取走重产物(缺失即硬错误,防止重复预处理)
   ├─ derive_qwen3_tts_sampling_seeds(seed) → (semantic, subtalker) 双种子
   └─ 构造 sglang.Req(extra_key="qwen3_tts:prompt:v1", sampling_seed=semantic)
   │
   │  每个 decode step(ModelRunner.execute,06 篇):
   │    before_decode → prepare_decode_buffers + 写 feedback buffer
   │    forward(Qwen3TTSTalker)→ codec_head logits → SGLang sample(带 seed)
   │    post_decode → _collect_codes:
   │        layer0_code + hidden → code_predictor_forward(★ 04 篇:预测其余 Q-1 层)
   │        快照 _output_codes/_output_embeds → output_codes.append(...)
   │        latest_stream_code_chunk = 本帧 chunk
   │    stream_output_builder()                          request_builders.py:1005
   │        首块把 ref_code 拼在前面 + metadata{num_quantizers, ref_code_len}
   │        → OutgoingMessage(type=stream, target="vocoder")
   │    终止:codec_eos_token_id → apply_sglang_qwen3_tts_result
   │        codes = cat(ref_code, stack(output_codes)) → 完整 [T, Q] 码本
  │
  ▼  双通道到达 vocoder
③ vocoder(Qwen3TTSStreamingVocoderScheduler,07 篇)
   ├─ 流式路径:ingest 码块 → 阈值触发 decode_plan(带 16 帧 left context 窗口)
   │    → 双异步 worker(initial + followup,按 playback deadline 优先级)
   │    → CUDA graph chunked_decode(或增量 codec 解码器)
   │    → pinned 缓冲 D2H → OutgoingMessage(type=stream, modality=audio)
   └─ 非流式路径:fallback_full_decode → 整段 waveform
  │
  ▼
HTTP 音频响应(24kHz PCM)
```

记住这条主线,后面每一篇都是在放大其中一个方框。

## 5. 三种任务模式(Base / CustomVoice / VoiceDesign)

`build_qwen3_tts_state()`(`request_builders.py:222`)是三种模式的分叉点,规则矩阵:

| | Base(声音克隆) | CustomVoice(预设音色) | VoiceDesign(音色设计) |
|---|---|---|---|
| 必需输入 | `ref_audio` + `ref_text`(或 `x_vector_only_mode`) | `voice`(缺省 `"Vivian"`) | `instructions` |
| 禁止输入 | — | ref_audio / ref_text / x_vector_only_mode | ref_audio / ref_text / x_vector_only_mode |
| `non_streaming_mode` | 默认 False(可流式) | **强制 True** | **强制 True** |
| 说话人条件 | x-vector + (可选 ICL ref_code) | 查 `config.spk_id` 表(键小写匹配) | 无(纯指令) |
| 语言 id 推断 | `language` 参数 | `spk_is_dialect[voice]` 可覆盖 `auto` | 同左 |
| prompt 构建函数 | `build_voice_clone_inputs` | `build_custom_voice_inputs` | `build_voice_design_inputs` |

两个容易被忽略的硬约束:

1. **任务类型可以隐式推断**(`normalize_qwen3_tts_task_type`):不带 `task_type` 时,有参考音频 → Base,没有 → CustomVoice。推断结果与检查点类型不匹配时,`_validate_qwen3_tts_model_task` 区分"用户显式要求了不支持的任务"与"用户没给够信息"两种错误文案——Base 检查点收到纯文本请求时,若任务类型是推断出来的,错误信息会明确提示"需要 CustomVoice 或 VoiceDesign 检查点",这是面向用户的可诊断性设计。
2. **CustomVoice/VoiceDesign 强制非流式**(`resolve_non_streaming_mode`,`request_builders.py:479`):这两种模式的 prompt 组装走 `_finish_text_prompt` 的非流式分支(全部文本 embedding 一次性进 prefill,见 03 篇),`stream_codec_output=False` 也让 AR 引擎不产 stream 消息,声码器走整段解码路径。

## 6. 与 Qwen3-Omni 的架构级对照(总览,详细版在 08 篇)

| 维度 | Qwen3-TTS | Qwen3-Omni |
|---|---|---|
| 代码体量 | ~7,180 行 | ~13,468 行 |
| 流水线拓扑 | 3 阶段线性链 | 6(纯文本)/ 7(speech)阶段 DAG,带 fan-out/join |
| 部署形态 | 单进程单卡(`process="pipeline"` ×3) | 分散式(thinker GPU0/talker GPU1)或同卡共置,7 个进程 |
| 上游条件 | 一次性 prompt(参考音频/音色/指令,预处理期烧入 embedding) | thinker **逐 token 流式**输出 hidden states 到 talker |
| Talker 结构 | Dense MLP,talker 仅一个 `text_projection` | MoE(`Qwen3OmniMoeTalkerSparseMoeBlock`)+ 双投影(text/hidden)+ deepstack 第 N 层状态 |
| 说话人建模 | x-vector / spk_id 表 / 指令 | 固定 speaker map(`resolve_speaker_id`,缺省 Ethan) |
| 采样 | 双种子(semantic/subtalker)+ 自研 Triton 确定性内核 + 图内采样签名 | SGLang 标准 sampler(静态 SamplingBatchInfo,图内)+ 预测器纯 argmax |
| 流式接口 | `stream_to=["vocoder"]` code 帧 | thinker `stream_to=["talker_ar","decode"]`:hidden states 与文本 token 双流 |
| 首音延迟优化 | 小初始块(8 帧)+ 声码器异步 worker | 部分启动(`MIN_PARTIAL_START_CHUNKS=3`,`partial_start_min_chunks=5`) |
| 位置编码 | 普通 RoPE(转发 `mrope_positions` 仅为兼容) | 真 M-RoPE(`mrope_positions.py` 331 行,多模态交错) |
| 共享代码 | **两者共用**:解码层/注意力(`Qwen3OmniMoeThinkerTextAttention`)、DenseMLP、`ResizeMLP`、`apply_qk_norm`、`PendingTextTensorQueue`、`QwenTalkerModelRunner` 的 decode-embed 静态方法 | 同左(代码宿主在 `qwen3_omni/`) |

值得强调的复用关系:`qwen3_tts/model_runner.py` 直接 `from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner`,复用其 `_take_next_decode_input_embed`、`_append_decode_input_history`、`_projected_prefill_slice` 静态方法;`qwen3_tts/sglang_model.py` 复用 omni 的 `Qwen3OmniMoeTalkerDenseMLP`、`ResizeMLP`、`_bind_default_weight_loaders`。**Qwen3-TTS 的 talker 实质上是 Omni talker 的"无 MoE、无流式上游"近亲**,这是理解 03 篇的最短路径。

## 7. 底座框架:三个必须先建立的概念

这三个概念属于 sglang-omni 通用框架,TTS 与 Omni 站在同一底座上:

**(1)声明式请求状态(`payload_types.py`,35 行)。**
`Qwen3TTSState` 用 `wire(默认值, codec="int")` 声明字段如何跨 stage 序列化。`codec` 声明告诉框架该字段在 StagePayload 字典化/恢复时的编解码方式(`str_or` 表示可空字符串、`tensor_list` 表示张量列表等)。跨进程时只有被 `wire` 的字段会传播——这是"payload 投影"机制的声明式版本。对比 Omni 的 `Qwen3OmniPipelineState`(121 行):字段更多(含 `prompt`/`thinker_inputs`/`encoder_outs`),因为要在多 stage 间搬运多模态中间产物。

**(2)ModelRunner 钩子协议(`model_runner/base.py`,1007 行)。**
`ModelRunner.execute()` 是所有 AR 模型的共享执行管线:

```
_build_forward_batch → before_{prefill,decode} → forward(custom 或标准)
  → post_{prefill,decode} → sample → _publish_next_tokens → _finalize
```

子类只覆写钩子。异步解码(one-step lookahead)把 `execute` 拆成 `execute_launch`(GPU 侧:forward+sample+发布,记 event)/ `execute_resolve`(host 侧:等 event、读 launch_buf、收集)——两个时刻之间允许调度器塞入下一步 launch,即一步前瞻。TTS 的 runner 没有覆写 launch/resolve(它用同步 `execute`),Omni 的 talker 也没有;真正用上 lookahead 的是 thinker(`enable_async_decode=True`)。

**(3)StreamingVocoderBase 模板方法(`scheduling/streaming_vocoder.py`,513 行)。**
流式声码器调度器的生命周期骨架:状态注册表、stream 合同锁定(`latch_stream_contract`)、`ingest → should_decode → decode_delta → emit` 的阈值驱动循环、abort/stop 清理、全量兜底解码。模型只需实现 6 个抽象钩子。TTS 的 `Qwen3TTSStreamingVocoderScheduler` 与 Omni 的 `Code2WavScheduler` 是这个基座的两个实现(07 篇逐钩子对比)。

---

## 8. 阅读路线建议

- 想搞懂"prompt 是怎么拼出来的":03 篇(02 篇先看请求解析)。
- 想搞懂"12 层码本怎么一次生成"或"CUDA 图怎么绕过 host 分支":04 篇。
- 想搞懂"同样 seed 为什么两次生成完全一致":05 篇(bitonic 网络与 torch.topk 的逐位对齐是全系列最硬核的部分)。
- 想搞懂"为什么 input_ids 是 embedding 的哈希":02 篇 §radix cache。
- 想搞懂"声码器怎么做到边生成边播且不崩 CUDA 上下文":07 篇。
- 想横向理解两个模型的取舍:08 篇。
