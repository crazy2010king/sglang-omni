<!--
本文件由原 8 篇系列文档(Qwen3-TTS-01 ~ 08)无损合并而成,原文件已删除。
各篇标题保留为一级章节,内容与原系列一致。
-->
# Qwen3-TTS 深度解析(合并版)

> 基于 `sglang_omni/models/qwen3_tts/`(7,180 行)与 `qwen3_omni/`(13,468 行)现状代码的逐层精读。
> 与旧概览 `qwen3-tts.md` 的漂移处在对应章节标注。

## 目录

- 第一篇 总体架构与全景导览
- 第二篇 流水线阶段与请求构建
- 第三篇 AR 引擎与 Talker 模型
- 第四篇 CodePredictor 与 CUDA 图加速
- 第五篇 采样体系与 Triton 内核
- 第六篇 ModelRunner 执行流水线
- 第七篇 流式声码器与增量编解码
- 第八篇 与 Qwen3-Omni 的深度对比

---

# Qwen3-TTS 深度解析(一):总体架构与全景导览

> 系列导航(本合并文档内的八个部分):第一篇建立整体框架,后续七篇逐一深入。
> 第二篇 流水线与请求构建 · 第三篇 AR 引擎与 Talker 模型 · 第四篇 CodePredictor 与 CUDA 图 · 第五篇 采样体系与 Triton 内核 · 第六篇 ModelRunner 执行流水线 · 第七篇 流式声码器与增量编解码 · 第八篇 与 Qwen3-Omni 深度对比

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

# Qwen3-TTS 深度解析(二):流水线阶段、引擎装配与请求构建

> 本篇覆盖 `config.py`(121 行)、`stages.py`(203 行)、`compat.py`(132 行)、`engine_builder.py`(135 行)、`request_builders.py`(1398 行)。

---

## 1. PipelineConfig:三阶段的静态声明

`Qwen3TTSPipelineConfig`(`config.py:15`)是 pydantic 配置类,框架通过 `EntryClass = Qwen3TTSPipelineConfig` 发现它。逐项拆解:

### 1.1 类变量协议

```python
architecture: ClassVar[str] = "Qwen3TTSForConditionalGeneration"   # 检查点架构匹配用
requires_model_capabilities: ClassVar[bool] = True                  # 走 models/registry.py 能力协商
stage_config_types: ClassVar[dict] = {"tts_engine": EngineStageConfig}  # 哪个 stage 是引擎态
```

`requires_model_capabilities=True` 意味着 YAML/CLI 配置里可以要求模型具备某种能力标记,由 `models/model_capabilities.py` 校验。`stage_config_types` 是框架区分"普通 stage"与"SGLang 引擎 stage"的类型映射——只有引擎 stage 会走 `engine_builder` 装配路径并占用 SGLang 的 KV cache 池。

### 1.2 `process_local_edges`:同进程边的显式契约

```python
@classmethod
def process_local_edges(cls) -> frozenset[tuple[str, str]]:
    return frozenset({("preprocessing", "tts_engine")})
```

这条声明的存在理由藏在注释里:预处理把重产物放进**模块级注册表 `_PREPARED_REQUESTS`**,AR 引擎构建时同进程读取。框架据此知道这两个 stage 之间的 payload 只传轻量字典、重张量不经 IPC 序列化——跨进程部署这里会直接失效,所以被显式约束为进程内边。**这是整条流水线最"脏"但最务实的设计**:预处理要调 GPU 上的 speech tokenizer 和 speaker encoder,而这两个模型物理上挂在 AR 引擎进程里;与其跨进程传 wav + 回传 code,不如直接借用引擎进程的模型。

对比 Omni:Omni 完全没有这种模块级全局,stage 间全部走显式 `StagePayload` 投影(`project_payload` 函数族),代价是要写 1199 行 `request_builders.py` 里的投影/合并函数。TTS 用 132 行的注册表代码换掉了这一整套,前提是单进程拓扑被锁死。

### 1.3 确定性推理的代价清单

```python
def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
    if not self.enable_deterministic_inference:
        return {}
    if stage_name == "preprocessing":
        return {"max_concurrency": 1}                        # 串行化预处理
    if stage_name == "tts_engine":
        return {"server_args_overrides": {"enable_deterministic_inference": True}}
    if stage_name == "vocoder":
        return {"enable_deterministic_inference": True,
                "initial_cuda_graph": False,                 # 关掉声码器 CUDA 图
                "followup_cuda_graph": False}
```

注释说明这是 opt-in 而非默认:它同时**串行化预处理、关掉声码器 CUDA 图、降低吞吐**。深意在于:确定性不是"打开一个开关"就能全局成立的属性——三个 stage 各有自己的非确定性来源(线程并发、CUDA 图重放时 batch 补零、声码器流式窗口切片),必须逐个收缴。07 篇会看到确定性模式下声码器还要退化为逐请求 B=1 解码。

### 1.4 模型路径嗅探

`_is_qwen3_tts_base_model()`(`config.py:98-118`)把 model_path 按 `/\` 切开后做 `casefold + 连字符归一`,路径段含 `custom_voice/voice_design` 标记 → 非 Base;含 `_base` 或 `_base_` → Base。它决定:

- `requires_uploaded_voice_for_named_voice()` / `supports_uploaded_voice_references()`:只有 Base 检查点接受上传音色(需要 x-vector/ICL 提取);CustomVoice/VoiceDesign 检查点的音色是表驱动的,天然不支持。

用路径做能力判断当然脆弱,但它是唯一在"加载权重之前"就能拿到的信息——引擎还没起,没法读 config 里的 `tts_model_type`。真正的权威校验在请求期的 `_validate_qwen3_tts_model_task`(§3.1)。

## 2. 兼容垫片:`compat.py` 的三处猴子补丁

`qwen-tts 0.1.1` 的模型代码是按 Transformers 4.57 写的,而 SGLang-Omni 运行栈是 Transformers 5.12。`apply_qwen_tts_transformers_compatibility_patches()`(`compat.py:78`)是**所有** qwen3-tts 入口(stages 加载 tokenizer、引擎构建、weight 加载、speaker encoder 实例化)的第一行调用,做三件事:

1. **`ROPE_INIT_FUNCTIONS.setdefault("default", _compute_default_rope_parameters)`**:5.12 移除了 `"default"` 键。补的实现按 `rope_theta`/`partial_rotary_factor`/`head_dim` 算 inv_freq,签名兼容 5.12 的 `(config, device, seq_len, layer_type)`。
2. **mask 工厂改名垫片**(`_patch_mask_factories`):5.12 把 `create_causal_mask` / `create_sliding_window_causal_mask` 的参数从 `input_embeds` 改成 `inputs_embeds` 并移除了 `cache_position`。垫片用 `inspect.signature` 确认确实是新签名后才包一层,`kwargs.setdefault("inputs_embeds", kwargs.pop("input_embeds"))` + `kwargs.pop("cache_position", None)`。补丁函数打上 `_sglang_omni_qwen_tts_compat_patched` 防重复,并用 `threading.Lock` 保证并发导入安全。
3. **`check_model_inputs` 装饰器双形态兼容**:4.57 支持裸函数调用(非装饰器用法),5.12 只支持装饰器。垫片检测原函数签名是否"单位置参数且无默认值",是则包一层使其两种用法都工作。

设计原则值得学习:**垫片永远先检查再替换,永远幂等**;`cookbook/qwen3_tts.md` 明确禁止用 qwen-tts 自己的依赖解析解决这类报错(会连坐升级 numpy/transformers 打破整个运行时),报错应回到垫片层修。

另一个必须理解的补丁在 `stages.py:_register_qwen3_tts_hf_config`:`Qwen3TTSConfig.__init__` 被包一层,把 `self.text_config = talker_config`——因为 SGLang 侧代码(以及 `Qwen3TTSTalker`)统一从 `config.talker_config` 取子配置,而某些 transformers 路径读 `text_config`。随后 `AutoConfig.register("qwen3_tts", Qwen3TTSConfig)` 让 `AutoConfig.from_pretrained` 能识别架构。

## 3. 引擎装配:`engine_builder.py`

`Qwen3TtsEngineBuilder(TtsEngineBuilder)` 是框架与模型之间的装配协议,按调用时序:

### 3.1 引擎默认参数(`generation_defaults`,`engine_builder.py:56-72`)

```python
{
    "max_running_requests": 16, "max_queued_requests": 16,
    "cuda_graph_max_bs": 32, "torch_compile_max_bs": 32,
    "dtype": dtype, "disable_cuda_graph": False,
    "disable_overlap_schedule": True,          # ← 注意
    "enable_torch_compile": False,
    "mem_fraction_static": 0.85, "max_prefill_tokens": 8192,
    "sampling_backend": "pytorch", "trust_remote_code": True,
}
```

两个参数是硬约束的信号:

- **`disable_overlap_schedule: True`**:SGLang 的 overlap scheduler 让 batch 准备与 GPU 前向重叠,但要求 forward 是"纯函数"——而 TTS 的 forward 依赖每步之前写入的 `_decode_feedback_embedding`/采样参数暂存缓冲(06 篇),batch 准备路径被自定义 runner 深度接管,overlap 语义不再成立。Omni 的 thinker 恰恰相反(`enable_async_decode=True`,还要做一步前瞻),因为它的 forward 才是标准的"读 input_ids 出 logits"。
- **`sampling_backend: "pytorch"`**:05 篇会看到,`multinomial_with_seed` 的 float64 端点行为被 Triton 内核逐位复刻,flashinfer 后端不支持 seeded sampling 的 top-k/top-p 组合(base runner 里 `_validate_seeded_sampling_supported` 会直接 raise)。

`adjust_overrides` 里还有一条硬拒绝:`enable_torch_compile=True` 直接 ValueError——torch.compile 无法处理 `_output_codes` 等持久缓冲的就地写模式。

### 3.2 模型挂载(`setup_model`,`engine_builder.py:74-100`)

```python
model = model_worker.model_runner.model                 # SGLang 已加载的 Qwen3TTSTalker
speech_tokenizer = qwen3_stages._load_qwen3_tts_tokenizer(...)
model.load_speech_tokenizer(speech_tokenizer)           # 挂在模型对象上
processor = AutoProcessor.from_pretrained(checkpoint_dir, fix_mistral_regex=True)
self.wrapper = Qwen3TTSModel(model=model, processor=processor,
                             generate_defaults=...(generation_config.json))
request_builders.set_qwen3_tts_preprocessing_context(model=model, wrapper=self.wrapper)
```

`Qwen3TTSModel` wrapper 是 qwen-tts 包的上游类,这里只用它的高层工具(`_tokenize_texts`、`_build_assistant_text`、`_normalize_audio_inputs`、`create_voice_clone_prompt`、`_merge_generate_kwargs`),**生成主循环完全不用它**。`set_qwen3_tts_preprocessing_context` 同时预热了 ad-hoc 参考编码服务(§4.4),并清空 `_PREPARED_REQUESTS`——重建引擎(如热重载)时旧预处理产物必须作废。

### 3.3 适配器装配(`make_adapters` / `extra_scheduler_kwargs`)

```python
request_builder, result_adapter, self._stream_output_builder = (
    request_builders.make_qwen3_tts_scheduler_adapters(model=model, wrapper=self.wrapper))
```

三个函数对象构成 OmniScheduler 的适配层:

| 适配器 | 职责 | 方向 |
|---|---|---|
| `request_builder` | StagePayload → `Qwen3TTSSGLangRequestData`(含 sglang.Req) | 进入引擎 |
| `result_adapter` | 请求数据 → 终态 StagePayload | 离开引擎 |
| `stream_output_builder` | 每步输出 → `OutgoingMessage(type=stream, target=vocoder)` | 中途流式 |

`extra_scheduler_kwargs` 附带 `request_build_max_workers: 4, request_build_max_pending: 16`:请求构建(即 `build_sglang_qwen3_tts_request`,含 pop prepared、构造 Req、seed 派生)在线程池并行,上限防积压。

`model_arch_override = "Qwen3TTSTalker"`:SGLang 按 `config.architectures` 找模型类,这里强改为我们自己的入口类名;`supports_breakable_prefill_cuda_graph` 从 CAPABILITIES 透传——预填充 CUDA 图支持"可断段"(03 篇 QK-norm/RoPE 段强制 eager 就依赖它)。

## 4. `request_builders.py` 上半场:请求解析与状态构建

### 4.1 参数的三层来源与优先级

`build_generation_kwargs()`(`request_builders.py:494`)的合成顺序:

```
stage_params["tts_engine"][field]  >  params[field]  >  隐式默认过滤  >  硬默认
```

精妙处在**隐式默认过滤**:

```python
_IMPLICIT_SAMPLING_DEFAULTS = {
    "temperature": {1.0, 0.8}, "top_p": {1.0, 0.8},
    "top_k": {-1, 30}, "repetition_penalty": {1.0, 1.1},
}
```

如果用户没有显式传 `explicit_generation_params`,而这些字段取值恰好是上游 OpenAI 兼容客户端的"默认值"(1.0/0.8/30/1.1 这类),则**丢弃不进 generation_kwargs**。为什么?因为下游要区分"用户显式要求 temperature=0.9"和"用户没提 temperature";前者覆盖 wrapper 的 `generate_defaults`(checkpoint 里的 `generation_config.json`),后者应完全沿用检查点默认。不做这个过滤,客户端框架无意识填充的默认值就会悄悄覆盖模型作者调好的采样配置。这是 API 兼容层里非常经典的"显式性传播"问题。

### 4.2 状态对象的全部字段即全部请求语义

`Qwen3TTSState`(`payload_types.py:14`)24 个字段中值得盯住的:

- `task_type_explicit`:任务类型是显式指定还是推断(§1.4 的错误文案区分依赖它)。
- `x_vector_only_mode`:无 ref_text 的"仅说话人向量"模式;ICL(上下文学习)模式下 ref_code 会被编进 prompt,此模式下 ref_code 置 None,只用 speaker embedding。
- `non_streaming_mode`:决定 prompt 结构与 vocoder 路径(03/07 篇)。
- `ref_code_len`(wire, `emit="truthy"`):参考音频码长,沿 pipeline 传到 vocoder 用于裁掉参考段音频。

### 4.3 种子派生:一次请求,两个确定性源

```python
def derive_qwen3_tts_sampling_seeds(seed: int) -> tuple[int, int]:
    normalized = _normalize_qwen3_tts_seed(seed)          # & SAMPLING_SEED_MASK
    return (_derive_qwen3_tts_child_seed(normalized, "semantic"),
            _derive_qwen3_tts_child_seed(normalized, "subtalker"))
```

`derive_sampling_seed("qwen3-tts", seed, label)` 是框架级种子派生(同 namespace 下 label 不同产生独立流)。语义层(SGLang sampler)与子说话人层(预测器内采样)各有独立种子,但都由同一个公开 seed 决定——用户换一个 seed,两层同时变;用户复现,两层都复现。`_normalize_qwen3_tts_seed` 对 bool/非整浮点显式报错(bool 是 int 子类,不拦会静默当 0/1)。

### 4.4 参考音频编码的三级缓存体系

这是 `request_builders.py` 最具工程密度的部分。三级结构:

**第一级:上传音色缓存(`speaker_cache`)。** `_qwen3_tts_uploaded_voice_cache_key()` 构造 `SpeakerCacheKey(model_type="qwen3_tts_icl"|"qwen3_tts_xvec", voice_name, voice_version=created_at, artifact_kind="voice_clone_prompt")`。命中则直接 `_qwen3_tts_voice_prompt_from_cache` 反序列化(张量 detach+clone 回设备)。注意 `voice_version` 用 `uploaded_voice_created_at` 参与键控——同名音色被用户重新上传后旧缓存自然失效。

**第二级:ad-hoc 参考服务(`ReferenceEncodeService`)。** `_get_qwen3_tts_adhoc_reference_service_locked` 以 `(id(model), id(wrapper))` 为属主单例,LRU 上限 256 项 / 64MB / 130s 超时。键控在 `_Qwen3TTSAdhocReferenceHook`:

```python
def input_key(self, item) -> str | None:      # 音频内容哈希
    # data: URI → "data:"+hash;路径 → reference_path_cache_key(信任 stat);bytes → hash
def options_key(self, item) -> str:           # {"ref_text":..., "x_vector_only_mode":...} 规范 JSON
```

输入键与选项键分离:同一段音频换 ref_text 不复用缓存(正确!ICL prompt 依赖 ref_text)。`encode_one` 是实际编码:

```python
normalized = self._wrapper._normalize_audio_inputs([item.ref_audio])   # → (waveform, sr)
ref_code_future = self._ref_code_batcher.submit(waveform, sample_rate) # 异步提交
if sample_rate != speaker_sample_rate:                                 # 24k
    speaker_waveform = librosa.resample(...)
speaker_embedding = self._model.extract_speaker_embedding(audio=..., sr=...)
ref_code = _record_ref_code_consumer_stream(ref_code_future.result(timeout=130))
```

**第三级:`_Qwen3TTSRefCodeBatcher` 批处理线程。** 独立守护线程 + `queue.Queue`,核心状态机:

- `submit()` 只入队返回 Future;`encode()` = submit + result(130s 超时)。
- `_drain()` 攒批:首个阻塞取,然后以 `max_batch_wait_ms=2` 为窗攒到 `max_batch_size=8`。
- `_run()` **按采样率分组**(speech tokenizer 的 `encode(waveforms, sr)` 要求同 sr),组内批量编码;批编码失败时**逐条重试**降级,单条失败则 Future 携带异常(错误归属到具体请求,不炸整批)。
- **专用 CUDA stream**(`_encode_stream`):编码 kernel 全部跑在私有流上,`_synchronize_outcomes` 用 `Event.record(stream)+event.synchronize()` 等待——注释点明关键:**等待编码流的事件不会触碰默认流,默认流上并发跑的 speaker-embedding kernel 不被牵连**。设备无法预解析的老 tokenizer 退化为按设备同步默认流。
- `_record_ref_code_consumer_stream`:ref_code 可能分配在批处理私有流上,通过 `record_stream(当前流)` 向 caching allocator 注册消费流——否则下一批可能回收该显存块而消费端 kernel 还在读。**这是跨流张量生命周期的教科书式处理**,任何自己写多流代码的人都会在这里踩坑。

## 5. `request_builders.py` 下半场:请求适配器

### 5.1 伪 token id:radix cache 的哈希键

```python
def build_embedding_cache_key_ids(input_embeds: torch.Tensor) -> list[int]:
    rows = input_embeds.detach().to(dtype=torch.float32, device="cpu")
    for row in rows:
        digest = hashlib.blake2b(row.numpy().tobytes(), digest_size=8).digest()
        key_ids.append(int.from_bytes(digest, "little") & ((1 << 63) - 1))
```

TTS 的 prefill 输入是**连续 embedding,没有真 token 序列**;但 SGLang 的 radix cache 以 token id 序列为键。方案:逐行 embedding → blake2b 8 字节 → 取低 63 位当"token id"。效果:

- 同样的 prompt(相同参考音频 + 相同文本)得到**逐位相同的伪序列**,radix cache 命中,跳过整个 prefill 前向。
- 不同音频几乎必然碰撞率为 0(64-bit 哈希,生日界 ≈ 2^31 个不同请求)。
- SGLang 内部 KV cache 按 token 定位,伪 id 与真实 embedding 由 `prompt_input_embeds` 侧通道对齐(runner 的 `_build_prefill_input_embeds` 按 extend 切片回填,06 篇)。

`Req` 上同时设置三个 omni 私有标记:`req._input_embeds_are_projected = True`(embedding 已在模型空间,无需再投影)、`req._omni_prompt_only_radix = True`、`req._omni_prompt_cache_key = req.extra_key`。

`extra_key="qwen3_tts:prompt:v1"` 是 **radix cache 命名空间**:SGLang 按前缀匹配时 extra_key 必须一致,不同模型的 KV 页几何不同,绝不能跨模型共享前缀树。对比 Omni 的 talker:Omni 没有设 extra_key(其 prefill embedding 每请求不同——thinker 输出不可复现,radix 复用无意义);TTS 的 prompt 是确定性的,复用收益巨大。

### 5.2 `Qwen3TTSSGLangRequestData`:调度器持有的请求态

继承框架 `SGLangARRequestData`,增量字段:`output_codes`(逐帧码本累积)、`latest_stream_code_chunk`(本步待发流块)、`stream_ref_sent`(ref 码是否已随首块发出)、`ref_code/ref_code_len`、`prompt_input_embeds`、`semantic_sampling_seed`、`subtalker_*` 全套、`engine_start_s`(perf_counter,供 engine_time_s 指标)。

### 5.3 流式输出协议:ref 码为什么由 AR 引擎前置

`stream_output_builder`(`request_builders.py:1005`)每次 decode 被调用:

1. 条件闸门:仅当 `stream_codec_output`(非 non_streaming_mode)且 `params["stream"]`,否则返回 `[]`。
2. 取 `latest_stream_code_chunk`(取走即置 None,消费语义)。
3. **首块时把 `ref_code` 拼在生成码前面**,metadata 写 `ref_code_len` 与 `num_quantizers`,置 `stream_ref_sent=True`。

为什么 ref 码要由引擎而不是 vocoder 自己拼?因为流式声码器的解码窗口需要**连续的码流**(参考段 + 生成段共享因果卷积历史,07 篇);把 ref 码并入首帧块,vocoder 的 `latch_stream_contract` 记下 `ref_frames` 后,后续窗口切片就能统一按绝对帧号算。协议一致性:非流式路径 `apply_sglang_qwen3_tts_result` 同样 `cat(ref_code, output_codes)` 并携带 `ref_code_len`——两条路径的码流合同完全一致,vocoder 无需知道自己是流式还是非流式接收。

metadata 里还透传 `initial_codec_chunk_frames` 请求参数(`INITIAL_CODEC_CHUNK_FRAMES_PARAM`)——客户端可以在请求里指定首块解码长度,声码器按它锁定流合同(07 篇)。

### 5.4 终态归一化

`_qwen3_tts_finish_reason` 处理三种来源:调度器原始 reason → 归一为 length/abort/error;缺失时(终步由 stage 自己拥有)按 `len(output_codes) >= max_new_tokens` 判 length,否则 stop。`apply_sglang_qwen3_tts_result` 汇总 `prompt_tokens=ref_code_len`、`completion_tokens=len(output_codes)`、`engine_time_s`——usage 语义里把参考段算 prompt,生成段算 completion。

---

## 6. 与 Qwen3-Omni 的对应环节对比

| 环节 | Qwen3-TTS | Qwen3-Omni |
|---|---|---|
| 预处理产物 | `Qwen3TTSPreparedRequest`(embedding/attention_mask/ref_code/...),进程内注册表 | `Qwen3OmniPipelineState`(`prompt`、`thinker_inputs`、`mm_inputs`),显式 StagePayload 传递 |
| 预处理并发 | `ThreadedSimpleScheduler(max_concurrency=8)` 对齐批大小 | 单请求/次调度(注释:每 scheduler call 一个 prompt) |
| 请求构建 | pop prepared + 构造 Req + 伪 token id + extra_key 命名空间 | `build_sglang_thinker_request`(真 token id + 多模态占位) / `build_sglang_talker_request`(codec_bos 重复序列作 dummy ids) |
| radix cache | 主动设计(哈希键,命中即跳过 prefill) | thinker 侧常规 token 前缀复用;talker 侧embedding 每请求不同,基本无复用 |
| 种子 | 双种子派生(semantic/subtalker) | 单 seed 经 `_resolve_seed(params)`,talker 侧未 seed 时用 rid 派生的 rank 共享种子 |
| 流式协议 | code 帧 + ref 前置,metadata 携带 num_quantizers | thinker 双流:hidden states→talker_ar,token_id→decode;talker 流 code→code2wav |

一个值得记住的结构性结论:**TTS 的预处理是"重"的(GPU 编码、缓存、批处理线程都在这),Omni 的预处理是"轻"的(纯 CPU 分词 + 路由);TTS 的引擎阶段是"标准"的(每步 SGLang sampler + 外挂预测器),Omni 的引擎阶段是"自定义到底"的(sampling 都在模型 forward 内部完成)。** 两头相反,中间的 runner 钩子协议是同一个。

下一篇进入引擎内部:`Qwen3TTSTalker` 的模型结构与 prompt 组装数学。

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

# Qwen3-TTS 深度解析(六):ModelRunner 执行流水线

> 本篇覆盖 `model_runner.py`(370 行)及其底座 `model_runner/base.py`(1007 行)。这是"调度器输出 → GPU 前向 → 请求状态更新"的缝合层,也是文档 `qwen3-tts.md` 与当前代码漂移最大的一篇。

---

## 1. 底座协议:`ModelRunner` 的钩子面

base.py 定义了所有 AR 模型的执行骨架,先列全钩子再讲 TTS 怎么填:

```
execute(scheduler_output)                     # 同步全流程
  ├─ _build_forward_batch                     # ForwardBatch.init_new(+ capture_hidden_mode)
  ├─ _prepare_and_forward
  │    ├─ before_prefill / before_decode      ← 子类钩子
  │    ├─ custom_prefill_forward / custom_decode_forward(返回 None 则走 tp_worker 标准前向)
  │    ├─ sample_before_post_* → _sample_next_token_ids(可选提前采样)
  │    └─ finally: clear_omni_prefill_inputs + cleanup_prefill
  ├─ post_prefill / post_decode               ← 子类钩子
  ├─ _ensure_next_token_ids / _publish_next_tokens   # 令牌轨道发布(下一步前向的 input_ids)
  └─ _finalize                                 # output_processor.process → generation_steps++ → ModelRunnerOutput

execute_launch(scheduler_output) → _PendingStep    # 异步前瞻:GPU 侧
execute_resolve(pending) → ModelRunnerOutput       # 异步前瞻:host 侧
```

异步前瞻的关键不变量(base.py `_PendingStep` docstring):**同一时刻至多一个 pending step**;launch 采样并发布下一步 token 轨道,record event;resolve 等 event、读 launch_buf、跑收集循环。`lookahead_eligible` 默认把带历史依赖采样参数(重复/频率/存在惩罚、min_new_tokens、自定义 logit processor)的批关在同步路径——**因为 launch 的采样比 resolve 的 append 早一步,读到的 output_ids 会滞后一个 token**。TTS 没有覆写它:层0 采样依赖 SGLang penalizer(历史敏感),理应被 gate;但注意 `generation_defaults` 里 `disable_overlap_schedule=True` 已在调度器层面禁用了 overlap,这条 gate 是第二重保险。

底座还内置两个通用 logit 处理(与 TTS runner 的覆写形成显式对照):

- `_apply_repetition_penalty`(base):host 侧增量维护 `unique(output_ids)` 集合(`_rep_penalty_unique_tokens` 的前缀折叠优化,注释点明"每步重建 unique 是二次方"),构造 (row, token, penalty) 索引张量打惩罚。**TTS 覆写为 no-op**(§4)。
- `_apply_codec_suppress_tokens`(base):按请求的 `suppress_tokens` 内容哈希缓存设备张量,按行组打 -inf。**TTS 覆写为固定区间版**(§4)。

## 2. TTS runner 的钩子填空总表

| 钩子 | Qwen3TTSModelRunner 的实现 | 一句话 |
|---|---|---|
| `before_prefill` | `prepare_decode_buffers` + `attach_omni_prefill_inputs(input_embeds=_build_prefill_input_embeds)` | 预填 embedding 注入 |
| `before_decode` | `prepare_decode_buffers` + `_write_feedback_buffers` | 行号寻址输入 + 采样参数暂存 |
| `post_prefill` / `post_decode` | `_collect_codes` | 层0码 + hidden → 预测器 |
| `sample_before_post_*` | True(两个阶段都提前采样) | 让 `_collect_codes` 拿得到 next_token_ids |
| `_sample_next_token_ids` | 先装 semantic seed 再 super() | §5 篇 |
| `_apply_repetition_penalty` | **del(空操作)** | 交给 SGLang penalizer |
| `_apply_codec_suppress_tokens` | 尾 1024 固定区间抑制 | 保留 codec_eos |
| `post_process_outputs` | 快照 → output_codes/反馈队列 | 每步的账本 |
| `_execution_context` | extend 时先恢复惩罚历史 | retraction 修复 |

## 3. Prefill 路径:embedding 切片回填

`_build_prefill_input_embeds`(`model_runner.py:296`):

```python
for sched_req in requests:
    req = data.req
    req_len = int(req.extend_range.length)          # 本段 extend 长度
    prefix_len = len(req.prefix_indices)            # radix cache 命中的前缀(可 >0!)
    if data.prefill_input_embeds is None:
        data.prefill_input_embeds = data.prompt_input_embeds   # 首次:全量 prompt
    piece = QwenTalkerModelRunner._projected_prefill_slice(
        sched_req=sched_req, prefix_len=prefix_len, extend_len=req_len, device=...)
    if piece is None or piece.shape[0] != req_len:
        raise RuntimeError(f"prefill embed mismatch ... have {have} rows, need {req_len}")
    pieces.append(piece)
return torch.cat(pieces).to(device, dtype)
```

这就是 02 篇"伪 token id"体系的另一半:**radix cache 命中前缀时,只有余下 extend 段需要前向,embedding 必须按 `[prefix_len : prefix_len+extend_len]` 精确切出**。`_projected_prefill_slice`(复用 omni talker runner)还处理 retraction 重放:`prefill_input_embeds` 不够长时,从 `_decode_input_history`(decode 输入的历史账本)补齐生成段 embedding。`attach_omni_prefill_inputs` 把张量挂到 forward_batch 的 sidecar,SGLang 前向读到 `input_embeds` 时走 embedding 注入而非查表(`req._input_embeds_are_projected=True` 告诉框架勿再投影)。

**对比 Omni thinker 的 prefill 注入**:`thinker_model_runner.py`(324 行)在 prefill 时识别图像/视频/音频占位 token 位置并替换为编码器 embedding——Omni 是"token 序列 + 局部占位替换",TTS 是"全序列 embedding"。前者保持 radix cache 逐 token 语义,后者用哈希伪 id 重建前缀语义。两个模型对"embedding prefill 与 KV cache 复用如何共存"给出了两个方向的答案。

## 4. Decode 路径:三个缓冲写手

### 4.1 `_write_feedback_buffers`(行号寻址的写入侧)

```python
decode_feedback_embedding = self.model._decode_feedback_embedding
input_ids = forward_batch.input_ids
# 断言:input_ids 至少 B 个;B 不超过嵌入行数
row_ids = self._decode_row_ids(batch_size, input_ids)       # arange 缓存(≥64 复用)
for row_idx, sched_req in enumerate(requests):
    combined = QwenTalkerModelRunner._take_next_decode_input_embed(sched_req=..., device, dtype)
    if combined is None:                                     # 无反馈且文本队列空(非正常)
        token_id = input_ids[row_idx:row_idx+1]
        combined = get_input_embeddings()(token_id)          # 兜底:codec 查表
    _append_decode_input_history(sched_req.data, combined)   # retraction 重放账本
    rows.append(combined)
torch.stack(rows, dim=0, out=decode_feedback_embedding.weight[:B])   # 一次栈写
input_ids[:batch_size].copy_(row_ids)                         # input_ids 变行号!
```

四个精度点:

1. **`torch.stack(out=...)` 直写权重行**——`_decode_feedback_embedding.weight` 就是写目标,无中间分配。
2. **`input_ids` 在 decode 期承载行号**(03 篇通道②),`copy_` 是 in-place——CUDA 图里 `embedding(input_ids)` 读到的每步内容都由这一行决定。
3. **`combined = 反馈.peek() + 文本.popleft()`** 的合成在 host 循环里逐行做(小张量 GPU 运算),失败兜底查 codec embedding(理论上不该发生,但留着比崩溃好——注意这行兜底 Omni 没有,Omni 直接 raise)。
4. **`_append_decode_input_history`**:每个用于前向的输入都记账,retraction 时 `_generated_prefill_slice` 用它重放生成段——TTS 的 decode 输入是 embedding,回收重算时没有 token 可重建 embedding,**账本就是重放机制本身**。

### 4.2 `prepare_decode_buffers` 的调用时序

TTS 在 `before_prefill` 与 `before_decode` **都**调用它(03 篇 §5 已拆内部)。prefill 时暂存采样参数的意义:prefill 采出的首 token(经 `_collect_codes` → 预测器)同样需要子说话人采样参数。Omni talker 的对应函数在 forward 内部还有 `invalidate_decode_buffers`(extend 后强制全量重暂存,因为 prefill 采样绕过了 `_sampled_token_ids` 缓冲),TTS 用 `(rid, epoch)` 幂等键达到同一效果。

### 4.3 `_collect_codes`:一步的产出

```python
layer0_codes = result.next_token_ids([B]) → unsqueeze
hidden = result.logits_output.hidden_states([B,1,H] 规整)
semantic_positions = self._sample_positions(forward_batch, device)   # decode: positions;prefill: seq_lens-1
self.model.code_predictor_forward(layer0_codes, hidden, semantic_positions=...)   # 04 篇
self._stage_token_ids(result, result.next_token_ids)    # pinned ping-pong 预暂存(见下)
self._has_pending_code_step = True
```

`_stage_token_ids`(base 提供):next_token_ids 异步拷进 pinned 双缓冲并 record event——**后续 `.tolist()` 等的是 event 而不是阻塞的页锁定拷贝**(base.py 注释:"decode 循环里的 c32 热点")。这是 runner 层能做的最后一段异步化。

## 5. `post_process_outputs`:快照、账本、流块

```python
eos_id = codec_eos_token_id
codes_snap = self.model._output_codes[:B].detach().clone()    # 一次性批克隆!
embeds_snap = self.model._output_embeds[:B].detach().clone()
for row_idx, sched_req in enumerate(scheduler_output.requests):
    req_output = outputs[sched_req.request_id]
    if req_output.data is None or int(req_output.data) == eos_id:  continue
    code_chunk = codes_snap[row_idx]
    sched_req.data.output_codes.append(code_chunk)             # 码本账本(终态汇总用)
    sched_req.data.latest_stream_code_chunk = code_chunk       # 流式出口(02 篇 §5.3)
    sched_req.data.pending_feedback_queue.append(embeds_snap[row_idx])  # 反馈账本
```

**为什么必须 clone?** 注释给出性能+正确性双重理由:"per-row clones 曾是 c32 解码环热点;行必须是快照的视图,绝不能是复用图缓冲区的视图"。`_output_codes/_output_embeds` 是**持久缓冲**(CUDA 图的固定地址),下一步前向原地覆写;若把视图直接 append 进 Python 账本,账本里所有"历史"都别名同一块显存——数据错乱。`[:B].detach().clone()` 一次分配搞定全批,clone 的所有权随后交给账本。eos 行跳过:codec_eos 是音频终点,该帧不计入码流。

**对比 Omni 同名函数**(`QwenTalkerModelRunner.post_decode`):Omni 的 post_decode 从 `model._sampled_token_ids`(图内采样缓冲)读回 next_token_ids,然后**直接往 outbox 发 stream 消息**给 code2wav,同样往 `pending_feedback_queue` 快照反馈。TTS 的差异是流消息不在这里发——`stream_output_builder` 由调度器在 output processor 阶段调用(职责分离:runner 管模型状态,适配器管消息),且首块拼 ref 码的协议在适配器里。**两份代码同一血脉,消息发送的层次不同,源于 TTS 的流协议更复杂(ref 前置 + num_quantizers 合同)。**

## 6. 重复惩罚的现状:与旧文档的对齐说明

旧文档 `qwen3-tts.md` §2.4 描述的 `_shape_masks`/`_mask_fingerprint` **在当前代码中不存在**。当前设计(以及为什么更优):

```python
def _apply_repetition_penalty(self, logits_output, requests):
    """Leave repetition-penalty ownership to SGLang's sampling state.
    ScheduleBatch.prepare_for_decode accumulates committed output tokens
    in SGLang's device-resident repetition penalizer. ModelRunner.sample
    applies that state once using the public SamplingParams value."""
    del logits_output, requests
```

SGLang 的 `BatchedRepetitionPenalizer` 在 `prepare_for_decode` 里**增量**累积已提交 token(base.py 的 `_rep_penalty_unique_tokens` 那套 host 端前缀折叠正是它的 Python 对照物),sample 时一次应用。TTS 的层0 采样走 SGLang sampler,惩罚状态机自然归它——runner 不再自建设备掩码。**代价是 retraction**:SGLang 的 `prepare_for_extend` 会为重新入队的请求建全新 sampling state,而 TTS 的 retraction 保留语义 `output_ids`(因为输入是 embedding,重放需要它们,03 篇 §4.6 账本)。于是:

```python
@staticmethod
def _restore_repetition_penalty_history(schedule_batch):
    """Seed a fresh SGLang penalizer before a request is re-prefilled. ..."""
    orchestrator = schedule_batch.sampling_info.penalizer_orchestrator
    penalizer = orchestrator.penalizers.get(BatchedRepetitionPenalizer)
    if penalizer is None or not penalizer.is_prepared(): return
    for row, req in enumerate(schedule_batch.reqs):
        retained_ids = {int(v) for v in req.output_ids if 0 <= v < vocab_size}
        rows/tokens/penalties 收集
    scaling_penalties[row_indices, token_indices] = penalty_values    # 批量直写
```

`_execution_context` 覆写:

```python
if schedule_batch.forward_mode.is_extend():
    self._restore_repetition_penalty_history(schedule_batch)
return super()._execution_context(schedule_batch, isolate_sampling=isolate_sampling)
```

——只在 extend(re-prefill)瞬间把保留 token 的惩罚一次性种回 penalizer 的 `scaling_penalties`,后续 decode 恢复正常每步累积。**这是"绕过框架状态机复用其能力"的完整样本**:不是重新实现 penalizer,而是在它的数据结构上做一次迁移。

抑制 token(`_apply_codec_suppress_tokens`)则仍然是 runner 职责——SGLang 没有对应的原生机制:

```python
configured_vocab = model.config.vocab_size
suppress_start = max(0, configured_vocab - 1024)      # 词表末 1024 个保留给 codec 控制
if suppress_start <= codec_eos < suppress_stop:
    logits[:, suppress_start:codec_eos] = -inf
    logits[:, codec_eos+1:suppress_stop] = -inf        # 只放行 codec_eos
else:
    logits[:, suppress_start:suppress_stop] = -inf
```

与 base 版(任意 token 集 + 内容哈希缓存)不同,TTS 是**固定区间**——无需缓存键,直接切片写。区间语义:词表尾部 1024 个 id 是 codec 控制符空间,生成期除 EOS 外全部禁出。

## 7. 每步生命周期的完整时序(收拢)

```
scheduler_output(step N)
  │ before_decode: prepare_decode_buffers(幂等)→ _write_feedback_buffers(行号+反馈+文本)
  │ forward(Qwen3TTSTalker,图或 eager)→ codec_head → logits
  │ _sample_next_token_ids:装 semantic seed → SGLang sample(penalizer 已在 prepare_for_decode 累积)
  │ _collect_codes:层0码+hidden → 预测器(04 篇)→ _output_codes/_output_embeds
  │ _stage_token_ids:pinned 预暂存
  │ _publish_next_tokens:下一步前向的 token 轨道(execution bridge)
  │ _finalize:output_processor.process(等 event 读 host ids)→ post_process_outputs:
  │      快照 clone → output_codes.append / latest_stream_code_chunk / pending_feedback_queue
  │      generation_steps++ → 引擎层判定 eos/length → result 或继续
  ▼
stream_output_builder → vocoder(07 篇);终态 result_adapter → 汇总 codes
```

## 8. 与 Omni talker runner 的差异小结(呼应 03/05 篇)

| | Qwen3TTSModelRunner | QwenTalkerModelRunner(Omni) |
|---|---|---|
| decode 就绪门 | 无(输入总能合成,兜底查表) | `is_decode_batch_ready` + `before_decode` 硬 raise(必须等 thinker 喂文本) |
| post_decode 的 next_token_ids | SGLang sampler 产出 | `model._sampled_token_ids` 图内采样回读 |
| 流消息 | 不发(适配器发) | runner 直发 outbox(含 `metadata={"stream": is_streaming}`) |
| 重复惩罚 | SGLang penalizer + retraction 恢复 | 模型内掩码暂存(`prepare_decode_buffers` 的 rep_rows 增量置位) |
| prefill 清理 | 同 omni:不清 `prefill_input_embeds`(注释:retract 会重新入队) | 同左 |
| 复用关系 | 复用 QwenTalkerModelRunner 的 4 个静态方法 | 被复用者 |

下一篇进入数据平面的最后一站:流式声码器与增量编解码——`streaming_vocoder.py` 1722 行的并发与显存安全设计。

# Qwen3-TTS 深度解析(七):流式声码器与增量编解码

> 本篇覆盖 `streaming_vocoder.py`(1722 行)与 `incremental_codec.py`(471 行)。这是全模块对并发正确性与 CUDA 上下文安全要求最苛刻的文件:两个异步 worker、优先级流调度、pinned 传输槽、双层 CUDA 图、以及对"同步失败即进程级故障"的极端处理。

---

## 1. 位置与骨架:StreamingVocoderBase 的模板方法

`Qwen3TTSStreamingVocoderScheduler(StreamingVocoderBase[_Qwen3TTSStreamState, None])`。基座(`scheduling/streaming_vocoder.py`)的契约(其模块 docstring 是权威文档):

- 请求级状态注册表:首个 chunk/payload 时 `create_stream_state`,终态清理;
- 流合同锁定:`latch_stream_contract` 在首个流块上锁死 `num_quantizers`/`ref_code_len`/`initial_codec_chunk_frames`;
- 阈值循环:`on_stream_chunk → ingest → should_decode → decode_delta → emit`;
- `on_stream_done` 序列:冲剩余 → 末块 → 终态 result → 清态;
- 无输出时 `fallback_full_decode` 整段兜底;
- abort/stop → `release_stream_resources` → 弹态;迟到的块对已清理请求直接丢弃。

模型侧要实现的钩子:`create_stream_state / latch_stream_contract / validate_chunk / ingest / should_decode / decode_delta / final_result_data / fallback_full_decode`。

时间参数(构造默认,全部可由工厂 kwargs 覆盖):

| 参数 | 默认 | 语义 |
|---|---|---|
| `stream_stride` | 16 帧 | 稳态步长(16 帧 @12Hz ≈ 1.33s 音频) |
| `stream_followup_stride` | 8 帧 | 稳态**触发阈值**增量 |
| `initial_chunk_frames` | 8 帧 | 首块解码长度(TTFA 关键) |
| `stream_left_context_frames` | 16 帧 | 解码窗口的左侧上下文 |
| `initial_max_batch_size / wait` | 32 / 2ms | 首 worker 攒批 |
| `followup_max_batch_size / wait` | 8 / 1ms | 后续 worker 攒批 |

`stream_chunk_ramp` 是两旋钮的泛化替代(互斥,配置校验拒绝混用):`ramp[0]` 为首块,其余为逐次步长;且校验 `ramp[0] <= stream_stride`,注释:"请求期解析器会把首块 clamp 到稳态步长,更大的配置值会静默跑出另一种不可用图形状的调度"。

## 2. 流状态与合同

`_Qwen3TTSStreamState` 全字段即流式会话的账本:

```python
code_chunks: list[Tensor]      # 未解码码块(可裁剪)
total_frames / ref_frames / pruned_frames / emitted_generated_frames   # 四个游标
next_decode_generated_frames   # 下次触发阈值
decoded_chunks                 # 已解码次数(ramp/legacy 分支判据)
num_quantizers                 # 合同:Q
pending_ref_frames             # 首块携带的参考帧数(消费一次)
initial_chunk_frames           # 合同:首块长
initial/followup/final_pending # 三态在途标记(防重复入队)
playback_deadline_s            # 播放截止钟(优先级队列键)
incremental_codec_state        # 增量解码器状态(§6)
incremental_codec_fallback     # 增量失败后降级标记
```

`latch_stream_contract`(`streaming_vocoder.py:313`):

- payload 来源:从 `request.params` 解析 `initial_codec_chunk_frames`;
- metadata 来源(流块):`num_quantizers` 必须存在且**与会话值一致否则 raise**(合同不可变);`ref_code_len` 只允许出现在首个块且不得迟于流开始(`"reference codes arrived after stream start"`);`initial_codec_chunk_frames` 可在流上更新(经 `resolve_initial_codec_chunk_frames` clamp 到稳态步长)。

`validate_chunk`:`[Q]` 或 `[T, Q]` 归一为 [T, Q];Q 与合同比对;**CPU 张量**才做范围检查(`[0, 2048)`)。为什么跳过 CUDA 张量?§4 的答案(设备端断言会毒化上下文,改在解码前统一 clamp+筛查)。

## 3. 解码计划与窗口数学

核心不变量:码流 = `[参考段 ref_frames | 生成段]`,绝对帧号;已发出 `emitted_generated_frames`;窗口取 **`[max(0, 已发绝对帧 - left_context), ref + generated)`**:

```python
def _build_decode_plan(state, *, is_final, max_generated_frames=None):
    available = state.total_frames - state.ref_frames
    if available <= state.emitted: return None                       # 无新帧
    next_frames = self._next_decode_threshold(state)
    if not is_final and available < next_frames:
        state.next_decode_generated_frames = next_frames; return None  # 未达阈值
    generated = min(available, max_generated_frames or inf)
    absolute_emitted = state.ref_frames + state.emitted
    window_start = max(0, absolute_emitted - self._stream_left_context_frames)
    window_end   = state.ref_frames + generated
    while code_chunks and pruned + len(chunk0) <= window_start:      # 整块裁剪
        state.pruned_frames += len(state.code_chunks.pop(0))
    codes = torch.cat(state.code_chunks, dim=0)
    decoder_input = codes[window_start - pruned : window_end - pruned]
                    .transpose(0, 1).unsqueeze(0)                    # → [1, Q, T]
    return _Qwen3TTSDecodePlan(...)
```

- **left context 的必要性**:codec 解码器的因果卷积/transformer 有感受野;只解码新帧会用错历史。经典流式方案 = 携带窗口重解码 + 裁掉重叠输出(`_extract_delta` 按 `absolute_emitted - window_start` 帧数 trim 掉头部)。
- **整块裁剪**的注释:"window_start 只前进,后面的帧是死帧;整块 pop 保持 cat O(窗口) 而非每解码 O(流长)"——`pruned_frames` 是绝对坐标到裁剪坐标的换算偏移。
- `should_decode`:非 final 时 `generated_frames >= 初始阈值或 next_decode_generated_frames`。

`_commit_decode_plan`:提交时校验顺序(`emitted == plan.emitted`,乱序即 RuntimeError)、更新 `emitted_generated_frames`、`decoded_chunks++`、计算下次阈值(`plan.generated + _next_followup_stride`)、并推进**播放截止钟**:

```python
duration_s = len(delta) / sample_rate
state.playback_deadline_s = max(state.playback_deadline_s, now) + duration_s
```

**播放截止钟 = 该请求音频播完的时刻**。它是后续 followup 优先级队列的键:欠账越多(播放截止越近),解码越优先——这是把"实时播放"硬实时约束映射进调度优先级的一手设计。

`_next_followup_stride` 两种模式:legacy(首次 8、其后 16)或 ramp(按 emitted 帧数在 ramp 累积表上定位,注释:"backlog 冲过头会回到稳态步长")。

## 4. 异步解码:双 worker + 优先级流 + pinned 传输

### 4.1 两级 worker

`on_serving_start` 启动两个守护线程:

- **initial worker**(`_initial_queue`, FIFO):每个请求的**第一次**解码。`_run_initial_batch` 在 `_state_lock` 内构建计划(且限 `max_generated_frames=initial_chunk_frames`,防 backlog 冲爆首块),按**解码器输入形状分组**(`_group_decode_plans`),组内拼批解码。提交后 commit(`_commit_initial`):发流块、视剩余决定转 followup 或收尾。
- **followup worker**(`_followup_queue`, **PriorityQueue 键=(playback_deadline_s, seq)**):后续解码,优先级=播放截止钟(同键按入队序稳定)。

两级分离的动机:**首解码决定 TTFA,必须最优先且批容忍度大**(32/2ms);后续解码按截止钟公平排序(8/1ms)。`_schedule_initial/_schedule_followup` 的 `pending` 标记防重复入队;`_collect_async_batch` 是标准的攒批循环(阻塞取首 + 截止窗内凑批)。

失败隔离:`_decode_group` 捕获 `_Qwen3TTSInvalidCodeRows`,**只淘汰坏行所在请求**(`bad` 集合过滤后重试),其余请求不受牵连;其它异常则整组失败。`_commit_*` 里 commit 失败 → `_emit_error` + abort + 清理。

### 4.2 解码流与 CUDA 图

两条专用低优先级 CUDA stream(`priority_range()` 的最次优先级+1):`_decode_stream`(initial)与 `_followup_decode_stream`。**低优先级是刻意设计**:声码器与 AR 引擎共卡时,codec 解码要让路给骨干前向——流优先级是 GPU 硬件级的抢占顺序。

双层 CUDA 图(`_Qwen3TTSInitialDecodeGraphs`,同一个类服务两处):

```python
initial 图:input_frames = left_context + initial_chunk(= 16+8 = 24)
followup 图:input_frames ∈ {left+ramp 各步, left+steady}(= 24/24/…/16+16)
batch 桶:(1,2,4,8);确定性模式只 (1,)
```

`capture()`:`(frames, batch)` 全组合,边流 warmup×2 → capture stream 捕获,**共享 graph_pool**;失败仅跳过该组合(warning)。

`decode(codes)`:

```python
codes [B, Q, T]:T ∈ 图输入帧集合 且 Q 匹配 → 找 ≥B 的桶
static_input.zero_(); [:B].copy_(codes); replay; return outputs[:B].clone()
```

窗口形状不在图集合时(短参考/早期流被截断,见 §1 注释)→ 图返回 None → eager `chunked_decode`。

### 4.3 pinned 传输槽:`_DecodeSlot` 与生命周期状态机

每线程一个槽(`threading.local`,`_thread_decode_slot`),由 `GrowablePinnedBuffer`(输入码)+ `PinnedTransferSlot`(输出音频,自带复用 Event)构成。数据流(类 docstring 原文):

```
CPU codes [B,Q,T] → input_codes(pinned) → CUDA decoder
CUDA 音频增量 [S] → output_transfer(pinned + event) → 独立 CPU 张量(非 pinned)
```

`busy`(在途)/`broken`(粘性,永不再用)两态。`_launch_async` 的关键序列:

```python
stream.wait_stream(current_stream)          # 与当前流建立依赖
with torch.cuda.stream(stream):
    gpu_input = stage_decoder_input(...)     # H2D(pinned 非阻塞)
    waveform = graphs.decode(gpu_input) or chunked_decode(gpu_input)
    deltas = [extract_delta(...)]
    staged = stage_deltas(deltas, slot)      # D2H 写 pinned 视图
    slot.output_transfer.record(stream)      # 事件即使全空也记录——注释:"事件同时 fence H2D 与解码"
return _Qwen3TTSDecodeHandle(staged, bad_rows, slot=slot, ...)
```

**零长度 `.cpu()` 陷阱**(非 pinned fallback 分支注释):"零元素 `.cpu()` 不会入队任何 D2H,也就不会等待前面的解码——所以必须显式 `stream.synchronize()`"。这类细节是异步正确性的常见翻车点。

`resolve()`(handle 的终态方法):

1. 等待事件(event 失败 → 退而 `stream.synchronize()`);
2. 事件失败但流成功:丢视图、槽标记 broken(事件永不再信)、释放、**仍然 raise**(本批音频作废,但资源回收);
3. **事件与流都失败:进程级故障**。`_CONTEXT_FATAL_RETAINED` 全局列表**永久持有**该解码触碰的一切(槽、输入、输出、keepalives),`owner._cuda_decode_failed=True` 永久禁用 CUDA 解码。注释逻辑链:"事件和流都无法同步 ⇒ 无法证明 GPU 已用完这些缓冲 ⇒ 分配器可能把它们交给后续工作 ⇒ 唯一安全做法是让它们活到进程结束"。这是一条**宁可停服也不产出未定义音频**的底线设计。
4. 成功:`[delta.clone() for delta in deltas]`——把 pinned 视图物化为独立 CPU 张量再还槽(clone 的必要性同 06 篇快照)。
5. `resolve` 幂等:重复调用返回缓存结果或重抛同错,不再触碰 event/槽。

`_screen_out_of_range_codes`(`streaming_vocoder.py:628`)值得整段理解:

```python
bad_rows = ((decoder_input < 0) | (decoder_input >= 2048)).flatten(1).any(1)
decoder_input.clamp_(0, 2047)
```

注释:越界 id 会让 embedding lookup 触发**设备端断言,毒化 CUDA 上下文并杀死本进程所有在途流**;`validate_chunk` 拦不住设备张量,故统一 clamp 保 lookup 安全,同时返回 per-row 判定——CPU/确定性路径在解码前 raise,异步 CUDA 路径在 `resolve()`(事件后)读取判定再对坏行 raise,**不为筛查添加任何同步点**。被 clamp 的行按失败处理,永不发声。

### 4.4 出口整形

`_decode_and_emit`(同步路径)有一个补偿逻辑:首解码若冲掉积压(如客户端慢,阈值前攒了很多帧),**把 delta 按 ramp 切成多块发出**,保持客户端可见块形的 ramp 形状——`split_frames = (initial,) + ramp[...]`。注释:"without a ramp only the legacy initial-boundary split applies"。这是流式 API 的"块形稳定性"细节:音频总量不变,但块序列的形状影响客户端的播放平滑度。

`_store_vocoder_result`(非流式终态)按 `ref_code_len / total_frames` 的**帧数比例**裁掉波形头部参考段——比例裁剪而非按 samples_per_frame 精确裁,因为整段解码器的上采样不一定整除,注释以 `int(ref/total × len)` 实现,音质无感但边界对齐到可用长度。

## 5. 确定性模式在声码器的落点

`enable_deterministic_inference=True` 时(02 篇 §1.3):

- `_launch_decode_plans` 对多计划**逐个 B=1 重解**:注释引用 issue #1475——批内其它请求影响 kernel 的归约顺序,输出不确定;逐个解则输出与批组成无关。校验合并批(坏行索引仍指原组)后循环单发。
- CUDA 图 initial/followup 桶退化为 `(1,)`。
- `_vocode_payloads` 的非流式整段解码也走逐请求循环。

## 6. `incremental_codec.py`:有状态流式解码器(免窗口重解)

窗口重解(left-context 方案)每步重复计算重叠帧;`enable_stateful_codec_decoder=True` 时改走**真增量解码**:解码器内部状态(卷积历史 + transformer KV)跨步保持,每步只算新帧。

### 6.1 状态结构

```python
@dataclass
class Qwen3TTSIncrementalCodecState:
    frame_position: int                    # 已消费绝对帧
    transformer_context_length: int        # pre_transformer KV 有效长
    transformer_keys/values: dict[layer→Tensor]
    conv_histories: dict[key→Tensor]       # 每个因果 Conv1d 的左历史
    transconv_overlaps: dict[key→Tensor]   # 每个转置卷积的右重叠
    def clone(self): ...                   # 深拷贝所有张量
```

### 6.2 三类原语

**增量因果卷积**(`incremental_causal_conv1d`):`cat(历史, 新帧)` → 正常 conv → 输出新帧长 → **历史 = 合并尾 padding 长度的 clone**。要求 stride=1(转置卷积另算)。

**增量转置卷积**(`incremental_causal_transconv1d`):上采样块的核心难题。`conv_transpose1d(新帧)` 的输出含"上一帧的右重叠"区:`overlap` 存在则 `output[..., :overlap_len] += overlap`(跨块求和,因为转置卷积的重叠贡献是加性的);输出分为 `emitted(新帧数×stride)` 与 `tail(== right_pad,校验)`;`tail` 存为下步 overlap;bias 单独加到 emitted。**数学**:标准转置卷积对分块输入不是线性的(邻块贡献重叠区),手动维护重叠和才能与整段解码逐位一致。

**增量 transformer**(`_incremental_transformer`):标准 KV 缓存 + 滑窗:

```python
query_pos = arange(frame_position, +fresh)
key_pos   = arange(frame_position - ctx_len, +fresh)
allowed   = key_pos <= query_pos;  滑窗: & key_pos > query_pos - sliding_window
retained  = max(0, window_size - 1)
next_keys[layer] = key[..., -retained:, :].clone()      # 逐层裁剪保留
state.transformer_context_length = min(retained, ctx_len + fresh)
```

注意 attention 内显式构造 mask(而非 SDPA 因果标志),因为 key 序列起点是全局坐标偏移过的。

### 6.3 结构守护:`_require_attrs`

构造时对解码器逐模块断言字段存在(pre_conv/pre_transformer/upsample 每块/decoder 每块……),**布局不符立刻 TypeError**——一个防 qwen-tts 升级悄悄改模块结构、增量路径静默走错的前置闸。`decode()` 末尾校验 `state.frame_position == 期望终点`,任何漂移即 RuntimeError(上游 `decode_delta` 捕获后 `incremental_codec_fallback=True`,**该请求其余部分永久降级到窗口方案**,注释 "using the legacy left-context decoder for the rest of the request")。

`decode_delta` 的增量分支还有两道一致性断言:`consumed_frames == ref + emitted`(位置账本对齐)与 `consumed >= pruned`(裁剪不超前)。增量路径输出按 `reference_frames × samples_per_frame` 精确裁参考段——比窗口路径的整段解码裁剪更精确(7bit 对齐)。

## 7. 与 Omni `Code2WavScheduler` 的对照(同为 StreamingVocoderBase 实现)

| 维度 | Qwen3-TTS 流式声码器 | Omni Code2Wav |
|---|---|---|
| 状态 | `_Qwen3TTSStreamState`(游标 + 增量 codec 态) | `Code2WavStreamState`(游标 + unchecked 扫描) |
| 触发模型 | `should_decode` 阈值(初始 8/稳态 16/阈值增量 8/ramp) | `_ready` 扫描 + `_step_frames` + serial threshold(上下文按 chunk 增长至 cap) |
| 图键 | `(frames 固定集合, batch 1/2/4/8)` ×initial/followup 两层 | `GraphKey(batch, frames)`,帧序列=按 chunk 递增至 `chunk+left_context`;批组合 `_DECOMPOSE_SIZES=(8,4,2,1)` **分解**(B=6→4+2) |
| 批组策略 | 双 worker 两种批参数 + playback deadline 优先级 | 单线程 inbox 泵 + `_batch_deadline` + 形状分组分解 |
| D2H | 每线程 pinned 槽 + 双层故障保留 | `_PinnedSlot` 池(acquire/release/quarantine/reap 四态,事件复用) |
| 输出重叠 | pinned slot + record/resolver | 输出重叠开(1024 行处 `output_overlap`)+ chunk-aligned dispatch |
| 增量解码器 | 独立实现(卷积历史+转置重叠+滑窗 KV) | 无(HF Code2Wav 窗口解码) |
| 特殊保障 | clamp+bad_rows 筛查;`_CONTEXT_FATAL_RETAINED`;全批禁用 | slot 隔离区与回收;`gpu_memory_fraction=0.02` 显式预算(图缓冲预算来自 `total_gpu_memory_fraction`) |

两者共有的深层模式:**"固定形状图 + 死行清零 + pinned 暂存 + 事件物化 + 槽故障粘性"**——同一条流式 GPU 服务蓝图在两个模型上的两份实现,差异只在触发策略(TTS 的实时播放优先级 vs Omni 的 chunk 对齐)与解码器内部状态(TTS 有增量 codec,Omni 没有)。

---

## 8. 本篇要点回顾

1. 窗口解码 = left-context 重解 + trim;增量解码 = 卷积历史 + 转置重叠 + 滑窗 KV,两者互为降级。
2. 双 worker(首块 TTFA / 后续截止钟优先级)把实时播放约束编入调度。
3. pinned 槽 + 事件是 D2H 异步化的最小完整方案;`resolve()` 的故障矩阵(事件失败→流同步→进程级保留)定义了"什么时候可以继续服务"的边界。
4. clamp-then-check 模式:先保 CUDA 上下文活着,再按行问责。
5. 确定性模式的全链路代价:预处理串行 + AR 确定性 + 声码器 B=1 逐个解。

最后一篇把两个模型放在一起做总账式对比。

# Qwen3-TTS 深度解析(八):与 Qwen3-Omni 的深度对比

> 本篇是系列收官。前 7 篇把 Qwen3-TTS 拆到了实现层;本篇把两个模型**放回同一个框架坐标系**里逐层对照——拓扑、数据流、模型结构、采样、执行、声码器、部署——并提炼可迁移的设计结论。
> 对照基准:`qwen3-omni.md` 的描述 + `sglang_omni/models/qwen3_omni/` 13,468 行现状代码 vs `sglang_omni/models/qwen3_tts/` 7,180 行。

---

## 1. 设计定位:一句话的分歧

**Qwen3-Omni = "听得懂、看得见、还会说"的多模态模型。** 架构中心是 thinker(30B MoE 文本/多模态骨干);talker 是它的语音外设,**持续消费 thinker 逐 token 流出的 hidden states** 来生成语音。语音是 thinker 输出的第二种消费方式(第一种是文本 detokenize)。

**Qwen3-TTS = 纯 TTS。** 没有上游语义模型;AR 骨干(talker 结构)自己就是生成主体,条件信息(参考音频 x-vector/ICL、预设音色、设计指令)在**预处理期一次性烧进 prompt embedding**。它是 Omni talker 的"截断 + 改造"后代。

这个定位差决定后面所有差异。用一个比喻:Omni 是"边想边说"(thinker 想一个 token,talker 跟着说一点);TTS 是"想好了再说"(条件先固化,然后自顾自说)。

## 2. 流水线拓扑:线性链 vs DAG

```
Qwen3-TTS(3 阶段,单进程):                    Qwen3-Omni speech(7 阶段,DAG):

preprocessing ──▶ tts_engine ──▶ vocoder        preprocessing ──┬─▶ image_encoder ──┐
 (线程池)        (EngineStage)    (terminal)                     ├─▶ audio_encoder ──┤
      ▲ process_local_edges                              └─▶ (直接) ────────┤
      △ 模块级 _PREPARED_REQUESTS 交接                                     ▼
                                                  [wait_for: pre+img+aud]
                                                  thinker ──stream──┬─▶ decode (文本)
                                                    │  stream        └─▶ talker_ar ──stream──▶ code2wav
                                                    └─▶ talker_ar(部分启动)
```

| 维度 | Qwen3-TTS | Qwen3-Omni |
|---|---|---|
| stage 数 | 3 | 6(text)/ 7(speech) |
| 图形状 | 线性链,`next` 单后继 | fan-out(preprocessing→3 路)、fan-in(thinker/talker `wait_for` 3 源 + `merge_fn`) |
| 路由 | 无 | `route_fn` 运行期决定下一站(`resolve_preprocessing_next_stages[_speech]`,纯文本请求可跳过 encoder) |
| payload 投影 | 无(进程内,只有状态字典) | `project_payload` 函数族(`project_thinker_to_decode` 剥多模态张量、`project_talker_to_code2wav` 换轻量闩锁)减 IPC |
| 多模态流接口 | `stream_to=["vocoder"]`(code 帧) | thinker `stream_to=["talker_ar","decode"]` **双流**;talker `stream_to=["code2wav"]` |
| 部署形态 | 全部 `process="pipeline"`、`gpu=0`,单进程 | 分散式(thinker GPU0 / talker GPU1)或 colocated(5 个 GPU stage 同卡,CPU 环境默认 `OMP_NUM_THREADS=8` 防超订) |
| 进程间数据 | 模块级注册表(`_PREPARED_REQUESTS`)+ `process_local_edges` 显式声明 | 全显式 StagePayload/CUDA IPC(`disable_direct_cuda_ipc_payload` 按需关) |

**结论 1:TTS 的"脏全局"是线性拓扑的特权。** 单进程 + 单后继,让 `_PREPARED_REQUESTS` 这种注册表交接成为最低摩擦方案;Omni 的 DAG 有并行分支与跨进程边,任何全局状态都会引入竞态,必须走消息。**拓扑复杂度决定状态管理风格,而不是反过来。**

**结论 2:两边的"流"语义不同物。** TTS 的 stream 是**产物分片**(生成的 code 按块交付声码器);Omni thinker 的 stream 是**上游条件流**(talker 的输入依赖 thinker 的每一个 token)。后者的耦合度一级别:talker_ar 的解码就绪门(`is_decode_batch_ready` / `before_decode` 硬 raise)就是为了在数据未到时**挡住调度**,而 TTS 的 runner 永远能自己合成输入(兜底查表)。

## 3. 条件注入:静态 prompt vs 动态 hidden 流

这是两个模型最本质的架构分歧,值得单独一张表:

| | Qwen3-TTS | Qwen3-Omni |
|---|---|---|
| 条件何时确定 | 预处理期(请求到达时) | 生成期逐 token(thinker 边跑边定) |
| 注入介质 | prompt embedding 拼接(x-vector 位、ICL 段、instruct 前缀) | **两种**:① prefill 期 thinker prompt 的投影重建(`TalkerPrefillBuilder.build_prompt_prefill`:从 safetensors 直接读 thinker embedding 行、重放 prompt、拼 assistant chunk);② decode 期每 token 的 hidden state 流(`stream_output_builder` 分双层的 embed/layer-N hidden) |
| 文本通道 | `PendingTextTensorQueue` 预装满(`trailing_text_hidden`);`_finish_text_prompt` 切首 token 进 prefill | 同一队列类,但 `append_text_chunk` 随 thinker 流式 append;`mark_thinker_done` 后补 `tts_eos` |
| 每步输入 | 反馈(ΣQ 嵌入)+ 队列文本 | 反馈 + thinker hidden 投影(文本位 text_projection,**多模态位 hidden_projection**——deepstack 双投影) |
| 首音优化 | 小首块(8 帧)+ 声码器两级 worker | 部分启动:`MIN_PARTIAL_START_CHUNKS=3`,`partial_start_min_chunks=5`——thinker 攒够最少 chunk 就放 talker 起跑,不必等 prefill 全完 |

**结论 3:同一个 `PendingTextTensorQueue`,在 TTS 里是"字幕机"(预排好,逐帧滚),在 Omni 里是"同传耳机"(上游说一句进一句)。** 数据结构相同(`pending_text_queue.py`,设备驻留 chunked FIFO,O(1) peek),生命周期语义完全不同——好的基础设施组件常常如此:接口为两种用法收敛,时序语义留给上层。

**结论 4:Omni talker 的 prefill 是"重放",TTS 的 prefill 是"拼装"。** Omni 必须把 thinker 的完整 prompt(含多模态占位)重建为 talker 视角序列(`max_seq_len` 因此从 8192 提到 32768——30 帧视频 prompt 约 22K 位置,注释记录过一次 `FusedAddRMSNorm` 非法内存访问事故);TTS 的 prompt 是自包含的小序列(角色头 + codec 前导 + 说话人位 + ICL 段 ≤ 文本长)。这是"有无上游"对 prefill 成本的影响。

## 4. Talker 模型结构:同一骨架的两次实现

直接血缘证据(均为 `qwen3_tts/sglang_model.py` 顶部 import):

```python
from sglang_omni.models.qwen3_omni.components.talker import (
    Qwen3OmniMoeTalkerDenseMLP, ResizeMLP, _bind_default_weight_loaders)
from sglang_omni.models.qwen3_omni.components.thinker_model import Qwen3OmniMoeThinkerTextAttention
from sglang_omni.models.qwen3_omni.pending_text_queue import PendingTextTensorQueue   # (request_builders)
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner   # (model_runner)
```

| 组件 | Qwen3-TTS | Qwen3-Omni |
|---|---|---|
| 注意力 | `Qwen3OmniMoeThinkerTextAttention`(共享) | 同左 |
| FFN | `Qwen3OmniMoeTalkerDenseMLP` | MoE(`SparseMoeBlock` + shared expert MLP) |
| 文本→talker 投影 | 单 `text_projection`(ResizeMLP) | `text_projection` + `hidden_projection` 双投影 + `prepare_input_embeds` 按 mask 选路(deepstack 层 N 状态) |
| decode 输入 | `_decode_feedback_embedding` 行号寻址(共享设计) | 同构 |
| 反馈缓冲 | `_feedback_buffer/_mask`(共享设计) | 同构 |
| 持久输出 | `_output_codes [B,Q] / _output_embeds [B,H]` | 同构(维度可能不同) |
| logits | `codec_head` 线性 | 同 |
| 采样发生处 | SGLang sampler(runner 层) | **模型 forward 内** `_sample_decode_tokens` |
| 说话人条件 | x-vector/spk 表/instruct | 固定 speaker map |
| speaker encoder | 内嵌(`Qwen3TTSSpeakerEncoder`,仅 base 检查点) | 无(不需要) |

**结论 5:`_decode_feedback_embedding` 行号寻址是两个 talker 共同的图安全基石。** 把"每步输入 embedding"变成"预写缓冲 + 行号索引",CUDA 图才能以固定输入形状重放——两个模型独立演化出同一方案(或同源复制),说明这是 embedding 输入模型接 CUDA 图的**标准答案**。TTS runner 里复用 omni 的静态方法(`_take_next_decode_input_embed` 等)则把"重放账本 + 反馈/文本合成"这段逻辑彻底归一。

**结论 6:采样位置相反(呼应 05 篇)。** Omni 把采样搬进 forward 是为了图的完整性;TTS 把层0采样留在 SGLang sampler(要 per-request seed + 原生 penalizer 生态),预测器采样单独图化。判断依据都是"哪部分必须进图",但两模型的图边界画在了不同地方——**CUDA 图边界是模型语义的函数,不是框架的函数。**

## 5. 预测器与 CUDA 图:复杂度差 = 采样语义差

(细节见 04 篇 §5,这里给结论级对照)

| | Qwen3-TTS | Qwen3-Omni |
|---|---|---|
| 图键 | `(bucket, argmax/sampled, max_top_k 梯级, has_top_p, has_unbounded)` | `(bucket, dtype)` |
| 图内采样 | seeded Gumbel(Triton fused)+ `torch.where(do_sample)` | 纯 argmax |
| 捕获防护 | 容量 32 键 / 失败 8 次全局禁用 / env 开关 / TP 拒绝 / 签名钉状态 / 共享池 | 键级禁用 |
| 预测器 KV | 自管稠密 cache(与 omni 同构) | 同构 |
| fused kernel | gather+add、raw-logit 采样、H100 addmm(共 3 处) | 无 |

**结论 7:图键的维度 = 图内控制流的自由度。** TTS 的五元键几乎每一维都来自"per-request 采样参数必须进图";Omni 只需 dtype。若给 Omni 也加 seeded 采样,它的键会立刻长出同样的维度——这是评估"给现有图化模块加功能"成本的通用推演法。

## 6. Runner 层:门卫的松紧

| | Qwen3TTSModelRunner | QwenTalkerModelRunner(Omni) |
|---|---|---|
| decode 就绪门 | 无(自给自足) | `is_decode_batch_ready` / `before_decode` raise(必须等 thinker 文本 + 反馈) |
| prefill 输入 | prompt embedding 切片(radix 命中后按 extend 段回填) | thinker hidden 投影重建(无 radix 复用价值) |
| 惩罚 | SGLang `BatchedRepetitionPenalizer` + **retraction 恢复历史**(`_restore_repetition_penalty_history`) | 模型内掩码暂存(`prepare_decode_buffers` 增量置位) |
| 抑制 | 固定区间(尾 1024 保留 codec_eos) | per-request `suppress_tokens` 缓存张量 |
| 流消息 | 适配器层(`stream_output_builder`,ref 前置协议) | runner 直发 outbox |
| lookahead | 默认 gate(惩罚历史敏感);实际被 `disable_overlap_schedule` 双保险 | thinker 侧 `enable_async_decode=True` 真正使用 |

**结论 8:门卫的松紧 = 上游确定性的函数。** TTS 的每步输入完全本地可推(队列 + 反馈),不需要门;Omni 的 talker 输入依赖跨 stage 流,必须有门 + FIFO 交接。`is_decode_batch_ready` 也接进了调度器(批不齐不解码),这是流式上游模型在调度层的额外耦合面——TTS 完全没有。

**结论 9:retraction 是 embedding-prefill 模型的专属税。** SGLang 回收假设"input_ids 能重建一切";TTS/Omni talker 的输入是 embedding,必须自账(`_decode_input_history` + `_generated_prefill_slice` 重放 / TTS 保留 output_ids + 惩罚历史种回)。任何"输入非 token"的模型接 SGLang 都要交这份税,两个模型交税的方式可以互相抄。

## 7. 声码器:同一基座,两种实时策略

(细节见 07 篇 §7。结论级:)

- **共同蓝图**:StreamingVocoderBase 模板方法 + 固定形状 CUDA 图 + 死行清零 + pinned 暂存 + 事件物化 + 槽粘性故障。
- **分叉点 1——调度目标**:TTS 用 **playback deadline**(实时播放的硬约束,欠账优先);Omni 用 **chunk 对齐 + 分解批**(吞吐与延迟折中,批形状可分解为 8/4/2/1 组合)。
- **分叉点 2——解码器状态**:TTS 有 `incremental_codec`(卷积历史/转置重叠/滑窗 KV)做真增量;Omni 的 Code2Wav 是无状态窗口解码(依赖图键的"上下文阶梯"渐增序列)。增量解码器的正确性负担(三套状态 + 逐位一致性)换的是每个窗口少算 left-context 的重叠帧。
- **分叉点 3——资源预算**:Omni colocated 显式给 code2wav `gpu_memory_fraction=0.02` 并为图缓冲要求明确预算;TTS 依赖全局 `mem_fraction_static=0.85` 与低优先级流。

**结论 10:TTS 的声码器复杂度(1722 行)显著高于 Omni 同层(1065 行),多出的部分几乎全部是"实时性 + 故障边界"代码**——播放截止钟、两级 worker、`_CONTEXT_FATAL_RETAINED`、clamp-then-check。纯 TTS 产品对首块延迟和连续播放的要求比 Omni 的语音回复更苛刻(后者还有 thinker 延迟在前面垫底)。

## 8. 配置、引擎与部署差异

| | Qwen3-TTS | Qwen3-Omni |
|---|---|---|
| 配置类 | `Qwen3TTSPipelineConfig`(单类) | Base + Speech + **SpeechColocated** 三类 + Variants 字典(`text/speech/speech-colocated`) |
| placement | 无 | `Qwen3OmniPlacementPolicy`(147 行)+ `terminal_stages_fn` |
| 引擎默认 | `disable_overlap_schedule=True`、`sampling_backend=pytorch`、torch.compile 硬拒绝 | thinker `enable_async_decode=True`;talker `max_seq_len=32768`;`SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` 防 FP8 首请求编译卡顿 |
| 兼容层 | `compat.py`(transformers 5.12 × qwen-tts 4.57 API 垫片)+ HF config 注册 | `vision_compat.py` + `hf_config.py`(409 行,模型族配置) |
| 内存 | `mem_fraction_static=0.85` 单引擎 | `total_gpu_memory_fraction` 分账 + colocated `encoder_mem_reserve`(给 image/audio encoder 留非 KV 显存) |
| 环境默认 | 确定性推理 opt-in(串行预处理/关声码器图) | colocated:`OMP_NUM_THREADS=8`、`TOKENIZERS_PARALLELISM=false` |

**结论 11:Omni 的配置复杂度主要花在"放置"(placement/内存分账/进程编排),TTS 的花在"确定性"(全局 opt-in 的三处代价)。** 两者的 `stage_factory_kwargs` 都是三行 if——但一个在分 GPU,一个在关并发。

## 9. 共享代码清单(改动需双侧回归)

| 共享物 | 宿主 | TTS 用法 | Omni 用法 |
|---|---|---|---|
| `Qwen3OmniMoeThinkerTextAttention` | thinker_model.py | TTS 解码层注意力 | Omni thinker/talker 解码层 |
| `Qwen3OmniMoeTalkerDenseMLP` / `ResizeMLP` / `_bind_default_weight_loaders` | talker.py | TTS FFN/投影 | Omni talker FFN(共享 expert 同源) |
| `apply_qk_norm` | vendor | 预测器注意力 | thinker/talker 注意力 |
| `QwenTalkerModelRunner` 4 个静态方法 | talker_model_runner.py | TTS runner 复用(decode 输入合成/重放/prefill 切片) | 自身 |
| `PendingTextTensorQueue` | pending_text_queue.py | 预装满的字幕机 | 流式同传队列 |
| `StreamingVocoderBase` | scheduling/ | Qwen3TTS 流式声码器 | Code2WavScheduler |
| `SGLangARRequestData` / `OmniPrefillInputs` / seed 派生 | scheduling/ sampling/ | 请求底座 | 同 |
| 预测器 CUDA 图模式(`_PredictorDecodeGraph`) | 各自实现 | 签名五元键 | (bucket, dtype) 二元键 |

**回归提示**:TTS 侧大量"import 自 omni"意味着 omni 目录的重构(尤其 talker.py / talker_model_runner.py / pending_text_queue.py)必须同时跑两边的测试——`tests/unit_test/` 下两族用例都在。

## 10. 汇总:一张决策表

当你要为新模型选型/新增能力时,两个实现给出了两套可复用的答案:

| 问题 | TTS 的答案 | Omni 的答案 |
|---|---|---|
| 条件信息何时可得? | 预处理期 → 静态 prompt + radix 哈希键 | 生成期 → 流式 hidden + 部分启动 |
| 采样要不要 per-request seed? | 要 → 双种子派生 + Triton 位级对齐 + 图签名 | 不要 → 图内静态 sampler + argmax 预测器 |
| 输入是 embedding 怎么接 KV cache? | 伪 token id(blake2b)+ extra_key 命名空间 | 占位符 + 多模态注入(thinker) |
| decode 输入怎么进图? | 行号寻址嵌入(共享方案) | 同 |
| 流式下游怎么交付? | ref 前置 + 合同锁定 metadata | 闩锁 + 流数据平面分离 |
| 实时播放怎么排程? | 播放截止钟优先级队列 | chunk 对齐 + 分解批 |
| retraction 怎么活? | 惩罚历史种回 + embedding 账本重放 | embedding 账本重放 + 缓冲失效 |
| 同步失败怎么办? | clamp-then-check + 进程级资源保留 | slot 隔离区 + 回收 |

**最后的总评**:Qwen3-TTS 的 7,180 行可以概括为"**把 Omni talker 的语音半边拆下来,补上 TTS 特有的条件注入、确定性采样与实时声码器**";Qwen3-Omni 的 13,468 行则是"**以 thinker 为中心的多模态编排系统**"。前者胜在单点深度(采样内核、图签名、增量 codec、故障边界),后者胜在系统广度(拓扑、放置、双流、部分启动)。读懂两者,基本就读懂了 sglang-omni 框架里"一个模型接入"的全部正面与背面。

---

## 附:本文档章节索引

- 第一篇 总体架构与全景导览 — 三阶段流水线、请求生命周期、任务矩阵、底座框架
- 第二篇 流水线阶段与请求构建 — 配置声明、兼容垫片、引擎装配、三级参考缓存、伪 token id
- 第三篇 AR 引擎与 Talker 模型 — 类层次、三条输入通道、prompt 组装数学、持久缓冲、权重装载
- 第四篇 CodePredictor 与 CUDA 图加速 — 预测器链、自管 KV、签名/桶/捕获/重放、降级矩阵
- 第五篇 采样体系与 Triton 内核 — Murmur3 Gumbel、torch.topk 位级对齐、bitonic 网络、三级 fallback
- 第六篇 ModelRunner 执行流水线 — 钩子协议、行号寻址、快照账本、retraction 惩罚恢复
- 第七篇 流式声码器与增量编解码 — 窗口/增量解码、双 worker、pinned 槽、故障保留
- 第八篇 与 Qwen3-Omni 深度对比 — 本篇
