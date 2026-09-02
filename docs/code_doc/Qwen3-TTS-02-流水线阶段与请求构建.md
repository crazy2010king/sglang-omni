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
