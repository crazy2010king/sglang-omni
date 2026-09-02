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

## 附:系列文档索引

1. `Qwen3-TTS-01-总体架构与全景导览.md` — 三阶段流水线、请求生命周期、任务矩阵、底座框架
2. `Qwen3-TTS-02-流水线阶段与请求构建.md` — 配置声明、兼容垫片、引擎装配、三级参考缓存、伪 token id
3. `Qwen3-TTS-03-AR引擎与Talker模型.md` — 类层次、三条输入通道、prompt 组装数学、持久缓冲、权重装载
4. `Qwen3-TTS-04-CodePredictor与CUDA图加速.md` — 预测器链、自管 KV、签名/桶/捕获/重放、降级矩阵
5. `Qwen3-TTS-05-采样体系与Triton内核.md` — Murmur3 Gumbel、torch.topk 位级对齐、bitonic 网络、三级 fallback
6. `Qwen3-TTS-06-ModelRunner执行流水线.md` — 钩子协议、行号寻址、快照账本、retraction 惩罚恢复
7. `Qwen3-TTS-07-流式声码器与增量编解码.md` — 窗口/增量解码、双 worker、pinned 槽、故障保留
8. `Qwen3-TTS-08-与Qwen3-Omni深度对比.md` — 本篇
