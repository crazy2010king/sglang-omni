# Qwen3-Omni 源码深度解析全集

> 基于仓库源码逐行核对写成，是 `qwen3-omni.md` 的深化版。由原 12 篇系列文档（00–11）合并为一册：
> 章节即原篇目编号，正文中的交叉引用（如"05 篇 §3.4""详见 02 篇"）按章号检索即可。
> 阅读顺序即章节顺序：先第 00 章建立全局框架，再按 01→11 逐一深入。

## 目录

| 章 | 标题 |
|----|------|
| 00 | Qwen3-Omni 总览与代码导读（整体框架篇） |
| 01 | 流水线框架与 Stage 运行时 |
| 02 | 调度器契约与双调度器 |
| 03 | 配置系统与部署拓扑 |
| 04 | 预处理与多模态编码器 |
| 05 | 多模态融合与 M-RoPE |
| 06 | Thinker 引擎与多模态嵌入注入 |
| 07 | Thinker-Talker 流式耦合与部分启动 |
| 08 | Talker 模型与解码交接 |
| 09 | Code2Wav |
| 10 | decode 阶段与流式反分词 |
| 11 | 通信引擎与张量传输 |

---

## 00 · Qwen3-Omni 总览与代码导读（整体框架篇）

> 本系列文档基于源码逐行核对，是对 `docs/code_doc/qwen3-omni.md` 的深化与纠偏。
> 本篇是"先有整体框架"的总纲：先建立全局拓扑与请求生命周期的心智模型，
> 后续 11 篇再逐一深入到最底层的实现细节。

---

### 0. 阅读本系列的方式与节奏

本系列采用 **"总—分—深"** 三层节奏：

| 层次 | 篇目 | 目标 |
|------|------|------|
| 总 | 00 总览（本篇） | 一张图 + 一次请求的完整旅程 + 核心抽象清单 |
| 分 | 01~03 框架层 | 流水线运行时、调度器契约、配置与拓扑 |
| 分 | 04~06 数据层 | 预处理/编码器、融合、M-RoPE |
| 深 | 07~10 引擎层 | Thinker 注入、Thinker-Talker 耦合、Talker 解码交接、Code2Wav |
| 深 | 11~12 终端与传输 | decode 反分词、通信引擎 |

每篇都遵循同一结构：**它在整条流水线中的位置 → 数据如何进来 → 内部如何算 → 数据如何出去 → 边界条件与工程权衡**。
所有结论均给出源码路径与关键行号，读者可随时回查。

---

### 1. Qwen3-Omni 模型本体：三个子模型

Qwen3-Omni（以 `Qwen/Qwen3-Omni-30B-A3B-Instruct` 为代表）在 HuggingFace 上的架构名是
`Qwen3OmniMoeForConditionalGeneration`，权重在物理上拆成三个子模型：

| 子模型 | 规模/角色 | 输入 → 输出 |
|--------|-----------|-------------|
| **Thinker**（思考者） | 30B MoE 主干 + Vision Tower + Audio Tower，文本自回归 | 文本/图像/音频/视频 token → 文本 token + 每步隐藏态 |
| **Talker**（说话者） | 独立的小型 MoE LM（codec 词表），接收 thinker 的隐藏态 | 隐藏态流 → 逐帧声学 codec 码（多组 RVQ 残差码） |
| **Code2Wav** | 流式声码器（DSP + 神经上采样） | codec 码 `[B, Q, T]` → 24kHz PCM 波形 |

三个关键 token 空间要分清：

1. **Thinker 文本词表**（约 15 万级，含 `image_token_id` / `video_token_id` / `audio_token_id` 占位符）；
2. **Talker codec 词表**（约 2 千级：`codec_bos_id=2149`、`codec_eos_token_id=2150`、`codec_pad_id=2148`、
   `codec_nothink_id=2155`、`codec_think_bos/eos=2156/2157`，外加说话人 embedding 行）；
3. **codec 码本**：`num_code_groups` 组 RVQ 码，第一组（layer0 码）由 talker 主干采样，
   其余残差组由 talker 内嵌的 **code_predictor** 贪心生成（见 08 篇）。

---

### 2. 两种部署拓扑：6 阶段文本 / 7 阶段语音

**先纠一个原文档的偏差**：`qwen3-omni.md` 写"8 阶段语音流水线"，但当前源码
`sglang_omni/models/qwen3_omni/config.py` 中语音拓扑是 **7 个 stage**：

```
文本拓扑（Qwen3OmniPipelineConfig，6 阶段）：
  preprocessing → image_encoder ─┐
                → audio_encoder ─┼→ mm_aggregate → thinker → decode(terminal)

语音拓扑（Qwen3OmniSpeechPipelineConfig，7 阶段）：
  preprocessing → image_encoder ─┐
                → audio_encoder ─┼→ thinker ──(next)──→ decode(terminal)
                                 │     │  ↑(stream 每步隐藏态)
                                 │     └─(stream 每步 token)→ decode
                (同时) preprocessing/encoders → talker_ar(独立 join)
                                        talker_ar ──(next)──→ code2wav(terminal)
                                        talker_ar ──(stream 每步 codec 帧)→ code2wav
```

为什么语音拓扑**没有 `mm_aggregate` 阶段**？因为 join（多路汇合）从"独立阶段"内联进了
`thinker` 与 `talker_ar` 两个 AR 阶段自身：二者的 `StageConfig` 都声明
`wait_for=["preprocessing","image_encoder","audio_encoder"]` + `wait_for_fn` + `merge_fn`
（`config.py:131-158` 与 `config.py:171-182`）。这是本仓库对原文档的一个实质演进——
join 内联后省掉一次全量 payload 的中转与投影，代价是 AR 阶段的调度器要承担多源输入聚合。

三份配置类（`config.py:295-360`）：

- `Qwen3OmniPipelineConfig`：纯文本，6 阶段，单进程 `process="pipeline"`；
- `Qwen3OmniSpeechPipelineConfig`：**分散式**默认，thinker 在 GPU0、talker_ar 链在 GPU1，partial-start 开启；
- `Qwen3OmniSpeechColocatedPipelineConfig`：**共置式**，5 个 GPU 阶段挤进一张卡，
  `enable_partial_start=False`，并注入 `OMP_NUM_THREADS=8` / `TOKENIZERS_PARALLELISM=false`
  防止 7 个进程把 CPU 打满（`config.py:34-38`）。

`Variants` 字典（`config.py:397-401`）把 `"text" / "speech" / "speech-colocated"` 映射到三份配置，
`EntryClass = Qwen3OmniSpeechPipelineConfig` 是仓库对外默认入口。

---

### 3. 请求的完整生命周期（以语音请求为例）

下面按时间顺序走一遍一次"图+文→语音回复"请求。每一步标注执行者与源码锚点。

#### 阶段 A：提交与预处理（CPU）

1. 客户端 → Coordinator（`pipeline/coordinator.py`）：`SubmitMessage` 送达 entry 阶段
   `preprocessing`。Coordinator 记录 `RequestInfo`，按 `terminal_stages_fn`
   （`request_builders.resolve_terminal_stages`：有音频输出时为 `[decode, code2wav]` 双终态）决定何时算完成。
2. `Qwen3OmniPreprocessor`（`components/preprocessor.py`）在 CPU 完成：
   chat template 渲染 → HF `Qwen3OmniMoeProcessor` 产出 `input_ids` + `mm_inputs` + 各模态
   `encoder_inputs`（pixel_values / input_features / 网格张量），并为媒体算 **cache_key**
   （xxhash）。产出写进 `Qwen3OmniPipelineState`，再通过 `project_payload` 按目标阶段"投影"
   成 3~4 份轻量 payload，分别发往 `image_encoder`、`audio_encoder`、`thinker`（+`talker_ar`）。
   动态路由由 `resolve_preprocessing_next_stages_speech` 决定：**没有图的请求不会空跑 encoder**。

#### 阶段 B：双塔编码（GPU）

3. `image_encoder`（`components/image_encoder.py`）：Vision Tower + **Conv3d→Linear 的 PatchEmbed
   替换**（kernel==stride 时数学等价，7~15× 提速）。支持跨请求批量（`stages.py:_batch_image_encoder_payloads`）
   与同批去重、4GiB CPU LRU 输出缓存。
4. `audio_encoder`（`components/audio_encoder.py`）：Audio Tower + **层间 CUDA 图**
   （`_GraphedLayerStack` 让 32 层循环只走一次 Python，`_SegmentSplits` 消除每层一次的 D2H 同步）。
   同样支持批量与去重。

两塔完成后各自经 `project_encoder_to_mm_aggregate` / `project_encoder_to_talker_ar`
（后者剔除 deepstack 多尺度特征，talker 用不到）把 `encoder_outs` 单键 payload 发出。

#### 阶段 C：Thinker（GPU，SGLang 引擎）

5. `thinker` stage 的 join：等齐 preprocessing + 活跃 encoder 源后执行 `merge_for_thinker`
   （`merge.py`）——把图/音频 embedding、网格、音频长度、`video_second_per_grid` 等装进
   `thinker_inputs.model_inputs`，并生成 `media_cache_keys`。
6. `build_sglang_thinker_request`（`request_builders.py`）构造 SGLang `Req`：
   - **pad 值替换**：把 `image/video/audio_token_id` 替换成 `vocab_size + xxh3_64(cache_key) % 2^62`
     的假 token id，防止 radix cache 把不同媒体的前缀错误共享；
   - CPU 上向量化计算 **M-RoPE 三轴位置**（见 06 篇）；
   - 预记录 `_omni_mm_positions`（各模态占位符的 prompt 绝对位置），免得 merge 时在 GPU 上做 mask 同步。
7. `OmniScheduler`（`scheduling/omni_scheduler.py`，SGLang 调度器的 omni 适配层）驱动
   chunked prefill（8192）+ **mixed chunk**（把在跑 decode 揉进 prefill 步，恢复 decode 占空比）+
   异步 decode。每生成一个 token：
   - `stream_output_builder`（`request_builders.make_thinker_stream_output_builder`）产出两条
     stream 消息：`torch.tensor([token_id]) → decode`，`hidden_state → talker_ar`；
   - 隐藏态来自 **静态捕获 buffer**（`_hidden_capture.py` 在第 0/24 层装 forward-pre-hook，
     chunk 进静态显存 buffer，CUDA-graph 安全）；
   - chunked prefill 进行中（`inflight_middle_chunks>0`）时**抑制隐藏态外流**，
     防止 prompt 侧隐藏态被当成首个 assistant token 泄进 TTS。

#### 阶段 D：Talker（GPU，第二条 AR 引擎）

8. `talker_ar` 在攒够 `partial_start_min_chunks`（分散式=5，硬下限 `MIN_PARTIAL_START_CHUNKS=3`）
   个 thinker 流块后**部分启动**（TTFA 优化）：不等 thinker 生成完。
9. `TalkerPrefillBuilder`（`components/talker_prefill.py`）**重建完整对话 prompt 的 talker 投影**：
   直接从 safetensors 里按行捞 thinker embedding（带缓存），文本行走 `text_projection`，
   多模态行走 `hidden_projection`；assistant 段构造 HF 对齐的 9 行布局
   （`[前3token]+[4×pad]+[tts_bos]+[第4token]` ⊕ `[3×0]+[6 个 codec 特殊 embedding]`），
   把"未来的文本行"交给 `PendingTextTensorQueue`（设备端 FIFO）。
10. 每个 decode 步的 `before_decode` 钩子（`talker_model_runner.py:45-66`）：
    `下一输入 = feedback(上一步 codec embedding) + text(队列头或 tts_pad_embed)`，
    两者都就绪才允许 decode，否则由 `QwenTalkerScheduler._is_batch_ready_to_run` 推迟
    并做 KV 池回滚（`_rollback_decode_prep_after_skip`）。
11. 主干采样出 layer0 codec 码 → `code_predictor` 用**逐层单回合 KV cache** 贪心补齐残差码 →
    `post_decode` 把 `[num_code_groups]` 码帧 `stream` 给 code2wav，同时把求和 embedding
    塞回自己的 `pending_feedback_queue`——这就是 talker 的"自反馈回路"。

#### 阶段 E：Code2Wav（GPU，流式声码器）

12. `Code2WavScheduler`（`components/code2wav_scheduler.py`）攒帧：每 `stream_chunk_size=10` 帧
    开一个滑窗（左上下文最多 `left_context_size=25` 帧），裁掉上下文对应的样本后输出增量音频；
    EOS 采用 **lazy scan**（批量扫描代替逐帧 `.item()` 同步）。
13. 开启 output-overlap 时：首个窗口同步 D2H（保证 TTFA），之后每个窗口 `non_blocking`
    拷进 **pinned 槽位** + event 记录，GPU 继续算下一窗，主机线程等上一窗 event 后再发——
    深度 2 的 D2H 流水线。
14. CUDA 图（`code2wav_cuda_graph.py`）：`GraphKey(batch_size, frames)` 精确形状捕获；
    batch=1 是"原子层"，失败即整体回滚；batch>1 尽力而为、逐键预算检查、失败缩容重试。

#### 阶段 F：文本终端

15. `decode` stage 的 `StreamingDetokenizeScheduler` 逐 token 反分词：不完整 UTF-8
    （出现 U+FFFD）会挂起待下一个 token 补齐；`stream_done` 与零 token 竞态用
    `_done_seen` OrderedDict 兜底；流式终态 result 刻意**不带全文**防客户端重复拼接。

---

### 4. 核心抽象清单（读任何一篇前先背下来）

#### 4.1 数据面三件套

- **`StagePayload`**（`proto/stage.py`）：跨阶段的信封 = `request_id + request + data`。
  `data` 是 **plain dict**（msgpack 可序列化），语义由各模型的 `PipelineState` 类解释。
- **`Qwen3OmniPipelineState`**（`payload_types.py:38`）：`payload.data` 的类型化视图，
  字段为 `raw_inputs / prompt / mm_inputs / encoder_inputs / encoder_outs / thinker_inputs /
  thinker_out / engine_outputs / stream_state`。`from_dict/to_dict` 负责与 dict 互转。
  **关键设计**：每个阶段的投影函数（`project_*`）本质是构造一个只含目标阶段所需字段的
  新 State——这是"IPC 开销裁剪"的全部手段。
- **`IncomingMessage / OutgoingMessage`**（`scheduling/messages.py`）：调度器 inbox/outbox 里的
  消息，`type ∈ {new_request, stream_chunk, stream_done, result, error}`。

#### 4.2 控制面 / 数据面分离

- **控制面**（ZMQ）：`SubmitMessage`、`DataReadyMessage`（"数据在哪"的通知）、
  `DataAckMessage`、`CompleteMessage`、`AbortMessage`、`AdminMessage`。只有控制消息走
  Coordinator/Stage 的 `control_plane`。
- **数据面**（relay）：真实张量走 `CommEngine` 选定的传输
  （`TransportKind = local_object | cuda_ipc | shm | mooncake`，见 `comm/data_ref.py:11`），
  控制面只传 `DataRef`（object_id + transport + 元信息）。收发双方用
  DataReady→read→Ack 握手，配额由 ack 驱动。

#### 4.3 Stage = IO 壳 + Scheduler

`Stage`（`pipeline/stage/runtime.py:100`）不做任何模型计算，只做五件事：
控制面收发、数据面读写、输入聚合（`InputHandler` + `wait_for`/`wait_for_fn`）、
流块路由（`StreamQueue` + `can_accept_stream_before_payload`）、把计算投递给 scheduler 的
inbox 并排空 outbox。Scheduler 分三类契约兼容的实现：

| 实现 | 用途 | 线程模型 |
|------|------|----------|
| `SimpleScheduler` | preprocessing / encoders（可选批量） | 专线程跑事件循环，`compute_fn` |
| `OmniScheduler`（及 `QwenTalkerScheduler`） | thinker / talker_ar（SGLang AR 引擎） | 专线程跑 `_event_loop_{async_decode,overlap,normal}` |
| `Code2WavScheduler` / `StreamingDetokenizeScheduler` | code2wav / decode（流式终端） | 专线程 + inbox/outbox |

#### 4.4 一个必须内化的不变量

**"payload 是快照，流是通道"。** `wait_for + merge` 聚合的是各源最终 payload；
而 token / hidden / codec 帧 / 音频样本这些**时序数据全部走 stream 通道**
（`stream_to`、`OutgoingMessage(type="stream", target=...)`）。
这也解释了为什么 `project_talker_to_code2wav` 只留一个空 latch（`request_builders.py:164-170`）：
"code2wav 该处理这个请求了"这条事实走 payload，真正的码张量走 stream。

---

### 5. 模块地图（带文件锚点）

```
sglang_omni/
├── pipeline/                      # 框架层
│   ├── coordinator.py             # 请求状态机/终态裁决/abort 广播/admin 聚合
│   ├── stage/runtime.py           # Stage IO 壳（本系列 01 篇主角）
│   ├── stage_workers.py           # 进程/worker 装配，故障域
│   ├── mp_runner.py               # 多进程 runner
│   └── replicas.py                # 副本拓扑与 RoundRobin 绑定
├── comm/                          # 02/12 篇
│   ├── engine.py                  # CommEngine：send/read + ack + KV 转移
│   ├── router.py                  # 传输选择：同 GPU CUDA IPC / shm / mooncake
│   ├── data_ref.py                # TransportKind/DataKind/DataRef
│   └── stage_io.py                # direct-IPC payload 的序列化捷径
├── scheduling/
│   ├── simple_scheduler.py        # 03 篇
│   ├── omni_scheduler.py          # 03/07 篇（SGLang AR 适配层，2677 行）
│   ├── streaming_vocoder.py       # 09 篇（Code2Wav 的模板基类）
│   └── sglang_backend/…           # server_args 装配、encoder_mem_reserve
├── models/qwen3_omni/
│   ├── config.py                  # 04 篇：三份 PipelineConfig
│   ├── stages.py                  # 各 stage 工厂 + 编码器批量 + 内存契约
│   ├── request_builders.py        # 请求构造/投影/流输出（全系列高频引用）
│   ├── merge.py                   # 05 篇：merge_for_thinker + decode_events
│   ├── mrope_positions.py         # 05 篇：向量化 M-RoPE
│   ├── pending_text_queue.py      # 07 篇：设备端 FIFO
│   ├── talker_scheduler.py        # 03/07 篇：partial start + decode 就绪策略
│   ├── talker_model_runner.py     # 08 篇主角
│   ├── thinker_model_runner.py    # 07 篇：prefill sidecar 适配
│   ├── bootstrap.py               # 06/07 篇：两个 AR scheduler 的装配
│   ├── placement.py               # 04 篇：拓扑合法性校验
│   └── components/
│       ├── preprocessor.py        # 05 篇
│       ├── image_encoder.py       # 05 篇：PatchEmbed 优化
│       ├── audio_encoder.py       # 05 篇：层图 + 段共享
│       ├── thinker_model.py       # thinker 文本主干（SGLang 层实现）
│       ├── thinker.py             # thinker 封装（含视觉注入）
│       ├── talker.py              # 08 篇主角（1811 行）
│       ├── talker_input.py        # 07 篇：HF 对齐 prefill 布局
│       ├── talker_prefill.py      # 07 篇主角
│       ├── code2wav_scheduler.py  # 09 篇主角
│       ├── code2wav_cuda_graph.py # 09 篇主角
│       ├── audio_layer_graph.py   # 05 篇：音频塔层图
│       └── streaming_detokenizer.py # 10 篇主角
├── model_runner/
│   ├── base.py                    # ModelRunner 钩子契约（before_prefill/before_decode/…）
│   ├── model_worker.py            # 06 篇：arch 覆盖 + 量化归一
│   ├── thinker_model_runner.py    # 07 篇：多模态嵌入注入
│   └── _hidden_capture.py         # 07 篇：静态隐藏态捕获
└── relay/                         # cuda_ipc / shm / mooncake / nccl / nixl
```

---

### 6. 阅读源码的推荐路径

1. **先跑通一个最小请求的 trace**：`SGLANG_OMNI_TRACE_ENCODER_CACHE=1` + profiler
   event recorder，观察 `stage_input_received → stage_dispatch → encoder_start/end → …`。
2. **按一条数据流读**，而不是按目录读：
   `preprocessor.py → merge.py → request_builders.build_sglang_thinker_request →
   thinker_model_runner._inject_multimodal_embeds → talker_prefill.build_prompt_prefill →
   talker_model_runner.before_decode → code2wav_scheduler.decode_delta`。
3. **框架层只需精读三个文件**：`stage/runtime.py`、`simple_scheduler.py`、`omni_scheduler.py`
   的主循环；其余框架代码按需跳转。
4. 遇到"为什么这么写"的疑问，**先搜代码里的 `Note (...):` 注释**——本仓库把几乎所有
   非平凡决策（含 PR 号与失败案例）都写成了就地注释，这是最硬核的一手资料。

---

### 7. 本系列章节导航

| 篇 | 位置 | 核心问题 |
|----|------|----------|
| 01 | 第 01 章 | Stage 如何聚合多源输入、路由流、处理崩溃 |
| 02 | 第 02 章 | Simple/Omni 双调度器与 talker 就绪策略 |
| 03 | 第 03 章 | 三份配置、placement 校验、内存契约 |
| 04 | 第 04 章 | CPU 预处理、双塔、批量与缓存 |
| 05 | 第 05 章 | merge、pad 值哈希、三轴位置 |
| 06 | 第 06 章 | SGLang Req 装配、注入、隐藏态捕获 |
| 07 | 第 07 章 | partial start、prefill 重建、FIFO |
| 08 | 第 08 章 | talker 前向、code predictor、自反馈 |
| 09 | 第 09 章 | 滑窗、重叠 D2H、精确形状图 |
| 10 | 第 10 章 | UTF-8 边界、竞态、终态契约 |
| 11 | 第 11 章 | 传输选择、握手、KV 转移 |


---

## 01 · 流水线框架与 Stage 运行时

> 主角：`sglang_omni/pipeline/stage/runtime.py`（1843 行）、`pipeline/coordinator.py`（875 行）、
> `pipeline/control_plane.py`、`pipeline/stage/input.py`、`pipeline/stage/stream_queue.py`。
> 上一篇建立了全景，本篇拆开框架的承重墙：**Stage 到底替你管了什么**。

---

### 1. 一条消息在框架里的分类学

Stage 收到的所有东西只有六种（`runtime.py:_handle_message`）：

| 消息 | 来源 | 语义 |
|------|------|------|
| `SubmitMessage` | Coordinator | 新请求的入口 payload（仅 entry 阶段） |
| `DataReadyMessage` | 上游 stage | "有一份数据可以来读"（payload / 流块 / 流信号 三态由字段区分） |
| `DataAckMessage` | 下游 stage | 对我方发出的数据的确认（配额/清理依据） |
| `CompleteMessage` | 上游 terminal | 终态回执 |
| `ProfilerStart/Stop`、`AdminMessage`、`ShutdownMessage` | Coordinator | 治理面 |
| `TPWorkMessage` | TP leader fanout | TP follower 收到的"干活"指令 |

`DataReadyMessage` 内部再三分流（`_schedule_receive_task`，`runtime.py:391-407`）：

- `is_done=True` 或 `error!=None` → `_on_stream_signal`（流终止/异常信号）；
- `chunk_id!=None` → `_on_stream_chunk`（时序流块）；
- 否则 → `_on_data_ready`（一次性 payload）。

三分流对应三种读取路径：**direct CUDA IPC 反序列化 → inline（消息内嵌小张量）→ relay 读**。
区分点在 `data_ref` 的 `object_id` 约定（`stage_io.is_direct_cuda_ipc_payload_ref` 等谓词）。
direct-IPC 与 inline 分支不经过 `CommEngine.read_stream_chunk`，因此必须**自己补发**
`comm_stream_read` 追踪事件（代码里有明确注释：否则同 GPU 边沿的 held-byte 统计会漏掉）。

---

### 2. 输入聚合：`wait_for` / `wait_for_fn` / `merge_fn`

这是理解 Qwen3-Omni 语音拓扑的钥匙。`mm_aggregate`（文本拓扑）和 `thinker`/`talker_ar`
（语音拓扑）都要等三个上游。

#### 2.1 聚合状态机

`InputHandler`（`stage/input.py`，两种实现：`DirectInput` 单源直通 / 多源聚合）按
`(request_id, 逻辑源名)` 收 payload：

1. 每个 `DataReadyMessage` 到达后 payload 被塞进聚合器；
2. 聚合器检查 `wait_for` 集合是否集齐；
3. 集齐 → `receive()` 返回 merged payload → Stage 调 `_execute(merged)` → 塞进 scheduler.inbox。

#### 2.2 动态等待集合：`wait_for_fn`

静态 `wait_for=["preprocessing","image_encoder","audio_encoder"]` 有个问题：纯文本请求没有
encoder 源，会永远等不齐。所以 Qwen 配置挂了 `wait_for_fn =
request_builders.resolve_mm_aggregate_wait_sources`（`request_builders.py:96-105`）：

```python
def resolve_mm_aggregate_wait_sources(request_id, from_stage, payload):
    if from_stage != "preprocessing":
        return None            # 维持现有等待集合，不缩
    state = Qwen3OmniPipelineState.from_dict(payload.data)
    return ["preprocessing", *_active_encoder_stages(state.encoder_inputs)]
```

语义约定：**preprocessing 的 payload 到达时**才有资格收窄等待集合——因为只有
preprocessing 知道本次请求带了哪些模态（`encoder_inputs` 里的 `_active` 标记，
由 `_has_encoder_model_input` 判定：图像看 `pixel_values/pixel_values_videos`，
音频看 `input_features`，且 `_skip=True` 的直接剔除）。
其它源到达时返回 `None` = "别动"。

#### 2.3 merge_fn：集齐之后做什么

集齐的 payload 们进 `merge_fn`。语音拓扑里 thinker 与 talker_ar 共用同一个上游集合，
但 merge 不同：

- thinker：`merge.merge_for_thinker` —— 生成完整 `thinker_inputs`（05 篇详拆）；
- talker_ar：`request_builders.merge_for_talker` = `project_mm_aggregate_to_talker_ar(
  merge_for_thinker(payloads))` —— **复用 thinker 的融合结果再做一次"早提交投影"**：
  剔除 deepstack 键，只保留 `prompt + thinker_inputs`（`request_builders.py:213-232`）。
  talker 的 V1 prefill 策略就是把整个 thinker prompt 重放为投影 embedding，
  所以它需要的输入和 thinker 几乎一样，只在特征取舍上不同。

#### 2.4 为什么 join 放进 AR 阶段而不是独立聚合阶段

原文本拓扑有独立 `mm_aggregate` 阶段（`_aggregate_stage`），语音拓扑取消之（见 00 篇）。
收益：省一次全量 payload 的收发与两次投影；代价：`OmniScheduler` 必须能接收
"一份数据到达时可能还差别的源"的语义，并且聚合发生在 stage 线程（asyncio）而非调度线程。
注意 `disable_direct_cuda_ipc_payload=True` 同时挂在 mm_aggregate 与 audio_encoder 上——
聚合阶段与音频塔明确禁用 direct IPC payload 捷径（多源合并前 payload 必须是可复制的普通内存）。

---

### 3. 接收顺序的硬保证：receive lane

多源、多流块的到达是乱序的，但 Stage 必须保证**同一逻辑源内的事件按到达序处理**。
实现是"每 lane 一个 future 链"（`runtime.py:391-434`）：

```python
lane = (request_id, from_stage)
predecessor = self._receive_lane_tails.get(lane)      # 该 lane 上一个任务
completion = loop.create_future()
self._receive_lane_tails[lane] = completion
task = create_task(self._run_receive_task(handler(msg, predecessor), lane, predecessor, completion))
```

每个接收任务先 `await predecessor` 再继续，形成**单 lane 串行、跨 lane 并行**的 DAG。
这解决了两个真实竞态：
1. 同一上游的 stream_chunk 与 payload 乱序（payload 先到才能建 `StreamQueue` 条目）；
2. abort 后迟到数据的丢弃顺序。

---

### 4. 流通道：StreamQueue 与 pre-payload 流

`StreamQueue`（`stage/stream_queue.py`）维护 `(request_id → 有序 StreamItem 队列)`。
关键开关是 Stage 构造参数 `can_accept_stream_before_payload`：

- `decode`、`talker_ar`、`code2wav` 都为 `True`（`config.py` 中三个 stage 均显式声明）；
- 原因：thinker 一边跑 prefill 一边就开始每 token 发流，此时 thinker 的终态 payload
  （`next="decode"` 那份）还没到。接收端必须允许"流先于 payload"。

`_open_pre_payload_stream_if_allowed`（`runtime.py:1193-1202`）：
队列已有 → 直接路由；没有且允许 → 先 `stream_queue.open(request_id)` 再路由；
不允许 → abort 该请求并向 Coordinator 发 failure（"stream arrived before the request
payload, but this stage is not configured to accept pre-payload stream data"）。

`stream_done` 同样受此开关管辖（`_receive_stream_signal`），到达后除入队外还向 scheduler.inbox
投一条 `IncomingMessage(type="stream_done")`——这是 OmniScheduler/Code2Wav 感知"上游说完了"
的唯一入口。

---

### 5. `_execute`：把请求交给调度器

```python
async def _execute(self, payload):
    if payload.request_id in self._aborted: return
    if self.role == "leader" and self._tp_fanout is not None \
       and getattr(self.scheduler, "requires_tp_work_fanout", False):
        self._tp_fanout.fanout_work(payload)      # TP: rank0 把活分给 followers
    msg = IncomingMessage(request_id=..., type="new_request", data=payload)
    if (enqueue := getattr(self.scheduler, "enqueue", None)):
        enqueue(msg)                               # OmniScheduler 有带准入控制的 enqueue
    else:
        self.scheduler.inbox.put(msg)              # Simple/Code2Wav 直接进队列
```

三个细节：
- **TP fanout**：`role="leader"` 且调度器声明 `requires_tp_work_fanout` 时，leader 先把
  payload 广播给 followers 再自己跑（follower 的 Stage 主循环直接把 `TPWorkMessage`
  转 `_execute`）。
- **scheduler 线程模型**：scheduler 在独立 daemon 线程跑（`Stage.start` 里创建，
  `gpu_id is not None` 时先 `set_device`），与 Stage 的 asyncio 主循环通过两个
  `queue.Queue`（inbox/outbox）解耦。线程崩溃通过
  `asyncio.run_coroutine_threadsafe(self._handle_scheduler_crash(exc), loop)` 回传主循环。
- **admin 操作**走 `loop.run_in_executor` 调 `scheduler.admin(action, payload)`，
  TP leader 会额外收集 followers 的 `AdminResult` 并聚合（`_on_admin`），失败判定为
  `all(rank_results.success)`。

---

### 6. Outbox 排空与结果路由

`_drain_outbox` 是一个常驻任务：`loop.run_in_executor` 以 0.1s 超时阻塞取 outbox，
单轮最多连取 `_OUTBOX_DRAIN_BATCH_SIZE=64` 条（取满后 `await asyncio.sleep(0)` 让出事件循环）。
三类出口（`_drain_outbox_external`）：

1. `type="result"` → `_route_result`：
   - 先给 `get_stream_done_targets(request_id, result)`（thinker 用
     `resolve_thinker_stream_done_targets`：要音频 → `[talker_ar, decode]`）逐个发 `stream_done`；
   - 再问 `get_next(request_id, result)`：
     - 返回 `None` → terminal，`send_complete` 给 Coordinator（携带 `result.data`）；
     - 返回 stage 名（单/多）→ `_send_to_stage`，并按目标数选择
       `allow_local_object`（单目标可走进程内对象直传）或
       `allow_projected_local_object`（多目标时投影后直传）。
2. `type="stream"` → `target is None` 时走 `stream_targets`（构造时静态配置，如 talker→code2wav）
   或直接发 Coordinator（terminal 流，如 decode 的文本增量）；有 `target` 则点对点。
3. `type="error"` → `_send_failure`。

follower 的排空是阉割版（`_drain_outbox_follower`）：result 只清状态、stream 直接丢、
error 直接抛——**外部 IO 是 leader 独占的**。

#### 路由前的投影：project_payload

`_send_to_stage` 在真正发送前查 `self._project_payload[target]`，把 payload 压成
目标阶段需要的形状。Qwen 使用的全部投影（`request_builders.py`）：

| 投影 | 从 → 到 | 保留什么 |
|------|---------|----------|
| `project_preprocessing_to_image_encoder` | preprocessing → image_encoder | 仅该塔的 `encoder_inputs` |
| `project_preprocessing_to_audio_encoder` | preprocessing → audio_encoder | 仅该塔的 `encoder_inputs` |
| `project_preprocessing_to_mm_aggregate` | preprocessing → join 目标 | `prompt` + 轻量 `mm_inputs`（只留 grid/mask/lengths 等元数据）+ `encoder_inputs` 元数据（cache_key/_skip/_active） |
| `project_encoder_to_mm_aggregate` | encoder → join 目标 | 仅本塔 `encoder_outs`（强制单键，否则 ValueError） |
| `project_encoder_to_talker_ar` | encoder → talker_ar | 同上，但剔除 `deepstack_visual_embeds_*`（talker 不用，见 `_TALKER_UNUSED_ENCODER_OUT_KEYS`） |
| `project_thinker_to_decode` | thinker → decode | 清 `thinker_inputs`、`extra_model_outputs`，深拷贝可变容器（张量除外） |
| `project_talker_to_code2wav` | talker_ar → code2wav | **空 data**（纯 latch，码张量走流） |

`project_thinker_to_decode` 里 `_copy_mutable_containers` 值得注意：dict/list/tuple/set/bytearray
递归复制，但 **torch.Tensor 原样返回**——共享引用以省拷贝，因为下游 decode 只读 token id。

---

### 7. 复制/_abort 与生命期管理

- `_active_requests` 集合是"我正在参与该请求"的事实记录；outbox 消息只有
  `request_id ∈ _active_requests` 才会被路由，`_clear_request_state` 在 result 路由完成后清理。
- `_aborted` 集合挡住迟到的 payload/流块（各 `_on_*` 入口第一件事就是查它）。
  abort 监听是常驻任务 `_abort_listener`（control plane 推 `AbortMessage`）。
- 请求结束后还必须**排干在途数据**（`_discard_data` / `_discard_stream_chunk_data`）：
  即便请求已 abort，已分配的 relay 对象仍要读一次并发 Ack，否则发送端的配额窗口会泄漏。
  KV_PAGES 是例外：直接拒绝并报"request was aborted"（KV 池页有独立回收路径）。

---

### 8. 崩溃语义：谁坏了、坏多大

框架的故障模型在多处注释里写得很清楚：

1. **scheduler 线程崩溃** → `_handle_scheduler_crash` → Stage `_running=False`，
   主循环 `run()` 在 finally 里重抛，进程级故障域（`stage_workers._run_process` 中同进程
   的多个 `role="single"` stage 共享进程也共享故障域）。
2. **单请求计算失败**（scheduler 内部 try/except）→ 只发该请求的 error 消息，例如
   `StreamingDetokenizeScheduler.start` 的注释明确说：异常若逃出 `start()` 会触发
   Stage 的 scheduler 崩溃处理，连带杀掉该 stage 上所有在途请求——所以每个消息循环
   都包了 per-request 的 try/except。
3. **后台任务失败**（abort 监听、outbox 排空、接收任务）→
   `add_done_callback(self._on_background_task_done)` 记录 `_background_task_error`，
   主循环退出时重抛。绝不允许静默吞掉。
4. **Coordinator 视角**：`fail_pending_requests` 把所有在途请求置 FAILED 并向各
   stream queue 注入失败 Complete；`shutdown_stages` 广播 Shutdown。

---

### 9. Coordinator：请求状态机

`coordinator.py` 的职责清单：

- **stage 注册表**：`register_stage(name, endpoint)`；
- **请求跟踪**：`_requests: dict[str, RequestInfo]`、`_completion_futures`、
  `_stream_queues: dict[str, asyncio.Queue[CompleteMessage|StreamMessage]]`——
  客户端迭代的就是后者；
- **终态裁决**：`terminal_stages` 或 `terminal_stages_resolver`（Qwen 的
  `resolve_terminal_stages` 按请求元数据返回 `[decode]` 或 `[decode, code2wav]`）。
  多终态 = **全部**终态都回执才算完成（语音请求同时等文本与音频两条链）；
- **准入**：`max_in_flight`（= 生成容量 + 队列容量），超限抛 `QueueFullError`；
- **副本绑定**：`ReplicaTopology` + `RoundRobinBindingPolicy`，每个请求为复制的
  Process 选一个副本实例，绑定随消息传播（`replica_bindings` 字段），
  Stage 端用 `_resolve_target_instance` 把逻辑目标名翻译成物理实例名；
- **admin 聚合**：`AdminOperation(op_id, action, payload, target_stages, timeout_s)`，
  `_AdminPendingOperation` 等所有目标回执，`_aggregate_admin_results` 汇总；
- **abort**：`AbortMessage` 只带 request_id；持强引用的 abort task 保证"本地准入已关 +
  广播能发出去"，即使调用方协程被取消。

---

### 10. 本篇小结（与其他篇的接口）

- Stage 是**无计算 IO 壳**：聚合（wait_for/wait_for_fn/merge_fn）→ 执行（scheduler inbox）→
  路由（get_next + project_payload + stream 通道）。
- 流与 payload 是两套物理通道、一套控制面；`can_accept_stream_before_payload` 是流先行的开关。
- 下一篇（02）进入 Stage 的对偶物：scheduler 侧的两种主循环与 OmniScheduler 的 AR 调度细节。


---

## 02 · 调度器契约与双调度器：SimpleScheduler / OmniScheduler / QwenTalkerScheduler

> 主角：`sglang_omni/scheduling/simple_scheduler.py`（327 行）、
> `sglang_omni/scheduling/omni_scheduler.py`（2677 行）、
> `sglang_omni/models/qwen3_omni/talker_scheduler.py`（158 行）、
> `sglang_omni/scheduling/streaming_vocoder.py`（Code2Wav 的模板基类）。

---

### 1. 唯一契约：inbox / outbox

框架对 scheduler 的全部要求（`Stage` 只依赖这些）：

```python
scheduler.inbox:  queue.Queue[IncomingMessage]   # new_request / stream_chunk / stream_done
scheduler.outbox: queue.Queue[OutgoingMessage]   # result / stream / error
scheduler.start()          # 阻塞当前线程（Stage 把它放进 daemon 线程）
scheduler.stop()
scheduler.abort(request_id)
scheduler.admin(action, payload) -> dict | AdminResult   # 可选
scheduler.requires_tp_work_fanout: bool                  # 可选
scheduler.enqueue(msg)                                   # 可选：OmniScheduler 的准入入口
```

`IncomingMessage.type` 只有三种与数据有关：`new_request`（整包 payload）、
`stream_chunk`（`data` 是 `StreamItem(chunk_id, data, from_stage, metadata)`）、
`stream_done`。**模型代码永远不碰 ZMQ/relay**——这就是"Stage = IO 壳"换来的解耦。

---

### 2. SimpleScheduler：非 AR 阶段的极简实现

#### 2.1 主循环

`_start_serial`：`inbox.get(timeout=0.1)` → 对 `new_request` 调 `_collect_batch` →
`_run_batch` → outbox。所有异常被捕获后**逐请求**发 error 消息；abort 过的请求
（`_consume_if_aborted`，带锁的 check-and-clear）静默丢弃。

#### 2.2 批量收集算法（编码器批量依赖它）

`_collect_batch(first_msg)`：

- 没配 `batch_compute_fn` 或 `max_batch_size<=1`：单条返回；
- 否则循环 `get_nowait` 攒批：
  - `batch_wait_when_idle=True` 时从**第一条消息到达**起算 deadline
    （`_encoder_batch_wait_ms` 环境变量控制，默认 0 = 不等）；
  - **代价上限**：`request_cost_fn` 计算每条消息的字节成本，
    `batch_cost + msg_cost > max_batch_cost` 时把该消息**塞回队首**
    （`_pending_messages.appendleft`）并结束攒批。
    图像编码器的成本函数（`stages.py:_create_image_encoder_request_cost_fn`）=
    `(原始像素字节 + 视觉 token 输出字节) × 5`，预算 `10 GiB`；
    视觉 token 数 = `grid_thw.prod(-1) // merge²`，输出层含 deepstack 的
    `1 + deepstack_layers` 份。
- 非 `new_request` 消息（流块等）一律塞 `_pending_messages` 延后处理，不打断攒批。

#### 2.3 并发模式

`max_concurrency > 1` 时切换到 `_start_concurrent`：N 个 worker 协程 +
`asyncio.to_thread(self._run_compute_in_thread, ...)`，把同步 compute_fn 挪出事件循环；
与 batch 路径互斥（构造时校验）。Qwen 的 preprocessing/encoders 都用默认串行 +
批量路径。

#### 2.4 Qwen 怎么用它

- preprocessing：`compute_fn = await preprocessor(payload)`（异步），单条；
- image/audio encoder：`_encode` 单条 + `_encode_batch` 批量（`stages.py:505-640`），
  `max_batch_size=32`；
- aggregate：`_identity`（join 已由 Stage 完成，merge 已在 Stage 侧做过）。

---

### 3. OmniScheduler：SGLang AR 引擎的适配层

`OmniScheduler` 是整个仓库最重的类（2677 行）。它**不是**重新实现调度，而是把
SGLang 的 `Scheduler`（schedule_batch / forward_batch / kv pool / cuda graph 那一套）
包进 inbox/outbox 契约。理解它只需要抓主循环和五个挂点。

#### 3.1 主循环（`start` → `event_loop`）

```python
def start(self):
    if self.enable_async_decode:   self._event_loop_async_decode()
    elif self.enable_overlap:      self._event_loop_overlap()
    else:                          self._event_loop_normal()
```

Qwen thinker 默认 `enable_async_decode=True`（`config.py` FactoryArgs）。
每轮循环的骨架（各 loop 变体共享）：

1. `recv_requests` → `_recv_scheduler_messages`：从 SGLang 的输入通道 + 本地 inbox 排空，
   合成 recv 列表；
2. `process_input_requests`：
   - `new_request` → **请求构建队列**：`_run_request_builder` 在独立线程池执行
     `request_builder(payload)`（即 `build_sglang_thinker_request`，含 CPU 上的 M-RoPE
     计算，不能阻塞调度循环）；构建完 `_admit_or_defer_built_request` 做准入
     （KV 容量检查 `_request_kv_capacity_error` / 等待队列满 `_reject_queue_full`）；
   - `stream_chunk` → `_on_stream_chunk`：`request_id` 已在册 →
     `stream_chunk_handler(req_data, chunk)`（talker 是 `prefill_builder.append_text_chunk`）；
     不在册 → 放进 `_pending_stream_ingress`（**流先于 payload** 的兜底）；
   - `stream_done` → `_on_stream_done`：置 `thinker_chunks_done`
     （talker 还会把 `tts_eos_embed` 追加进 pending_text_queue，见 07 篇）；
3. `get_next_batch_to_run` → SGLang 批次组装（含 deferred 请求的 recheck）；
4. `_run_batch`：`model_runner.execute`（进 ModelRunner 的
   before_prefill/before_decode/…钩子，见 06/08 篇）；
5. `process_batch_result` → `stream_output(...)`：把每步产出转成 outbox 消息
   （result / stream / error）。

#### 3.2 五个挂点（模型定制面）

| 挂点 | thinker 的实现 | talker 的实现 |
|------|----------------|---------------|
| `request_builder` | `make_thinker_scheduler_adapters()[0]` → `build_sglang_thinker_request` | `make_talker_scheduler_adapters()[0]` → `_build_talker_request_data`（要求已有 prefetched chunks） |
| `result_adapter` | `apply_thinker_result` 回写 `state.thinker_out` | 恒等（保持 payload 原样） |
| `stream_chunk_handler` | 无（thinker 是流的**生产者**） | `prefill_builder.append_text_chunk`：把隐藏态投影成 talker 行，追加进 pending_text_queue |
| `stream_done_handler` | 无 | `prefill_builder.mark_thinker_done`：置 done + 追加 tts_eos 行 |
| `stream_output_builder` | `make_thinker_stream_output_builder`：token→decode、hidden→talker_ar | 无（Code2Wav 的码帧由 ModelRunner 的 post_* 直接 outbox） |

#### 3.3 partial start（部分启动）的调度策略

`QwenTalkerScheduler` 覆写三个决策函数（`talker_scheduler.py:51-108`）：

1. `_is_request_build_ready`：非 pending_stream_done 时，必须
   `enable_partial_start` 且可用块数 ≥ `partial_start_min_chunks` 才允许构建 talker 请求。
   "可用块数"由 `_count_usable_prefetched_chunks` 计算：**末块若是 `<|im_end|>`
   （token_id == im_end_token_id）要扣掉**——它只是会话边界标记，不是语音内容。
   硬下限：`MIN_PARTIAL_START_CHUNKS=3`（构造时校验，`partial_start_min_chunks < 3` 直接
   ValueError）。分散式配置传 5（`config.py:181`），共置式关掉 partial start。
2. `_should_recheck_deferred_request_on_stream_chunk`：partial start 开启时，
   每来一个流块都把 deferred 请求标脏重查（攒够就立刻晋升为可构建）。
3. `_is_batch_ready_to_run` + `get_next_batch_to_run`：decode 批次还得过
   `model_runner.is_decode_batch_ready`（08 篇：每个请求都有 feedback+text 输入）。
   不就绪就 `_rollback_decode_prep_after_skip(batch)`——**这是全仓库最精妙的逆向操作之一**：

```python
# SGLang 的 prepare_for_decode 已经：分配了 KV 槽、递增了 seq_lens、
# 写了 req_to_token、递增了每请求的 decode_batch_idx / kv_committed_len。
# 回滚必须对称地撤销，但 seq_lens_sum 保持不动（此时恒为 None，
# 下次 forward 前会重算），penalizer 的 cumulate scatter_ 在 talker
# 自己的 SamplingBatchInfo 下是幂等的，可以不逆。
self.token_to_kv_pool_allocator.free(batch.out_cache_loc)
for req in batch.reqs:
    req.decode_batch_idx -= 1
    req.kv_committed_len -= 1
    req.kv.kv_allocated_len -= 1
batch.seq_lens.sub_(1); batch.seq_lens_cpu.sub_(1); batch.orig_seq_lens.sub_(1)
batch.req_to_token_pool.req_to_token[batch.req_pool_indices, batch.seq_lens] = 0
```

没有这段，"decode 批组好了但文本输入还没到"就会导致 KV 泄漏或池越界。

#### 3.4 talker 的 server_args 覆写

`configure_talker_server_args`（`talker_scheduler.py:14-27`）：

- `disable_radix_cache=True`：talker 输入是投影 embedding，前缀共享无意义且危险；
- `chunked_prefill_size=0`：talker prefill 一次性吃完（其输入本来就是自己构造的 9+行）；
- feedback 开启时 `disable_overlap_schedule=True`：overlap 调度会在 forward 与
  结果处理之间引入异步窗口，而 talker 的 `before_decode` 要同步写 feedback buffer，
  两者冲突。

返回值 `want_cuda_graph` 决定调用方是否在 ModelWorker 建好后补跑 `init_sglang_cuda_graphs`
（`bootstrap.py:233-236`）。

#### 3.5 流输出的落点

`stream_output(reqs, ...)`（`omni_scheduler.py:1579`）把 `stream_output_builder` 产出的
`list[OutgoingMessage]` 经 `_put_stream_messages` 压进 outbox。thinker 的构建器
（`request_builders.py:make_thinker_stream_output_builder`）有三条关键规则：

1. `req.inflight_middle_chunks > 0`（chunked prefill 还有中段没吃完）→ **整体不发**。
   注释原文：过早发出的块会让 prompt 侧隐藏态"冒充"第一个 assistant token，
   把用户/ref 文本泄进 TTS。
2. `stream=True` 的请求才发 token→decode（非流式请求不发 per-token 消息），
   但 **talker_ar 的隐藏态流无条件发**（只要请求要音频）——talker 不关心你的客户端
   是否流式，它只关心自己的输入。
3. 隐藏态要拆双层：`extra["hidden_states"]` 可能是 dict
   `{"embed": 层0输出, 24: 层24输出}`（对应捕获层 `[0, 24]`），
   embed 缺失时回退 layer_hidden；单个 1D/2D 张量归一化成 1D。

---

### 4. StreamingVocoderBase：Code2Wav 的模板基类

`streaming_vocoder.py` 顶部的 docstring 就是权威规格，摘要：

- 请求级流状态注册表：首个 chunk/payload 时 `create_stream_state`，
  结束/abort 时 pop，**迟到的 chunk 永远不能重建已清理的状态**；
- `is_streaming_payload = bool(params["stream"])`；
- 契约闩锁 `latch_stream_contract`：首个流块到达时把 stream 开关锁进状态
  （Code2Wav 用它锁 `stream_enabled`，之后不再逐块判断）；
- 阈值-累积→decode_delta→emit 的固定顺序，波形统一包成
  `OutgoingMessage(type="stream", metadata={"modality":"audio"})`；
- `on_stream_done` 顺序：flush 尾窗 → 末块流消息 → 终态 result → 清状态；
- 一块都没发过时走 `fallback_full_decode` 整段兜底；
- 跨请求合批通过四个可选钩子：`select_step_participants / build_step_plan / run_step /
  on_step_failure`——失败的 step 只会 error 掉其参与者，**绝不能殃及别的流**。

Code2Wav 在此之上叠加滑窗/EOS/重叠/图（09 篇），decode 的
`StreamingDetokenizeScheduler` 则没有用它（自建 inbox 循环，10 篇讲原因）。

---

### 5. 小结

- 调度器契约极窄（两个队列 + 生命周期方法），所有"分布式复杂性"被 Stage 吸收；
- SimpleScheduler 的价值在批量收集的**代价模型**（成本函数 + 预算 + 回塞队首）；
- OmniScheduler 的价值在**挂点设计**：request_builder（CPU 重活进线程池）、
  stream_chunk/done handler（流的消费端注入）、stream_output_builder（流的生产端格式）；
- talker 的三处覆写（部分启动门槛、流块触发重查、decode 就绪回滚）是
  "流式 AR 引擎"这个概念落到工程上的全部代价。

下一篇：三份 PipelineConfig 如何把以上所有组件拼成可部署拓扑（04 篇前的框架终章）。


---

## 03 · 配置系统与部署拓扑：三份 PipelineConfig、placement 校验与内存契约

> 主角：`sglang_omni/models/qwen3_omni/config.py`（403 行）、
> `sglang_omni/models/qwen3_omni/placement.py`（147 行）、
> `sglang_omni/config/schema.py`（EngineStageConfig/StageConfig/PlacementConfig 等）、
> `sglang_omni/models/qwen3_omni/stages.py` 的内存契约部分。

---

### 1. StageConfig：把一个阶段声明成数据

框架的核心理念是**声明式拓扑**：每个 stage 是一份纯数据（pydantic 模型），
进程装配器读取这些数据决定"起几个进程、每个进程里跑哪些 stage、连什么边"。
Qwen 用到的字段：

```python
StageConfig(
    name="audio_encoder",
    process="audio_encoder",                    # 逻辑进程名（可多 stage 共进程）
    factory_path="...stages.create_audio_encoder_executor",   # 工厂函数
    factory=FactoryArgs(enable_layer_cuda_graph=True),        # 工厂参数
    gpu=gpu,                                    # GPU 绑定
    next=["thinker", "talker_ar"],              # 静态出边
    route_fn="...resolve_encoder_next_stages",  # 动态路由（覆盖 next）
    wait_for=[...], wait_for_fn="...",          # 多源 join
    merge_fn="...",                             # join 后的融合
    project_payload={"目标": "投影函数"},         # 出边投影
    stream_to=["code2wav"],                     # 流出边
    stream_done_to_fn="...",                    # 动态流终止目标
    terminal=True,                              # 终态阶段
    can_accept_stream_before_payload=True,      # 允许流先于 payload
    disable_direct_cuda_ipc_payload=True,       # 禁用 direct-IPC payload 捷径
)
EngineStageConfig(...)   # StageConfig 子类：AR 阶段，engine 子配置（SGLang server_args 覆写）
```

`_Qwen3OmniBasePipelineConfig.stage_config_types` 声明哪些 stage 用
`EngineStageConfig`：thinker 恒是，talker_ar 在语音拓扑里也是。
`tensor_parallel_disable_custom_all_reduce_stages = ("thinker",)` 与
`topology_gated_custom_all_reduce_stages() -> {thinker}`：Qwen thinker 的自定义
all-reduce 受拓扑门控（TP>1 且跨进程时才启用），避免同进程多 stage 场景下的
NCCL 上下文冲突。

`stage_factory_kwargs`：按阶段给工厂注入额外 kwargs——encoders 不传设备
（**设备选择推迟到 worker**），thinker 传 `speech_enabled`，talker_ar 传
`speech_enabled + feedback_enabled`，code2wav 传 `enable_cuda_graph =
current_platform.enable_code2wav_graph()`。

---

### 2. 七个阶段的逐字声明

#### 2.1 preprocessing（`config.py:41-64`）

- 文本拓扑：`next = [有输入的 encoder..., mm_aggregate]`，路由
  `resolve_preprocessing_next_stages`；语音拓扑：`next = [encoders..., thinker]`，
  有音频输出时再追加 `talker_ar`（`resolve_preprocessing_next_stages_speech`）。
- 注意：**路由函数返回的是"逻辑上有出边的目标"**，真正发不发 payload 还要经
  project_payload；`_encoder_stages_with_model_inputs` 保证纯文本请求不会发往 encoder。
- `FactoryArgs(max_seq_len=8192)`：preprocessor 据此做 prompt 长度预检
  （`preprocessor.validate_prompt_seq_len`：`prompt + max_new_tokens >= max_seq_len` 直接拒）。

#### 2.2 image_encoder / audio_encoder（`config.py:67-92`）

两者通过 `_encoder_join_edges(speech_enabled)` 共享出边定义：

- 文本：`next="mm_aggregate"`，投影 `project_encoder_to_mm_aggregate`；
- 语音：`next=[thinker, talker_ar]`，路由 `resolve_encoder_next_stages`
  （**不要音频输出时只发 thinker**），投影两张：join 用
  `project_encoder_to_mm_aggregate`，talker 用 `project_encoder_to_talker_ar`。

audio_encoder 特有的两个开关：
`factory.enable_layer_cuda_graph=True`（层间图，04 篇）与
`disable_direct_cuda_ipc_payload=True`。

#### 2.3 mm_aggregate（仅文本拓扑，`config.py:95-108`）

`wait_for=[preprocessing, image_encoder, audio_encoder]` +
`resolve_mm_aggregate_wait_sources`（动态收窄）+ `merge_for_thinker` +
`next="thinker"`。它是个纯聚合 + 转发的 CPU/GPU 轻阶段（工厂 `_identity`）。

#### 2.4 thinker（`config.py:131-158`，EngineStageConfig）

```python
factory=FactoryArgs(max_seq_len=8192, enable_async_decode=True[, speech_enabled=True])
next="decode"
stream_to=["talker_ar", "decode"] if speech_enabled else ["decode"]
route_fn=resolve_thinker_next_stages          # 恒返回 "decode"
stream_done_to_fn=resolve_thinker_stream_done_targets   # 要音频→[talker_ar, decode]
project_payload={"decode": project_thinker_to_decode}
# 语音模式还挂 join 三件套（wait_for/wait_for_fn/merge_fn）
```

注释明确 async decode 默认开，`--thinker.factory.enable_async_decode false` 可关。

#### 2.5 decode（`config.py:161-169`）

`terminal=True` + `can_accept_stream_before_payload=True`，工厂
`create_decode_executor` → `StreamingDetokenizeScheduler`。它同时接收：
payload（thinker 的 next）+ 每 token 流 + stream_done 信号，三者缺一不可
（10 篇详述三者如何对齐）。

#### 2.6 talker_ar（`config.py:171-182`，EngineStageConfig）

```python
wait_for=[preprocessing, image_encoder, audio_encoder]   # 与 thinker 相同的 join
merge_fn=request_builders.merge_for_talker               # 融合 + 早提交投影
factory=FactoryArgs(max_seq_len=32768, enable_partial_start=..., partial_start_min_chunks=5)
next="code2wav"; stream_to=["code2wav"]
project_payload={"code2wav": project_talker_to_code2wav}
can_accept_stream_before_payload=True
```

**`max_seq_len=32768` 的血泪注释必须记住**（`config.py:175-179`）：
max_seq_len 必须 > `talker_max_new_tokens(4096) + prefill`，否则 req_to_token_pool 越界
直接打崩 talker_ar；且从 8192 涨到 32768 是因为 V1 talker prefill 会把整个 thinker
prompt 重放为投影 embedding——30 帧视频 prompt 约 22K 位置，8192 溢出后表现为
FusedAddRMSNorm 的非法内存访问（不是 Python 异常，是 CUDA 级崩溃）。

#### 2.7 code2wav（`config.py:185-192`）

`gpu_memory_fraction=0.02`（声码器很小）+ `terminal=True` +
`can_accept_stream_before_payload=True`，工厂
`components.code2wav_scheduler.create_code2wav_scheduler`。

---

### 3. 三份配置的差异矩阵

| 维度 | text | speech（分散） | speech-colocated |
|------|------|----------------|------------------|
| 阶段数 | 6（含 mm_aggregate） | 7（无 mm_aggregate） | 7 |
| thinker GPU | 0 | 0 | 0 |
| talker_ar GPU | — | 1 | 0 |
| code2wav GPU | — | **thinker_gpu(0)** | 0 |
| partial start | — | True, min_chunks=5 | **False** |
| encoder_mem_reserve（thinker 默认） | 0.05 | 0.05 | 走 total_gpu_memory_fraction 契约 |
| env_defaults | DEEPGEMM=0 | DEEPGEMM=0 | + OMP_NUM_THREADS=8, TOKENIZERS_PARALLELISM=false |
| 副本/TP | 任意 | 任意 | **禁副本、禁 AR 阶段 TP** |

细节澄清：code2wav 默认绑 `thinker_gpu`（`_speech_stages(gpu=thinker_gpu)`）而非 talker GPU，
因为它只消费 talker 的小码流、输出归 Coordinator，放 GPU0 便于与 decode 链路同卡。

---

### 4. placement.py：Qwen 专属拓扑合法性

通用规划器算出 `StagePlacementPlan` 后，`Qwen3OmniPlacementPolicy.validate` 再做五道闸：

1. **禁止 code_predictor 独立成阶段**：它是 talker_ar 内部组件（08 篇），拆出去拓扑不成立；
2. **语音拓扑必须是恰好七阶段**：`_validate_speech_topology` 对缺失/多余阶段给出精确 diff；
3. **非 colocated 时 thinker 与 talker_ar 不得同卡**：
   除非双方都无副本且 tp_size==1（此时同卡无 NCCL 冲突风险）；
   否则 `gpu_ids` 有交集即报错——自定义 all-reduce 的 communicator 无法在
   两个独立 engine 间共享一张卡的通道。
4. **colocated 四禁**：
   - 禁 process 副本（`replica_instances` 中长度 >1 即错）；
   - 禁 thinker/talker_ar TP（Phase 1 colocation 不支持）；
   - 五个 GPU 阶段必须**各持恰好 1 个 GPU id 且全体同卡**；
   - 五个阶段**必须显式给 `gpu_memory_fraction`**（内存契约的前提）。
5. **AR 阶段内存一致性**：若 engine.mem_fraction_static 显式给出，必须与
   `gpu_memory_fraction` 一致（±1e-3），否则报"conflicting colocated memory contracts"。

---

### 5. 内存契约：`_apply_colocated_ar_memory_contract`（stages.py:118-160）

这是共置部署的核心算法，逻辑分三支：

```
输入：overrides(dict，即将变成 SGLang server_args)、stage_name、
      total_gpu_memory_fraction(该阶段总预算)、encoder_mem_reserve(给编码器留的份额)

1) total 为 None → 不做共置契约（分散模式），mem_fraction_static 是否显式原样记录。
2) overrides 已显式 mem_fraction_static：
     - 若还传了 encoder_mem_reserve → ValueError（二选一，拒绝歧义）；
     - 若 |explicit - total| > 1e-3 → ValueError（与 placement 第 5 闸呼应）；
     - 否则钉死 explicit，reserve 视为 0。
3) 否则推导：effective = total - encoder_mem_reserve
     - reserve 必须 ∈ [0,1)；
     - effective < 0.1 → ValueError（安全下限，防止把 KV cache 挤没了）；
     - overrides["mem_fraction_static"] = round(effective, 3)。
```

为什么需要 `encoder_mem_reserve`？SGLang 启动时按 `mem_fraction_static` 一次性吃掉
"权重 + KV cache"的静态预算。共置模式下同一张卡还要住图像塔/音频塔/声码器，
若 thinker 把整卡吃光，后启动的 encoder 进程直接 OOM。默认 0.05
（`create_sglang_thinker_executor_from_config(encoder_mem_reserve=0.05)`），
且仅在 `total_gpu_memory_fraction is not None` 且未显式钉死时生效；
分散模式（total=None）则直接对 server_args 调
`apply_encoder_mem_reserve`（`sglang_backend`），语义相同但作用于单引擎。

thinker 与 talker 的启动日志都打了 `sglang_ar_startup/started`，含
pre/post load 的可用显存与进程显存——这是排查共置 OOM 的第一现场。

---

### 6. 环境默认值与 FP8

- `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`（`config.py:28-33`）：Qwen AR 阶段在就绪后才可能
  首次命中某些 FP8 dense 形状；若开启全量预编译，这个 miss 会变成一次漫长的
  "就绪后编译会话"。用 import 期环境变量关掉它（FIXME 注释承认这是权宜之计，
  等 SGLang 提供有界/预就绪编译策略后替换）。
- 共置追加 `OMP_NUM_THREADS=8`：共置 worker 会拉起 7 个 stage 进程，任其按整机核数
  建 OpenMP 池会在多 worker 同节点时把 CPU 打爆；`TOKENIZERS_PARALLELISM=false`
  则因为预处理每个调度步只处理一条 prompt，Rayon 大池纯属争抢。
- **FP8 量化归一**在 `model_worker.py:_configure_backend_policy`：
  `_apply_omni_quantization_adapters`（stage 本地 checkpoint 命名归一）→
  `_apply_model_worker_backend_common_policy` → 平台策略
  `current_platform.apply_model_worker_backend_policy`。AR 阶段（thinker/talker）的
  FP8 走原生路径；`stages.py` 里 `get_weight_preprocessor(root_config,
  fp8_scale_inverted=True)` 处理 talker 权重的 scale 反转兼容。

---

### 7. 启动装配链（一图流）

```
PipelineConfig(Variants[name])
  └─ placement policy 校验 → StagePlacementPlan（进程规划）
      └─ stage_workers：按 process 分组建进程
          └─ 每 stage：import factory_path → 工厂(...factory, **stage_factory_kwargs)
              ├─ SimpleScheduler 工厂 → 直接可跑
              ├─ OmniScheduler 工厂（thinker/talker_ar）
              │    ├─ build_generation_batch_overrides(...)（mixed chunk 等）
              │    ├─ _apply_colocated_ar_memory_contract(...)（内存契约）
              │    ├─ build_sglang_server_args → validate_generation_batch_policy
              │    └─ create_{thinker,talker}_scheduler（bootstrap.py，06/07 篇）
              └─ Code2Wav 工厂：enable_cuda_graph 且 total_gpu_memory_fraction is None
                   → 直接 ValueError（图缓冲必须有明确预算，09 篇）
```

两个"迟到失败"设计值得点出：
- code2wav 的图预算检查在工厂里就失败，而不是等首次请求；
- thinker 的 `validate_generation_batch_policy` 在 server_args 装配后立刻校验
  mixed-chunk/chunked-prefill 的组合可行性，把配置错误挡在加载权重之前。

---

### 8. 小结

- 拓扑 = 数据；阶段 = 数据；连边、join、投影、流边全是数据——这让"改部署"
  从改代码变成改配置，也让 placement 校验成为可能。
- 共置模式的三件事必须同时成立才安全：单卡绑定、显式内存预算、禁副本/TP；
  内存契约函数是唯一的预算推导点。
- 所有"魔法数字"（8192/32768/0.05/0.02/8/0）在源码注释里都有失败案例背书，
  读配置时务必连注释一起读。


---

## 04 · 预处理与多模态编码器：Preprocessor、双塔、批量与缓存

> 主角：`components/preprocessor.py`（728 行）、`components/image_encoder.py`（202 行）、
> `components/audio_encoder.py`（251 行）、`components/audio_layer_graph.py`（273 行）、
> `stages.py` 的批量/缓存部分（1216 行的一半）。

---

### 1. Preprocessor：CPU 上的"请求整形"

`Qwen3OmniPreprocessor` 在 `preprocessing` 阶段以 `compute_fn = await preprocessor(payload)`
运行（异步：媒体解码/下载可让出）。输出写进 `Qwen3OmniPipelineState`：

| 字段 | 内容 |
|------|------|
| `prompt` | `PromptInputs{input_ids, attention_mask, prompt_text}` |
| `mm_inputs` | 每模态元数据：`image_grid_thw` / `feature_attention_mask`+`audio_feature_lengths` / `video_grid_thw`+`video_second_per_grid`+`use_audio_in_video` |
| `encoder_inputs` | `{"image_encoder": {...张量+cache_key+_active}, "audio_encoder": {...}}` |
| `stream_state` | 贯穿全流的流状态（如 AIV 切分游标） |

#### 1.1 关键机制

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

### 2. Image Encoder：Vision Tower + PatchEmbed 优化

#### 2.1 PatchEmbed：Conv3d → Linear（image_encoder.py:26-79）

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

#### 2.2 forward 输出契约

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

### 3. Audio Encoder：打包 + 段共享 + 层图

#### 3.1 特征打包（audio_encoder.py:44-60）

HF 音频塔吃 `[mel, sum(len)]` 的**拼接**布局而非 batch 布局。
`pack_padded_audio_features` 走快路径的前提是 mask 是**前缀 mask**：
`torch.equal(mask, steps < lengths)` 成立时直接
`torch.cat([row[:, :length] ...], dim=-1)`；否则回退到布尔索引的 gather 路径
（注释点明：内部有洞就必须 gather）。

#### 3.2 `_SegmentSplits`：消除 32 次 D2H 同步

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

#### 3.3 层间 CUDA 图（audio_layer_graph.py）

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

### 4. 编码器批量：同批融合与去重（stages.py）

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

### 5. Encoder 结果如何流向下游

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

### 6. 小结与踩坑清单

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


---

## 05 · 多模态融合与 M-RoPE：merge_for_thinker、pad 值哈希、三轴位置编码

> 主角：`models/qwen3_omni/merge.py`（293 行）、`models/qwen3_omni/mrope_positions.py`（331 行）、
> `request_builders.py` 中 `build_sglang_thinker_request` 的 pad 值与 mm_positions 部分。

---

### 1. merge_for_thinker：把三路 payload 折成一份数学输入

#### 1.1 融合算法（merge.py:35-63）

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

#### 1.2 build_thinker_inputs：键的组装规则（merge.py:66-155）

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

#### 1.3 media_cache_keys：pad 值哈希的种子

```python
media_cache_keys["image"] = f"image:{image_ck}"
media_cache_keys["video"] = f"video:{image_ck}"   # 图像与视频共用 encoder cache_key！
media_cache_keys["audio"] = f"audio:{audio_ck}"
```

注释（Xuesong）点破关键：**图像与视频共享同一个 encoder cache_key**（同一塔产出），
如果不加前缀，两者的 pad 值哈希会碰撞（见下节），radix 前缀会互相污染。

---

### 2. pad 值替换：radix cache 的多模态安全阀（request_builders.py:305-360）

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

### 3. M-RoPE：三轴位置编码的向量化实现

#### 3.1 语义

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

#### 3.2 为什么是 numpy 而不是 torch（mrope_positions.py:70-110）

`_linear_pos_ids / _vision_pos_ids / _vision_t_index / _merge_audio_in_video` 全是 numpy，
且注释强调 **FP32 求值顺序必须与 sglang 参考实现逐位一致**：

```python
# (t * second_per_grid) * pps：left-assoc；先算 sec*pps 会改变 FP32 舍入
return (t * np.float32(second_per_grid)) * np.float32(position_id_per_seconds)
```

"bit-identical to sglang HF port"是函数 docstring 的第一句话。位置编码差 1 个 ulp
可能让 attention 的 rotary 相位整体漂移，输出分布可感知地变化——
这里工程上选择了"宁可用 numpy 循环也要逐位一致"。

#### 3.3 主循环结构（get_rope_index_qwen3_omni_vectorized:139-296）

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

#### 3.4 talker 的线性捷径（mrope_positions.py:20-48）

`linear_mrope_positions(seq_len)`：`arange` 广播三轴 + delta=0。
`talker_can_use_linear_mrope` 判定可用性：**无 grid，或（有 grid 但 input_ids 里
没有 vision_start / audio_start）**。注释（guozhihao）解释了为什么有 grid 还可能线性：
talker decode 的位置 = `seq_len + delta - 1`，只要不会发出多模态段，
arange+0 与完整 M-RoPE 数学等价；一旦存在多模态段就必须走全量计算
（否则位置分叉，#1149 Part B）。talker 的 prefill 重放 thinker prompt 时
（07 篇），这个判定让纯文本+音频输入免掉整段 CPU 网格计算。

---

### 4. decode_events：文本流的增量语义（merge.py:214-293）

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

### 5. 小结

1. merge 是"**减法**"的艺术：折入 thinker_inputs、清空 encoder_outs、
   削减 mm_inputs——每一次投影都在给 IPC 减肥。
2. pad 值哈希把"多模态前缀共享"这个正确性问题转化为**确定性的 id 重写**，
   并用 `_omni_mm_positions` 把代价一次性预付。
3. M-RoPE 实现的底线是**逐位对齐参考实现**：numpy、固定求值顺序、
   AIV 归并的 tie-break 都精确到注释。
4. delta 的存在让 prefill/decode 位置无缝衔接；线性捷径的守卫条件
   （无多模态段）必须与 decode 位置的数学定义严格对应。

下一篇（06）：这份精心构造的请求如何进入 SGLang 引擎并被逐层注入。


---

## 06 · Thinker 引擎与多模态嵌入注入：从 StagePayload 到 SGLang 前向

> 主角：`model_runner/model_worker.py`、`model_runner/base.py`（钩子契约）、
> `model_runner/thinker_model_runner.py`（490 行）、`model_runner/_hidden_capture.py`、
> `models/qwen3_omni/bootstrap.py`（267 行）、`model_runner/prefill_inputs.py`、
> `models/qwen3_omni/thinker_model_runner.py`（prefill sidecar 适配）。

---

### 1. ModelWorker：把"子模型"伪装成完整 LLM

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

### 2. bootstrap.py：两个 scheduler 的装配差异

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

### 3. ModelRunner 钩子契约（base.py）

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

### 4. 多模态嵌入注入（thinker_model_runner.py:77-320）

这是 thinker 的灵魂函数 `_inject_multimodal_embeds`。SGLang 的 prefill 是
**chunked** 的（8192/块），一次前向只覆盖 prompt 的一个区间
`[prefix, prefix+extend_len)`。注入的任务：把该区间内的占位符 token 换成
**对应模态 embedding 的连续切片**。

#### 4.1 算法骨架

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

#### 4.2 scatter 与前向分派

```python
if ds_embeds is None:
    forward_batch.input_embeds = input_embeds      # 普通路径：SGLang 自己处理
    return None
return self._forward_with_omni_embeds(forward_batch, input_embeds, ds_embeds, vis_masks)
```

deepstack 残差是**层间注入**（特定 decoder 层的输出上加视觉残差），
`ForwardBatch` 没有这个概念，所以必须走 custom forward（`_forward_with_omni_embeds`）。
没有 deepstack 时只设置 `forward_batch.input_embeds`，让 SGLang 原生路径接管。

#### 4.3 capture hidden 与 CUDA graph 的兼容

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

### 5. prefill sidecar：Qwen3OmniThinkerModelRunner（models/…/thinker_model_runner.py）

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

### 6. 流输出的生产：make_thinker_stream_output_builder

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

### 7. result_adapter：thinker 输出的归一化

`apply_thinker_result`（request_builders.py:479-505）把 SGLang 的 req_data 变回
`ThinkerOutput{output_ids, step, is_final=True, extra_model_outputs, finish_reason?,
weight_version?, output_token_logprobs?}`，同时写进 `state.thinker_out` 与
`state.engine_outputs["thinker"]`。终态 payload 沿 `next="decode"` 经
`project_thinker_to_decode`（清输入与 extra）到达 decode 阶段——10 篇会看到
decode 如何用"流里已经发过的 token"与"payload 里的终态"双通道对齐。

---

### 8. 小结

1. arch override 是"**用别人的引擎跑我的子模型**"的全部秘密；
2. 注入 = 全量查表 + 每 chunk scatter，游标协议（plan/validate/reconstruct）
   保证 chunked prefill + radix 复用 + 多模态三者的正确性交集；
3. deepstack 迫使部分 prefill 走 custom forward，这是无法用
   `forward_batch.input_embeds` 表达的层间注入；
4. 隐藏态捕获用静态 buffer + pre-hook，与 CUDA graph 共生而非对抗；
5. sidecar 的价值取向：**资格从严，失败即回退**——图加速永远不能买走正确性。

下一篇（07）：这些每步隐藏态如何变成 talker 的输入流，以及部分启动的完整机制。


---

## 07 · Thinker-Talker 流式耦合与部分启动：TalkerPrefillBuilder、PendingTextTensorQueue

> 主角：`components/talker_prefill.py`（407 行）、`components/talker_input.py`（273 行）、
> `pending_text_queue.py`（151 行）、`request_builders.py` 的 `_build_talker_request_data`、
> `talker_scheduler.py` 的 partial start 策略（02 篇已述）。

---

### 1. 问题定义：talker 是一条"未完成的请求"

Thinker 每 token 流出两个东西：token_id（→decode）与隐藏态（→talker_ar）。
talker 要在 thinker **还在生成时**就开始出声（TTFA 优化），这带来三个矛盾：

1. talker 的 prefill 需要完整 prompt 投影，但 thinker prompt 的 assistant 部分
   还没生成完；
2. talker 的 decode 需要"下一个文本行"，但文本一行行地在流里后到；
3. talker 的 KV cache 一旦建好就不能变，但 prompt 是增量到达的。

本仓库的解法是三件事的组合：

- **部分启动**：攒够 N 个流块就构建 prefill（N=5，硬下限 3）；
- **prompt 重建**：prefill 时不重放 thinker 的计算，而是**从 checkpoint 直接捞
  embedding 重建投影**（便宜且与 thinker 无关）；
- **未来文本行 FIFO**：prompt 里放不下/还没生成的文本行进
  `PendingTextTensorQueue`，decode 逐行消费。

---

### 2. 流块的到达与暂存

`OmniScheduler._on_stream_chunk`（02 篇）把 thinker 的流块递给
`stream_chunk_handler = prefill_builder.append_text_chunk`。在此之前，
框架层已经把"payload 未到时先到的流块"攒进 Stage 的 StreamQueue 与
scheduler 的 `_pending_stream_ingress`。payload（thinker join 后经
`merge_for_talker` 的早提交投影）到达后，`prefetched_chunks` 与
`prefetched_stream_done` 两个字段成为 talker request_builder 的直接输入
（`request_builders._build_talker_request_data:272-280`）：

```python
thinker_chunks = list(payload.prefetched_chunks)
thinker_done = bool(payload.prefetched_stream_done)
if not thinker_chunks:
    raise RuntimeError("talker request_builder requires prefetched thinker chunks; "
                       "check the partial-start readiness policy or upstream wiring")
```

即：**走到 request_builder 时流块已经由调度策略担保至少 `partial_start_min_chunks` 个**
（02 篇的 `_is_request_build_ready`），这里的 raise 只是最后一道防线。

---

### 3. TalkerPrefillBuilder.build_prompt_prefill 全流程

#### 3.1 重建 prompt 三态（`_reconstruct_prompt_states`）

```python
prompt_ids = state.prompt["input_ids"]            # [seq] long, CPU
prompt_embed  = self._load_prompt_token_embeddings(prompt_ids)   # [seq, thinker_hidden]
prompt_hidden = prompt_embed.clone()
# 多模态行覆写：
merge_prompt_modality(prompt_ids, prompt_embed, prompt_hidden,
                      token_id=audio_token_id, features=model_inputs["audio_embeds"])
# 同样处理 image / video
```

`merge_prompt_modality`：`mask = prompt_ids == token_id`，然后
`prompt_embed[mask] = feature_tensor`（搬设备/对 dtype），**`prompt_hidden[mask] = 0.0`**。
为什么 hidden 置零？因为投影是双轨的（见 §3.4）：

- 文本行 → `text_projection(embedding)`
- 多模态行 → `hidden_projection(thinker_hidden_state)`

而 talker prefill 阶段拿不到 thinker 的"prompt 侧隐藏态"（那是 prefill 中间量，
没有持久化），HF 的做法是：prompt 侧多模态行用 encoder embedding 的
`hidden_projection` 近似。把 hidden 置零等于声明"这一行将在投影时走 hidden 轨"。
talker 模型的 `prepare_input_embeds`（08 篇）用同一个 mask 分派两条投影。

#### 3.2 从 safetensors 直接捞 embedding（talker_prefill.py:20-59）

```python
_THINKER_EMBED_CANDIDATE_KEYS = ("thinker.model.embed_tokens.weight", "model.embed_tokens.weight")

def load_thinker_embedding_rows(model_path, row_ids):
    shard_path, tensor_name = _resolve_embed_source(model_path)     # 索引文件 → 分片定位，进程内缓存
    handle = _EMBED_HANDLE_CACHE[model_path]                        # safe_open 句柄复用
    tensor_slice = handle.get_slice(tensor_name)
    rows = [tensor_slice[row_id] for row_id in row_ids]             # 逐行 mmap 读取
```

关键设计：**不开整个 embedding 矩阵**（30B 模型的 embedding 是 GB 级），
只 mmap 读需要的行。`_load_prompt_token_embeddings` 再加一层
`_thinker_embed_cache: dict[int, Tensor]`（token_id → 行），
用 `torch.unique(return_inverse=True)` 去重查表后 `index_select` 还原顺序。
prompt 高频 token（im_start/system/user 等特殊 token）几乎全部命中缓存。

#### 3.3 chat template 分段（talker_input.segment_chat_template）

按 `<|im_start|>`（151644）切开输入，取其后一个 token 判角色
（system=8948 / user=872 / assistant=77091），产出
`[{"role", "start", "end"}]`（**含 im_start 本身**，与 HF 一致）。
系统段被跳过（talker 不需要 system 文本），多轮里**只有最后一个 assistant 段**
参与 assistant 构建（历史 assistant 轮次属于"user 侧"上下文，走 user_part）。

`build_prefill_input` 还有一个 off-by-one 修正（注释）：assistant 段末尾的
`<|im_end|>` 要剥掉——HF 的 `thinker_embed` 从不包含 EOS 的隐藏态
（`generate()` 在产出 EOS 前就停了），不剥会让 future_text_rows 整体偏移一行。

#### 3.4 user_part：双轨投影

```python
result[multimodal_mask] = hidden_projection(thinker_hidden[multimodal_mask])
result[~multimodal_mask] = text_projection(thinker_embed[~multimodal_mask])
```

多模态行此时 hidden=0（被覆写过），所以实际是 `hidden_projection(0 + embedding)`？
——不是：`merge_prompt_modality` 里 `prompt_hidden[mask] = 0.0` 后，
`_reconstruct_prompt_states` 又没有再往 hidden 里放东西，
因此多模态行真正喂给 `hidden_projection` 的是**零向量**。
零向量经 MLP（含 bias）得到的是"投影偏置"，随后与 assistant 侧布局相加。
这与 HF 的对齐方式一致：talker 侧真正携带多模态信息的是
`hidden_projection` 的输出作为"该行的特征"这一约定，
而 prompt 侧多模态信息其实由 **codec 侧 embedding 与后续 audio 特征行**
承担（语音合成关注的是 assistant 文本与说话人，prompt 图像/音频只提供语气上下文）。
读这段代码时不要试图在 prompt 多模态行里找"视觉信息"——它被刻意置零了。

#### 3.5 assistant_part：9 行布局（talker_input.build_assistant_part）

这是与 HF `_get_talker_assistant_parts` 逐位对齐的部分：

```
文本轨（text_projection 后，[9, hidden]）：
  [0:3]   assistant 前 3 个 token 的投影
  [3:7]   4 × tts_pad_embed                       ← 占位，稳定布局
  [7]     tts_bos_embed                            ← 语音开始符
  [8]     assistant 第 4 个 token 的投影            ← 还没生成时放零（后来经未来行队列补）
codec 轨（[9, hidden]）：
  [0:3]   3 × zeros
  [3:9]   codec_embedding([nothink, think_bos, think_eos, speaker_id, pad, codec_bos])
最终：input_embeds = text_hidden + codec_hidden
      input_ids   = 9 × tts_pad_token_id（占位 id，位置追踪用）
future_text_rows = projected[4:] + tts_eos_embed       ← 关键！
```

三个理解要点：

1. **为什么是 9 行固定布局**：talker 需要在 prefill 里就看到"语音启动序列"
   （nothink/think 边界/说话人/pad/codec_bos），HF 的布局把 codec 特殊 embedding
   与文本前 3 token 叠加，形成"文本轨承载语义、codec 轨承载声学身份"的加性混合。
2. **`future_text_rows = projected[4:] + tts_eos`**：从第 5 个 assistant token 起的
   所有投影 + 语音 EOS 行，构成"未来文本行队列"的**初始内容**——第一个 decode 步
   消费 `projected[4]`（正好接在 `[8]` 位置的"第 4 个 token"之后）。
3. **partial start 下的不完整**：`projected.shape[0] <= 3` 时第 4 token 槽放零
   （注释：初始 talker 请求可能在 thinker 发出 4 个 token 前构建，槽位保持稳定，
   由未来行队列后续更新）。`include_assistant_eos=False`（thinker 未完）时
   future_text_rows 去掉最后的 tts_eos 行——EOS 只有 thinker 真正结束后才该出现。

#### 3.6 增量与终止（append_text_chunk / mark_thinker_done）

```python
def append_text_chunk(self, req_data, chunk):
    if req_data.thinker_chunks_done: return
    if chunk.metadata.get("token_id") == im_end_token_id: return   # 会话边界不入队
    pending_text_queue.append(self.project_assistant_chunk(chunk))
    # project_assistant_chunk：有 token_id → 捞 embedding 行再 text_projection；
    #                          无 token_id → 直接用 chunk.data（已是隐藏态）投影

def mark_thinker_done(self, req_data):
    req_data.thinker_chunks_done = True
    pending_text_queue.append(req_data.tts_eos_embed)   # 补上被扣掉的 EOS 行
```

流块若带 token_id（thinker 侧 metadata 总是带），投影路径与 prefill 完全一致
（同源 embedding + 同一 text_projection），保证 prompt 内外的文本行分布一致。

---

### 4. PendingTextTensorQueue：设备端 FIFO 的精确设计

（`pending_text_queue.py` 全文 151 行，值得整篇读）

#### 4.1 数据结构

```python
@dataclass(slots=True)
class PendingTextTensorQueue:
    rows: torch.Tensor | None    # 当前头块
    cursor: int                  # 头块内的读位置
    _chunks: deque[torch.Tensor] # 后续整块
    _pending_rows: int           # 缓存的剩余行数（O(1) len）
```

- `append_rows`：行规格检查（1D→reshape(1,-1)；2D 要求 hidden 维非空），
  隐藏维必须与既有头块一致，然后 `rows.to(device=..., dtype=...)` **就地转到设备**
  并压入 `_chunks`（不拼接！避免 O(n) 拷贝）。
- `popleft`：读 `self.rows[self.cursor]`，cursor+1；**头块耗尽时
  `self.rows = self._chunks.popleft()`**——升块零拷贝，旧头块交给 GC。
- `__getitem__(0)` 走 O(1) 捷径（注释：talker decode 只 peek 第一行，
  热路径绝不 consolidate）；任意下标才 `torch.cat([remaining, *chunks])[idx]`。
- `coerce_pending_text_queue`：None/Queue/Tensor/Iterable 四态归一，
  Queue 会 `copy()`（跨请求共享必须复制，防串扰）。

#### 4.2 为什么必须设备端

decode 每步要做 `feedback + text` 的逐行相加（08 篇），若 text 行在 CPU，
每步一次 H2D + 同步。设备端 FIFO 让"peek 下一文本行"变成纯 GPU 读。
这与 talker 模型里 `pending_feedback_queue`（同为设备张量 deque）对称——
两条 FIFO 的行最终在 `_combine_feedback_with_next_text` 里相加
（`row + next_text`），都在 GPU 上完成。

---

### 5. 组装成 SGLang 请求（`_build_talker_request_data`）

```python
sampling_cfg = _resolve_talker_sampling_config(params)
#   max_new_tokens=talker_max_new_tokens(4096), temperature=0.9, top_k=50,
#   top_p=1.0, repetition_penalty=1.05, codec_eos_id=model.config.codec_eos_token_id,
#   suppress_tokens=[codec_vocab-1024 .. codec_vocab) 除 eos 外全部抑制]
#   —— 抑制区间：codec 词表尾部 1024 个是非语音保留位，除 EOS 外必须封死
if sampling_cfg["seed"] is None:
    sampling_cfg["seed"] = xxhash.xxh64_intdigest(request_id) & 0x7FFFFFFF
    # 请求级确定性种子；PyTorch 采样种子必须落在正 int32（MAX_INT32_POSITIVE）

prompt_prefill = prefill_builder.build_prompt_prefill(payload, thinker_chunks, thinker_done=...)
req_data = build_sglang_talker_request(
    thinker_hidden_states=torch.empty(0),        # 空占位：真实嵌入走 talker_input_embeds
    talker_input_embeds=prompt_prefill["input_embeds"],
    talker_input_ids=prompt_prefill["input_ids"],
    input_embeds_are_projected=True,             # 已在 talker 空间，forward 不再投影
    pending_text_queue=pending_text_queue,
    tts_pad_embed=prompt_prefill["tts_pad_embed"],   # 文本行耗尽后的填充行
    thinker_chunks_done=thinker_done,
    talker_model_inputs=prompt_prefill["prompt_model_inputs"],  # M-RoPE 判定用
    seed=...)
req_data.tts_eos_embed = prompt_prefill["tts_eos_embed"]         # mark_thinker_done 要用
```

`build_sglang_talker_request`（request_builders.py:518-680）里还有两件与 M-RoPE 相关的事：

- `talker_can_use_linear_mrope(ids, mm_model_inputs, thinker_config)` →
  可用则 `linear_mrope_positions`，否则全量 CPU 计算（05 篇 §3.4）；
- `talker_multimodal_mask`：assistant/重放行的多模态标记进
  `omni_model_inputs["talker_multimodal_mask"]`，供 forward 的双轨投影（08 篇）。

采样注释里的硬核警告必须复述：**pytorch 采样后端是为了绕开 SGLang 上游缺陷**——
`Sampler.forward` 不把 seed 传给 flashinfer，CUDA graph 下捕获的 RNG 变成
"启动时决定"，约 5% 的 prompt 触发退化循环（#408）。上游修复前不可回退。

---

### 6. 时序总图

```
thinker 每 token：
  OutgoingMessage(stream, hidden, target=talker_ar)
        │ (Stage 路由; payload 未到时先入 StreamQueue)
        ▼
talker_ar Stage ── payload 到达(join 后) ──▶ prefetched_chunks[] ──▶ OmniScheduler
        │                                                        │ _is_request_build_ready?
        │                                                        │  (>=5 块 或 stream_done)
        │                                                        ▼
        │                                    request_builder: TalkerPrefillBuilder
        │                                        ├─ 重建 prompt 嵌入(safetensors 捞行)
        │                                        ├─ 9 行 assistant 布局
        │                                        └─ future_text_rows → PendingTextTensorQueue
        │                                                        │
        │                                          SGLang Req(prefill_input_embeds=投影行)
        ▼                                                        ▼
之后每个流块 ──append_text_chunk──▶ pending_text_queue   prefill → decode 循环(08 篇)
stream_done ──mark_thinker_done──▶ 追加 tts_eos 行
```

---

### 7. 小结

1. partial start 的本质：**把"prompt 不完整"转化为"prompt 完整 + 未来行队列"**，
   KV cache 一次建对，后续只追加 decode 行。
2. prompt 重建选择"从 checkpoint 捞 embedding"而非"缓存 thinker 中间态"：
   与 thinker 的 prefill 策略解耦，且天然支持多请求共享只读权重。
3. 9 行 assistant 布局是 HF 兼容性的锚点；`projected[4:]` 的切分位置与
   `[8]` 槽位的"第 4 token"必须严格对应，错一行整个语音节奏就乱了。
4. PendingTextTensorQueue 的零拷贝升块 + O(1) peek 是 decode 热路径的微观优化，
   但其正确性约束（hidden 维一致、设备/dtype 就地转换）同样严格。
5. EOS 行的"预扣后补"（include_assistant_eos / mark_thinker_done）保证了
   "thinker 未结束就不发语音 EOS"这一语义。

下一篇（08）：talker 的前向、code predictor 与每步解码交接的完整闭环。


---

## 08 · Talker 模型与解码交接：QwenTalkerModelRunner 的自反馈闭环

> 主角：`talker_model_runner.py`（503 行）、`components/talker.py`（1811 行，本篇取前向/
   采样/code predictor 部分）、`model_runner/base.py` 的 decode 钩子。

---

### 1. talker 前向的两种形态（talker.py forward:1235-1305）

```python
def forward(self, input_ids, positions, forward_batch, input_embeds=None,
            input_embeds_are_projected=False, ...):
    if forward_batch.forward_mode.is_extend():
        self.invalidate_decode_buffers()          # prefill 采样绕过 _sampled_token_ids，
                                                  # 快速路径必须失效（akazaakane 注释）
    if input_embeds is not None and not input_embeds_are_projected:
        input_embeds = self.prepare_input_embeds(...)      # 双轨投影（见下）
    positions = mrope_positions if self._uses_mrope else forward_batch.positions
    hidden = self.model(input_ids, positions, forward_batch, input_embeds=...)
    if extend and input_embeds is not None:
        return self._manual_extend_logits(hidden, forward_batch)   # prefill：只取末位 logits
    logits_output = self._manual_decode_logits(hidden)             # decode：直接 codec_head
    if forward_batch.forward_mode.is_decode():
        sampled = self._sample_decode_tokens(logits_output.next_token_logits, forward_batch)
        self._sampled_token_ids[:bs].copy_(sampled)                # 固定地址缓冲
        self.code_predictor_forward(sampled.unsqueeze(1), hidden.unsqueeze(1))
    return logits_output
```

注意三个"SGLang 旁路"：

1. **手写 logits**：`_manual_extend_logits` 只对每个请求的最后一个位置取
   `codec_head`（`cumsum(extend_seq_lens) - 1` 找末位），绕开通用 LogitsProcessor
   （注释：该路径在投影 prefill 的 extend batch 上会挂）；`_manual_decode_logits`
   则因为 decode 无需 gather，直接过 head。
2. **手写采样**：`_sample_decode_tokens` 自己做 repetition penalty
   （`logits>0 ? logits/p : logits*p`，作用在 `_repetition_mask` 标过的
   历史码位上）、suppress mask（`-inf`），再交 SGLang sampler 或 argmax。
3. **自产自销**：decode 步采样结果不回调度器再进 predictor，而是**在同一前向里**
   直接喂 `code_predictor_forward`，把残差码与求和 embedding 写进固定缓冲
   `_output_codes / _output_embeds`。

#### 1.1 双轨投影（prepare_input_embeds:1130-1155）

```python
if thinker_hidden_states is None or is_multimodal_mask is None:
    return self.text_projection(thinker_embeds)          # 全文本轨
if thinker_embeds is None:
    return self.hidden_projection(thinker_hidden_states) # 全隐藏轨
# 混合：mask 分派（07 篇的 mask 语义在此落地）
```

prefill 侧 deepstack（`input_deepstack_embeds` + mask）进 talker 时走的正是这条混合路径；
07 篇的 prompt 重建已把行预投影（`input_embeds_are_projected=True`），forward 跳过投影。

#### 1.2 静态缓冲与采样参数分段上载（prepare_decode_buffers:1053-1128）

`Qwen3OmniTalker.__init__` 为 `max_running_requests` 预分配了一整组固定地址缓冲：
`_repetition_mask / _suppress_mask / _repetition_penalties / _sampling_temperatures /
top_ps / top_ks / min_ps / _sampling_seeds / _output_codes / _output_embeds / …`。

`prepare_decode_buffers(requests)`（`before_decode` 钩子调用，08 §2）把每请求的
采样参数写进缓冲，有两个微优化值得复述：

- **pinned staging + 位重解释**：6×B 的 int64 CPU pinned 缓冲，
  行 0-3 用 `.view(torch.float64)` 位重解释放 float 参数（penalty/temperature/top_p/min_p），
  行 4-5 放 int（top_k/seed）——一次 `copy_(non_blocking)` + event 就把全部参数上 GPU，
  省掉 6 次 H2D。注释强调 staging 必须 `device="cpu"`：模型 `__init__` 在 CUDA 默认
  设备上下文里执行，只有 CPU 张量能 pinned。
- **快速路径 `_reuse_decode_buffers`**：同一批 rid 且各请求输出长度恰好 +1
  （= 常规 decode 连续步）时，只把上一步采出的 token 补进 repetition mask
  （`_repetition_mask[rep_rows, _sampled_token_ids[rep_rows]] = True`），其余参数不动。
  `_decode_prep_rids/_decode_prep_out_lens` 是路径有效的证据链；
  `invalidate_decode_buffers` 在 prefill 后强制失效。

---

### 2. QwenTalkerModelRunner：decode 前后的两次交接

#### 2.1 before_decode（talker_model_runner.py:57-75）

```python
def before_decode(self, forward_batch, schedule_batch, requests, *, is_lookahead=False):
    if not self._feedback_enabled: return
    if not self._requests_ready_for_decode(requests):
        raise RuntimeError("Talker decode reached model runner without ready feedback/text input")
    self.model.prepare_decode_buffers(requests)
    self._write_feedback_buffers(requests)
```

`_requests_ready_for_decode` 检查每请求的 `_data_has_next_decode_input`：
`pending_feedback_queue` 非空 **且**（`pending_text_queue` 非空 **或**
(`thinker_chunks_done` 且 `tts_pad_embed` 就位)）。02 篇的
`_is_batch_ready_to_run` 用同一谓词把不就绪的 decode 批整个推迟——
这里再断言一次属于"防御性双保险"。

`_write_feedback_buffers` 把每请求的下一步输入写进模型的
`_feedback_buffer[:bs] / _feedback_mask[:bs]`：

```python
for row_idx, sched_req in enumerate(requests):
    combined = self._take_next_decode_input_embed(sched_req=sched_req, device, dtype)
    ...
if len(rows) == batch_size:      # 稠密稳态：切片赋值，避免 per-frame 可分页索引 H2D
    feedback_buffer[:bs] = embeds_stacked; feedback_mask[:bs] = True
else:
    rows_t = torch.tensor(rows, device=...)   # 稀疏才走索引上传
    feedback_buffer[rows_t] = embeds_stacked; feedback_mask[rows_t] = True
```

#### 2.2 下一输入的合成（_combine_feedback_with_next_text:437-460）

```python
feedback = peek(pending_feedback_queue)          # 上一步 codec 求和 embedding
combined = feedback                              # 必须存在，否则整体返回 None
next_text = peek(pending_text_queue)
if next_text is None:
    if not data.thinker_chunks_done: return None  # 文本还没来且没说完 → 不可解
    next_text = data.tts_pad_embed                # 说完了 → pad 填充行
return combined + next_text                       # 相加（两个投影空间的和）
```

**`feedback + text` 的加法是 talker 的核心数学**：codec 轨的上一帧求和 embedding
加上文本轨的下一行投影，作为本步输入。这与 prefill 布局的"加性混合"完全同构
（07 篇 9 行布局就是 `text_hidden + codec_hidden`）。两条 FIFO 各弹一行
（`_take_next_decode_input_embed` 里 pop），保持严格同步。

弹出的行同时 `append` 进 `decode_input_embeds` 历史——这是为 **retract（回退重跑）**
准备的：`_generated_prefill_slice` 在 decode 请求被重新 prefill 时从历史重放
已生成的行，缺失则报错（"Cannot replay retracted talker decode tokens"），
绝不静默编造。

#### 2.3 post_decode / post_prefill：码帧出口（talker_model_runner.py:77-140）

```python
def post_decode(self, result, forward_batch, schedule_batch, requests):
    batch_size = len(requests)
    result.next_token_ids = self.model._sampled_token_ids[:batch_size].clone()
    self._stage_token_ids(result, result.next_token_ids)
    self._emit_code_chunks_and_feedback(...)
```

`_emit_code_chunks_and_feedback` 是自反馈回路的出口：

```python
codes_snap  = self.model._output_codes[:bs].detach().clone()    # [bs, num_code_groups]
embeds_snap = self.model._output_embeds[:bs].detach().clone()   # [bs, hidden]
for idx, sched_req in enumerate(requests):
    is_streaming = params.get("stream", False)
    self._outbox.put(OutgoingMessage(request_id, "stream",
        data=code_chunk, target="code2wav", metadata={"stream": is_streaming}))
    sched_req.data.pending_feedback_queue.append(feedback_row)
```

**为什么必须 clone**（wenyao 注释）：`_output_codes/_output_embeds` 是固定地址缓冲，
下一帧（可能在 CUDA graph 内）会原地覆写；快照必须是新分配，否则发给 code2wav 的
张量会被后续帧改写。这是"图内写固定地址 + 图外消费"的经典竞态。

`post_prefill` 的差异：layer0 码直接取 `result.next_token_ids`
（prefill 采样结果），talker_hidden 取 `result.logits_output.hidden_states`
（捕获层），走同样的 predictor + emit。注释强调**不要清
`data.prefill_input_embeds`**——decode retract 可能把请求重新排队再 prefill，
而 `Req.input_embeds` 是 None。

---

### 3. Code Predictor：残差 RVQ 的增量生成

#### 3.1 结构

`Qwen3OmniMoeTalkerCodePredictor` 是一个小型 decoder LM，带
`num_code_groups` 个独立的 `lm_head` 与逐组 `codec_embedding`。
talker 主干只产**第 0 组**码（layer0 code），其余 `num_code_groups - 1` 组
由 predictor 按序贪心生成（`_sample_code_predictor_token`：argmax，
注释：匹配 HF `generate(do_sample=False)`）。

#### 3.2 增量算法（_code_predictor_forward_incremental_eager:1550-1607）

对每个位置 `pos`：

```
输入序列构造（predictor_input[:, 0:2, :]）：
  [0] = talker_hidden[pos]        ← talker 主干的隐藏态
  [1] = embedding(layer0_code)    ← 主干采出的第 0 组码
result_codes[:, :, 0] = layer0_code
pos_summed = layer0_embed          ← 求和 embedding 的起点

cache_len = 0
last_hidden = predictor_forward_one_token(输入[0])   cache_len=1   ← 先吃隐藏态
last_hidden = predictor_forward_one_token(输入[1])   cache_len=2   ← 再吃 layer0 码
for group in 1 .. num_groups-1:
    logits = lm_head[group-1](last_hidden)
    code = argmax(logits)
    result_codes[:, :, group] = code
    new_embed = codec_embedding[group-1](code)
    pos_summed += new_embed                            ← 求和 embedding 累加
    if group < num_groups-1:
        last_hidden = predictor_forward_one_token(new_embed)  cache_len+=1
```

`_predictor_forward_one_token` + `_predictor_cached_self_attention` 是手写的
单 token 前向：**每层一个单回合 KV cache**（`_predictor_k_cache/_v_cache[layer_idx,
:bs, :, cache_len+1, :]`），SDPA 对 cached 前缀做非因果 attention。
KV cache 位置用 `_predictor_positions = arange(num_code_groups+1)`——
predictor 的序列长度上限就是"组数+1"，无需 paged attention。

#### 3.3 predictor 的 CUDA 图（_PredictorDecodeGraph, talker.py:52-110）

seq_len==1 且张量在 CUDA 时走图：按 `(bucket_size, code_dtype)` 捕获/复用，
bucket 集合来自 `get_decode_cuda_graph_bs(server_args)`（归一化后保证包含
max_running_requests）。replay 前 `layer0_codes[live:].zero_()` 补零填充。
捕获失败（任意异常）→ 记入 `_predictor_decode_graph_disabled`，永久回 eager。
`_can_use_predictor_decode_graph` 里有一条 `torch.cuda.is_current_stream_capturing()`
守卫——**外层 talker 图捕获期间绝不能再触发内层图捕获**。

---

### 4. MoE 细节：共享专家的统一 all-reduce（talker.py:230-293）

`Qwen3OmniMoeTalkerSparseMoeBlock` 继承 thinker 的 MoE（路由专家部分），
新增 shared expert：

```python
shared_output = self.shared_expert(x)                    # reduce_results=False！
shared_output = shared_output * sigmoid(shared_gate(x))   # 门控
routed_output = self.experts(x, topk_output)
final = routed_output + shared_output                     # TP>1 且未融合时：
if ...: final = tensor_model_parallel_all_reduce(final)   # 只做一次 all-reduce
```

要点：shared expert 的 `down_proj` `reduce_results=False`，与路由专家的
部分和相加后**统一做一次 all-reduce**——省一半通信。还有个顺序性注释：
shared 分支必须在路由专家**之前**消费原始输入，因为 fused MoE 实现会原地改写
`hidden_states`。

DecoderLayer 直接继承 thinker 层、只换 mlp（`Qwen3OmniMoeTalkerDecoderLayer`）；
`Qwen3OmniMoeTalkerTextModel` 另有一套**手工 prefill 前向**
（talker.py:700-760，`_direct_self_attention`：qkv 分裂 + qk norm + rotary + SDPA
`is_causal=True, enable_gqa=...`），用于需要精确对齐 HF 的路径。

---

### 5. 权重加载（load_weights:1740-1810）

- `talker.` 前缀剥离；`thinker./code2wav.` 前缀直接跳过（负负得正：同一份
  checkpoint 被三个引擎分别加载）；
- 三段式映射：stacked（qkv_proj/gate_up_proj 拆分）→ MoE expert 参数
  （`FusedMoE.make_expert_params_mapping`）→ 直接匹配；
- `get_weight_preprocessor(root_config, fp8_scale_inverted=True)` 处理 FP8 scale
  的方向差异。

---

### 6. 小结：一次 talker decode 步的完整时间线

```
OmniScheduler.get_next_batch_to_run
  └─ is_decode_batch_ready? (每请求 feedback+text 就绪)   否 → 回滚 KV 分配，跳过本步
        │ 是
        ▼
before_decode: prepare_decode_buffers(参数分载/复用) + _write_feedback_buffers
  └─ combined = feedback + text   （两条设备端 FIFO 各 pop 一行，写 _feedback_buffer）
        ▼
forward(hidden=..., input_embeds=None)
  ├─ codec_head → logits
  ├─ _sample_decode_tokens（repetition penalty + suppress + sampler/argmax）
  ├─ _sampled_token_ids[:bs] ← sampled          （固定缓冲）
  └─ code_predictor_forward(sampled, hidden)     （图或 eager，残差码 + 求和 embedding）
        ▼
post_decode: result.next_token_ids ← _sampled_token_ids 克隆
  └─ _emit_code_chunks_and_feedback
       ├─ outbox → code2wav (stream, codes 帧快照)
       └─ pending_feedback_queue ← embeds 帧快照   ←→ 下一轮 before_decode
```

闭环成立的三个支点：**就绪谓词**（feedback+text 同步）、**快照克隆**
（对抗固定地址覆写）、**回滚对称性**（02 篇 §3.3 的 KV 回滚）。

下一篇（09）：码帧离开 talker 后，在 code2wav 里如何变成连续的音频。


---

## 09 · Code2Wav：流式声码器、输出重叠与精确形状 CUDA 图

> 主角：`components/code2wav_scheduler.py`（1065 行）、`components/code2wav_cuda_graph.py`（723 行）、
> `scheduling/streaming_vocoder.py`（模板基类，02 篇已述契约）。

---

### 1. 阶段定位与工厂

`code2wav` 是 terminal 阶段：消费 talker 的 codec 码帧流，产出音频。
工厂 `create_code2wav_scheduler` 的默认参数：

- `stream_chunk_size=10`：每 10 帧码开一窗（24kHz 下 codec 帧率约 12.5/s，
  即约 0.8s 音频/窗；`total_upsample` 由模型给出）；
- `left_context_size=25`：滑窗携带最多 25 帧历史上下文（声码器对长程上下文敏感，
  裁剪输出时只保留新增部分）；
- `sample_rate=24000`、`codec_eos_token_id=2150`；
- `enable_batching=False` 默认关（开了才有跨请求合批/图分解）；
- `enable_output_overlap=True` 默认开；
- `enable_cuda_graph` 默认关，平台开关 `current_platform.enable_code2wav_graph()` 注入
  （03 篇 stage_factory_kwargs）；
- **硬性约束**：开图必须给 `total_gpu_memory_fraction`，否则工厂直接
  `ValueError`——图缓冲的显存预算没有边界就是隐患（09 §5）。

模型加载：HF `Qwen3OmniMoeCode2Wav._from_config(code2wav_config)` +
`load_module(prefix="code2wav.")`，`strict=False`（保留 DSP 缓冲等非权重成员）。

---

### 2. 流状态与 EOS 处理

`Code2WavStreamState`：

```python
chunks: list[Tensor]        # 已入帧（每帧 [num_quantizers] long）
emitted: int                # 已消耗帧游标
audio_parts: list[ndarray]  # 已产出的音频段（拼接即整段）
stream_enabled: bool | None # 由 latch_stream_contract 闩锁（首个流块 metadata["stream"]）
due_since: float | None     # 合批的"到期时间"
checked: int                # EOS lazy scan 的已扫描下标
pending: _PendingWindow | None  # 深度-2 重叠流水线的在途窗
```

#### 2.1 EOS 的两种模式

- **eager**（未开重叠）：`ingest` 时逐帧 `codes[0].item() == eos` 判断——每帧一次
  **主机同步**；
- **lazy scan**（`enable_output_overlap and not batching`）：ingest 无条件入队，
  `should_decode` 在攒够一窗时调 `_scan_unchecked` **批量扫描**：

```python
unchecked = chunks[start:]
if all(codes.ndim == 1 for codes in unchecked):     # 生产者恒发 1-D 帧
    heads = torch.stack([codes[0] for codes in unchecked])
    is_eos = (heads == self._codec_eos_token_id).tolist()   # 一次 GPU 求值 + 一次同步
else:
    is_eos = [codes.ndim >= 1 and codes[0].item() == ... for ...]  # 异形回退逐帧
chunks[start:] = [c for c, eos in zip(unchecked, is_eos) if not eos]
state.checked = len(chunks)
```

注释点出设计动机："逐帧检查=每帧一次 sync；批量扫描=每窗一次"。
且**先扫描再判定就绪**：若不扫描，滞留的 EOS 帧会虚增帧数，触发一个短窗
而错失图 key（graph key 按精确帧数命中）。`decode_delta(is_final=True)` 与
`on_stream_done` 也会补扫（流终止不走 should_decode）。

---

### 3. 滑窗解码：decode_delta 的精确数学

```python
start, end = state.emitted, len(state.chunks)
context = min(self._left_context_size, start)
window = torch.stack(state.chunks[start - context : end], dim=0)   # [T, Q]
codes = window.transpose(0, 1).unsqueeze(0)                        # [1, Q, T]
wav, meta = self._forward_codes(codes, graph_eligible=not is_final)
wav = wav[..., -(end - start) * self._total_upsample :]            # 裁掉上下文对应样本
```

- 窗口帧数 = `context + 新帧`；输出裁剪保证**只发新增样本**（增量流）。
- `_forward_codes`：无图 → 直接 `self._model(codes)`；有图 →
  `runner.run(codes, eligible=graph_eligible)`（final 窗永不走图——尾窗帧数
  几乎必然不在 key 集里，标记 ineligible 省一次 key_miss 计数）。
- 返回 `execution_mode / graph_key / fallback_reason` 三元组进 profiler 事件，
  这是判定"图有没有真正命中"的第一手证据。

---

### 4. 输出重叠：深度-2 的 D2H 流水线（本文件最精巧部分）

#### 4.1 槽位池

```python
@dataclass(eq=False)                      # 恒等相等：池按"这是几号槽"提问，不能比较张量
class _PinnedSlot:
    buffer: torch.Tensor                  # pinned float32，只增不缩
    event: torch.cuda.Event               # 复用，不逐窗分配

_MAX_PINNED_SLOTS = 32                    # 每流 1 槽 + 1 在途，注释给出推导
_pinned_free / _pinned_created / _pinned_retired / _pinned_quarantined
```

槽位生命周期：`_acquire_slot`（free 优先 → 未达上限新建 → **返回 None 背压**）
→ 在途（`state.pending`）→ `_flush_pending` 后 `_release_slot`；
异常时 `_retire_slot`（可重查回收）或 `_quarantine_slot`（event 不可信，**永不复用**
但仍占名额，防止容量凭空膨胀），quarantine 同时 `_pipeline_active = False`
（整体退回同步路径）。
`_reap_retired_slots` 用**非阻塞** `event.query()` 把已完成 retire 槽收回 free——
在 `_next_message` 循环里调用（wenyao 注释：query 非阻塞，不会复现它所替代的
abort 卡死问题）。

#### 4.2 流水线时序（decode_delta 内）

```
第一窗（start==0）：同步路径 —— D2H 立即 .cpu()，保证 TTFA 不被流水线伤害
后续窗（pipeline_active 且非 final）：
  slot = acquire(samples)
  ├─ None 且有 pending → flush 自己的 pending（保序！只 flush 自己的窗）
  │                      再 acquire；再 None → 老实走同步路径
  └─ 拿到 → buffer[:samples].copy_(wav.reshape(-1), non_blocking=True)
            event.record()                        ← 同一 CUDA stream 保证顺序
            flush 上一窗 pending（等待其 event → host 拷贝 → append → 返回波形）
            state.pending = _PendingWindow(slot, samples)
```

正确性三根柱子（代码注释逐一写明）：

1. **同 stream 序**："same stream is what makes this safe —— 拷贝先于任何后续
   replay 排队，借来的静态输出不可能在排空完成前被覆写"（`Code2WavRunResult`
   docstring 亦强调：图输出是借来的，必须在下一次 replay 前完成读或同 stream
   排队读）；
2. **先 launch 后 flush**：`_flush_pending` 放在本次 launch 之后，主机等待
   上一窗的时间与 GPU 计算本窗重叠；
3. **异常槽位处理**：record 成功后异常 → retire（可回收）；record 前异常 →
   quarantine（无可信完成标记，绝不回池）。

`_flush_pending` 的顺序同样讲究：`event.synchronize() → numpy().copy() →`
**之后**才 `state.pending=None; _release_slot(slot)`——任何一步都可能抛，
提前归还等于把一个"活的 D2H 目标"交给池。

---

### 5. 精确形状 CUDA 图（code2wav_cuda_graph.py）

#### 5.1 key 体系与捕获计划

`GraphKey(batch_size, frames)`。键集由调度器构造（scheduler.py:26-56）：

- 串行键：`frames ∈ {c, 2c, ..., c+left_context}` 逐步推进（c=stream_chunk_size），
  每步一窗、上下文线性增长到上限；
- 批量键：同帧长 × `batch_size ∈ {2,4,8}`（`_DECOMPOSE_SIZES=(8,4,2,1)`，
  注释：分解表、捕获键、批上限全部由这个 tuple 派生，键不可能缺失）。

`Code2WavCudaGraphRunner` 的构建策略（`_build` / `_capture_attempt`）：

1. **两层**：`batch_size==1` 是"原子层"（`_tier0_keys`）——任何失败=整体禁用，
   绝不发布残缺矩阵；`batch_size>1` 是"尽力层"（`_tier1_keys`）。
2. **共享 mempool**：所有 key 共用一个 `graph_pool_handle`，总占用趋近**最大成员**
   的峰值而非逐成员累加；代价是"池内存只能整体回收"，所以重试单位是**整轮捕获**。
3. **最大优先**：`_priority_order` 按 `(batch_size, frames)` 降序——最大图先铺
   池子的峰值块，后续小图复用。
4. **逐键预算检查**：每捕获一个 tier1 键 → synchronize → `empty_cache()`
   （注释：warmup 的 eager 激活残留会让 reserved 虚高，必须清掉再量）→
   `footprint > graph_budget` 则停手。预算 =
   `total_bytes * fraction - 已加载模型 footprint`。
5. **失败收缩**：`shrink` 回调按"剔除最大 batch 档"（首键即违规/合计违规）或
   "截断到违规位"（中间违规）收敛，最多 `_TIER1_MAX_ATTEMPTS=6`
   （backstop，注释：重试集严格递减，正常情况到不了 6）。
6. **等价性验证**：每键捕获后 `eager_output vs graph.replay()` 逐位
   `torch.equal` + 形状 + 有限性（`_verify_equivalence`），不过即 `_BuildFailure`。
7. **发布前同步**：capture/replay/等价检查都排了 CUDA 工作，
   全部 key 完成并 device synchronize 后才 `_enabled=True`。

#### 5.2 运行时

`run(codes, eligible)` 的守卫链：**PID 所有权**（跨进程使用直接 RuntimeError——
图绑定进程，spawn 后必须重建）→ enabled/eligible → 形状/dtype/设备/量化器数校验
→ key 命中 → `static_input.copy_ + graph.replay()`。
运行期 replay 异常 → `_disable_runtime`（清全部图、释放池、empty_cache、
`logger.exception`），之后恒走 eager（fallback_reason=`disabled`）。
`stats()` 输出严格 JSON 安全的构建/运行快照（attempted/published、内存账本、
fallback 计数），工厂启动时打一条 `Code2Wav CUDA graph startup stats=...`。

---

### 6. 跨请求合批（enable_batching=True 时）

三个钩子的实现（scheduler.py:820-1000）：

- **`select_step_participants`**：
  - 首窗流（`emitted==0` 且攒够 `initial_codec_chunk_frames` 或整窗）：
    同 bucket 的首窗流优先出队（注释：同 bucket ⇒ 一个 trim 标量对整批成立），
    截到 `_batch_ceiling`；
  - 稳态流：`ready >= chunk_size` 时记 `due_since`，按 bucket
    `(context, context+step_frames)` 分桶；选**最老等待时间最短的那桶**
    （`min(due, key=最老等待)`），凑满 `batch_floor` 或等满 `max_batch_wait_s`
    或 drain 模式才触发（fire_reason ∈ {first, floor, deadline, drain}）。
- **`build_step_plan`**：图对齐开启时，按该窗长**实际发布**的批大小
  （`runner.available_batch_sizes(window)`，内存压力可能缺大图）把 N 请求
  分解成 `_decompose_batch(N, sizes=(8,4,2,1))` 的子批序列；**没有可用图的
  尺寸时退 `[1]*N`**——注释：批图缺席可能因为其 eager warmup OOM，
  把该批按 eager 重试并不安全，宁可全串行图。
- **`run_step`**：逐子批 `_run_sub_batch`（同窗长断言 → stack 成
  `[B, Q, T]` → 前向 → 按请求切分 → `state.emitted=window_ends[i]`）；
  某子批失败时**只 abort 尚未 decode 的尾部参与者**
  （`participants[cursor:]`）——已 decode 的音频是欠客户端的，不能丢。
  profiler 事件记录 `sub_batch_execution` 列表，混合模式汇总为 `"mixed"`。

---

### 7. 终态与兜底

- `on_stream_done`：先 flush 在途 pending（独立消息发出，注释：不并入尾块，
  消息边界必须与同步路径一致，且必须先于基类的终态 result），
  再走基类（flush 尾窗 → 终 result → 清状态）；
- `final_result_data`：流式时只回 `{modality, sample_rate}`（客户端已收全样本）；
  非流式 `np.concatenate(audio_parts)` 打包整段（`audio_waveform_payload`）；
  **一段音频都没有 → RuntimeError**（绝不返回空音频装成功）；
- `release_stream_resources`（abort 时）：**非阻塞** retire 在途槽
  （abort 在 stage 事件循环线程上跑，不能等 CUDA）；
- `on_serving_stop`：关机时同步排干 retire 槽（此时阻塞无害）。

---

### 8. 小结

1. 帧窗数学：`window = [start-context, end)`，输出裁剪 `[-(end-start)*upsample:]`——
   上下文只影响质量不影响长度。
2. EOS lazy scan 把"每帧一次 sync"降为"每窗一次"；先扫描后判窗的顺序
   是图命中率的前提。
3. 输出重叠 = 槽位池 + 事件 + 同 stream 排序 + 保守降级（首窗同步、槽尽同步、
   异常整体退同步）。
4. CUDA 图的工程学：原子层/尽力层、共享池、最大优先、逐键预算、等价验证、
   运行期一键禁用、PID 所有权——每条都是生产事故的解药。
5. 合批的三层闸门（参与者选择→分解计划→子批执行）保证了
   "图没命中就退串行、子批失败不殃及已完成音频"。

下一篇（10）：文本链的终端——流式反分词器的边界处理。


---

## 10 · decode 阶段与流式反分词：StreamingDetokenizeScheduler

> 主角：`components/streaming_detokenizer.py`（318 行）。
> 它是文本链的 terminal 阶段，也是全仓库最清晰地展示"payload / 流 / 信号
> 三通道如何对齐"的样本。

---

### 1. 为什么 decode 不用 SimpleScheduler

`create_decode_executor`（stages.py）直接返回
`create_streaming_detokenize_scheduler(model_path)`。docstring 说明来历：
它**替代了**基于 SimpleScheduler 的一次性 decode——因为 thinker 语音模式下
每 token 都会流过来（`stream_to` 含 decode），一个只处理 `new_request` 的
调度器无法消费流。它的 inbox 契约与 Code2Wav 同构：

- `new_request`：thinker 的终态 payload（`next="decode"` 那份）；
- `stream_chunk`：`StreamItem(data=torch.tensor([token_id]), metadata={"token_id": t})`；
- `stream_done`：`resolve_thinker_stream_done_targets` 的信号。

`Stage.can_accept_stream_before_payload=True` 对此阶段是必需的：
流必然先于 payload（payload 要等 thinker 生成完）。

---

### 2. 主循环与失败隔离

```python
while self._running:
    msg = self.inbox.get(timeout=0.1)
    try:
        if msg.type == "new_request":   self._on_new_request(msg.request_id, msg.data)
        elif msg.type == "stream_chunk": self._on_stream_chunk(msg.request_id, msg.data)
        elif msg.type == "stream_done":  self._on_stream_done(msg.request_id)
    except Exception as exc:
        self.abort(msg.request_id)
        self.outbox.put(OutgoingMessage(request_id, "error", data=exc))
```

注释（重要）：异常若逃出 `start()` 会触发 `Stage._handle_scheduler_crash`，
**杀掉 decode 阶段上所有在途请求**——所以每条消息一个 try/except，
坏请求只坏自己。这是全仓库统一的调度器契约（Simple/Omni/Code2Wav 同款）。

---

### 3. 增量反分词：UTF-8 边界安全

```python
def _on_stream_chunk(self, request_id, item):
    token_id = int(item.data.item()) if hasattr(item.data, "item") else int(item.data)
    s.pending_tokens.append(token_id)
    candidate = tokenizer.decode(s.pending_tokens, skip_special_tokens=True)
    if "�" in candidate:      # U+FFFD：多字节字符被截断
        return                 # 挂起，等下一个 token 补齐
    s.pending_tokens.clear()
    if not candidate: return   # 特殊 token 全部被跳过 → 无可发
    outbox.put(OutgoingMessage(request_id, "stream", target=None,
        data={"text": candidate, "modality": "text", "stage_name": "decode"},
        metadata={"modality": "text"}))
```

细节：

- `target=None` = **terminal 流**，Stage 会直接发 Coordinator（01 篇 §6），
  客户端收到的是 `{"text": delta}`；
- `_finalize` 时若 `pending_tokens` 还有残货（例如 max_tokens 截断在多字节字符中间），
  必须再 decode 一次发出去——注释：否则流式客户端会**永久丢失**这些尾字节，
  而非流式客户端（走整段 decode）却能看到，二者不一致。

---

### 4. 三通道对齐：payload × 流 × stream_done

#### 4.1 状态机

```python
@dataclass
class _RequestState:
    pending_tokens: list[int]
    payload: StagePayload | None
    done: bool = False
```

- `_on_stream_done`（无状态行时）：`_done_seen[request_id] = None`。
  两种合法情形：**零 token 生成**（没有任何 chunk 建过状态行）与
  **迟到重复 done**（finalize 已把状态行删掉）。
- `_on_new_request`：payload 到 → 查 `_done_seen`（零 token 情形：done 立即为真）→
  非流式或已 done → 立即 `_finalize`。

#### 4.2 _done_seen 的容量治理

```python
_DONE_SEEN_MAX = 10000      # 上限：零 token 竞态 + 迟到 done 的孤儿条目
_DONE_SEEN_EVICT_TO = 5000  # 超限后 FIFO 淘汰到 5000
```

OrderedDict 头进尾出，最老的先淘汰——两个常数配一对，
避免热路径上的反复淘汰抖动。

#### 4.3 为什么 done 可以先于 payload

thinker 的 `stream_done_to_fn` 在**生成结束时**就向 decode 发 done 信号，
而终态 payload 还要走 `project_thinker_to_decode` + relay 传输。
zero-token 请求（例如被 stop 条件立即终止）就是 done 先到、payload 后到的
真实案例。`_done_seen` 把这次竞态变成"payload 到时补 finalize"。

---

### 5. 终态 result 的构建（_build_result:220-310）

终态 `OutgoingMessage(type="result", data=payload)` 的 `payload.data` 被
替换成 result dict。构建过程：

1. 从 state 取 `thinker_out`（或 `engine_outputs["thinker"]`）；
2. 调 **merge.decode_events**（05 篇 §4）生成事件列表——
   同一函数保证流式路径与终态文本一致；
3. 取最后一个 `is_final / text_final / final` 事件的 payload 作为顶层字段；
4. **流式请求删掉 `text`**（注释全文值得背）：

   > 流式客户端已经通过逐 token 流收到全文；
   > `Client.completion_stream()` 的直接消费者会拼接每个 chunk 的 "text" 字段，
   > 终态再带全文等于**输出两遍**。
   > "Mirrors the code2wav slim-final contract for audio"——
   > 音频侧（09 篇 final_result_data 的流式分支）是同一契约。

5. 非流式且无 text → `tokenizer.decode(output_ids)` 整段补上；
6. `finish_reason` / `output_token_logprobs` / `weight_version` 透传；
7. **usage**：`prompt_tokens = input_ids.numel()`（兼容张量/列表）、
   `completion_tokens = len(output_ids)`、总数——注意 decode 阶段自己算
   usage，不依赖上游。

---

### 6. abort 语义

`abort(request_id)`：删 `_state` 与 `_done_seen` 条目。迟到的流块/终态
（`request_id` 已不在册）被静默丢弃——由 Stage 层的 `_aborted` 集合先行拦截，
这里只是第二道闸。

---

### 7. 与 thinker 流输出的一处精妙协作

回忆 06 篇：thinker 的 token 流只在 `stream=True` 时发。那么非流式请求的
decode 收到什么？——只有终态 payload。此时：

- `_on_new_request`：非流式 → 立即 finalize → `decode_events` 对完整
  `output_ids` 一次性产出 `text_final` 事件 → result 带 `text`。
- 流式请求：每 token 走 `_on_stream_chunk`，终态 payload 到时
  `decode_events` 会再算一遍（全量 token）——但它的 `text` 被删了，
  只留事件元数据与 usage。

也就是说 **decode 阶段对"流式/非流式"的全部差异处理**最终归结为一行
`result.pop("text", None)`，而文本内容的一致性由 `decode_events` 单点保证。
这是"终态与流态共享一个纯函数"设计红利的直接体现。

---

### 8. 小结

1. decode 是一个**三输入状态机**：流（增量文本）、信号（done）、payload（终态）；
   `_done_seen` 治理"done 先于 payload"竞态，`pending_tokens` 治理 UTF-8 边界。
2. 失败隔离模式（per-request try/except + error outbox）与全仓库调度器契约一致。
3. 流式终态的 slim contract（去 text）与音频侧互为镜像——
   **任何"已流式发送的内容不得在终态重复"**是整个系统对客户端的统一承诺。
4. usage/finish_reason/logprobs 的透传与补算都发生在这一层，
   Coordinator 拿到的 result 已经是可直接序列化的最终形态。

下一篇（11）：支撑这一切的物理层——CommEngine 与张量传输。


---

## 11 · 通信引擎与张量传输：CommEngine、传输选择、DataRef 与 KV 转移

> 主角：`comm/engine.py`（1136 行）、`comm/router.py`（399 行）、`comm/data_ref.py`（212 行）、
> `comm/stage_io.py`（818 行）、`relay/`（cuda_ipc / shm / mooncake / nccl / nixl）。

---

### 1. 三个枚举定义了一切（data_ref.py:11-33）

```python
class TransportKind(str, Enum):
    LOCAL_OBJECT = "local_object"   # 同进程内对象直传（Python 引用）
    CUDA_IPC     = "cuda_ipc"       # 同机跨进程 GPU 句柄零拷贝
    SHM          = "shm"            # 共享内存
    MOONCAKE     = "mooncake"       # 跨机传输引擎

class DataKind(str, Enum):
    STAGE_PAYLOAD / STREAM_CHUNK / STREAM_METADATA_TENSOR / KV_PAGES /
    WEIGHT_BUCKET / MOE_EXPERT_PAYLOAD

class DataLayout(str, Enum):
    PACKED_TENSORS / RAW_TENSOR / PAGED / BUCKETED / SCATTER
```

`DataRef` = `{object_id, transport, kind, layout, tensor_meta...}`——
**控制面上只传这张"取货单"**，真货走对应 relay。`TensorMeta`
（dtype/shape/stride 等）让接收端可以预分配。

---

### 2. CommRouter：传输选择的决策树（router.py:130-297）

```python
def _intra_node_transport(self, target):
    if self.is_local_object(target):          return LOCAL_OBJECT   # 同进程
    if self.can_use_direct_cuda_ipc(target):  return CUDA_IPC       # 同机 GPU
    return SHM                                                       # 同机 CPU 兜底
```

`can_use_direct_cuda_ipc` 的判定要素：双方都是 GPU stage、同节点、
**双方没有把 direct IPC 显式关掉**（StageConfig 的
`disable_direct_cuda_ipc_payload=True`，mm_aggregate 与 audio_encoder 挂了它）、
以及 `_cuda_ipc_peer_available(target)`（对端进程可达性缓存，失败一次即降级
并打 `comm_ipc_fallback` 警告——router.py:66-129 的降级注释是篇好短文）。

流与 payload 的选择还可以**按数据本身**细分：`outbound_stream(target, data)`
会看张量是否在 GPU 上（CPU 小张量不值得走 IPC 句柄），`outbound_payload`
看 payload 内容形态。也就是说**同一对 stage 的两条边可能用不同传输**。

---

### 3. CommEngine：发送/接收的执行体（engine.py:96-…）

#### 3.1 发送路径

```python
async def send_payload(self, target, request_id, payload, stream_targets_for_request=None):
    transport = self.router.outbound_payload(target, payload)
    relay = self.relay(transport)
    object_id = await self.write_payload(relay, request_id, payload)   # 放货
    await self._publish_data_ready(target, request_id, data_ref=...)   # 发取货单
```

- 发送任务按 `(target, transport)` 进**每键串行队列**
  （`_send_queue_for / _run_send_worker`），保证同一目标的写入有序；
- `_PayloadSendJob / _StreamSendJob` 是 msgspec.Struct（frozen）——
  队列里只放轻量任务描述；
- `_watch_pending / _arm_pending / _fail_pending`：等待 Ack 的超时看护，
  超时/失败把 pending 转异常，向上层传播；
- `ack_transfer(ack)`：Ack 回来后释放本端持有（引用计数/内存配额）。
  **Ack 驱动的配额**是背压的来源：接收端不读不 Ack，发送端持有不释放。

#### 3.2 接收路径

Stage 收到 `DataReadyMessage` 后：`CommEngine.read_data(relay, request_id, data_ref)`
→ relay 层反序列化 → Stage `_send_data_ack`。01 篇 §1 的三分支
（direct IPC / inline / relay）就是在这一层前分流，绕过 relay 的两个捷径
由 `stage_io` 的谓词与反序列化器支持：
`deserialize_direct_cuda_ipc_payload`（CUDA IPC 句柄直接 open 成 GPU 张量）与
`deserialize_inline_stream_chunk`（小张量随消息走，省一次取货）。

#### 3.3 KV 转移（KV_PAGES）

engine.py 的后半是**离散式部署的 KV cache 迁移**：`register_kv_pool` 注册本端池、
`prepare_kv_receive` 协商对端池布局（`KVPoolLayout.compatible_with`，
proto/kv_transfer.py）、`send_kv_pages` 按 `req_to_token` 页表搬运。
thinker→talker 虽然在当前 Qwen 拓扑中不共享 KV（talker 重放投影 prompt），
但这条通道是"同一引擎族共享 KV"愿景的基础设施
（Ming-Omni 等模型使用）。

---

### 4. Relay 实现（relay/）

| 文件 | 机制 | 适用 |
|------|------|------|
| `cuda_ipc.py` | `torch.multiprocessing` CUDA 事件/句柄共享，零拷贝 | 同机 GPU↔GPU，最高带宽最低 CPU |
| `shm.py` | POSIX 共享内存 + 元数据头 | 同机 CPU 段/回退 |
| `mooncake.py` | Mooncake 传输引擎 | 跨机 RDMA |
| `nccl.py` / `nixl.py` | 集合通信/传输库适配 | 特定部署形态 |

所有 relay 实现同一 `Relay` 基类契约：`write(object_id, data) / read(data_ref) /
close`。CommEngine 只面向契约，传输差异被完全封装——这正是 Stage/调度器
代码里看不到任何传输痕迹的原因。

---

### 5. 端到端一次 payload 传输（时间线）

```
thinker stage 线程(result 路由)              code2wav side
─────────────────────────────               ─────────────
get_next → target="code2wav"
project_talker_to_code2wav(payload)   →  空 latch
CommEngine.send_payload
  router: 同机 GPU → CUDA_IPC（若未禁用）
  relay.write(object_id)                 ← GPU 句柄注册
  控制面 publish DataReadyMessage  ──────────▶  Stage._on_data_ready
                                              stage_io 谓词: direct IPC?
                                              deserialize → payload
                                              control_plane.send DataAck ──▶
  ack_transfer → 释放持有
```

流块的差异只在 `chunk_id` 字段与 `read_stream_chunk`，以及 01 篇说的
"捷径分支必须自补 comm_stream_read 事件"。

---

### 6. 与 Qwen3-Omni 性能相关的三个事实

1. **payload 投影 = 传输裁剪**：05/06 篇的 `project_*` 家族之所以存在，
   是因为每条边的传输量直接决定 IPC 延迟。最极端的是
   `project_talker_to_code2wav` 返回空 data——"code2wav 该处理此请求"这条
   控制信息用一个空 payload 传递，几 KB 的声学码反而走流通道逐帧到达。
2. **`disable_direct_cuda_ipc_payload` 的两处使用都是保守正确**：
   audio_encoder（批量切分需要可复制普通内存）与 mm_aggregate（多源合并）。
   速度捷径让位于合并正确性。
3. **流的 inline 捷径**：token_id 包成 `torch.tensor([t])`（06 篇）除了
   "流传输只收张量"外，也让小张量可以走 inline 路径避免建 relay 对象。

---

### 7. 小结

- 通信栈的分层：**Stage（业务路由）→ CommEngine（配额/队列/看护）→
  CommRouter（传输选择）→ Relay（字节搬运）**。每层只依赖下一层的窄契约。
- DataRef"取货单"模式让控制面保持轻量、数据面可插拔；
  Ack 驱动的持有释放是系统背压的物理来源。
- 传输选择的三个维度：进程位置（同进程/同机/跨机）、数据位置（GPU/CPU）、
  数据形态（payload/流块/KV 页）。
- 对 Qwen3-Omni 而言：thinker→talker 的隐藏态流与 talker→code2wav 的码流
  都是小而频繁的 GPU 张量，CUDA IPC + inline 是它们的默认快车道；
  唯二的禁用点（audio_encoder、mm_aggregate）都是为多源合并的正确性让路。

至此 11 篇全部完成。建议回到 00 篇的模块地图，把每篇的机制在图上再走一遍。


---
