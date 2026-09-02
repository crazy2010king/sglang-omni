# Qwen3-Omni 总览与代码导读（整体框架篇）

> 本系列文档基于源码逐行核对，是对 `docs/code_doc/qwen3-omni.md` 的深化与纠偏。
> 本篇是"先有整体框架"的总纲：先建立全局拓扑与请求生命周期的心智模型，
> 后续 11 篇再逐一深入到最底层的实现细节。

---

## 0. 阅读本系列的方式与节奏

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

## 1. Qwen3-Omni 模型本体：三个子模型

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

## 2. 两种部署拓扑：6 阶段文本 / 7 阶段语音

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

## 3. 请求的完整生命周期（以语音请求为例）

下面按时间顺序走一遍一次"图+文→语音回复"请求。每一步标注执行者与源码锚点。

### 阶段 A：提交与预处理（CPU）

1. 客户端 → Coordinator（`pipeline/coordinator.py`）：`SubmitMessage` 送达 entry 阶段
   `preprocessing`。Coordinator 记录 `RequestInfo`，按 `terminal_stages_fn`
   （`request_builders.resolve_terminal_stages`：有音频输出时为 `[decode, code2wav]` 双终态）决定何时算完成。
2. `Qwen3OmniPreprocessor`（`components/preprocessor.py`）在 CPU 完成：
   chat template 渲染 → HF `Qwen3OmniMoeProcessor` 产出 `input_ids` + `mm_inputs` + 各模态
   `encoder_inputs`（pixel_values / input_features / 网格张量），并为媒体算 **cache_key**
   （xxhash）。产出写进 `Qwen3OmniPipelineState`，再通过 `project_payload` 按目标阶段"投影"
   成 3~4 份轻量 payload，分别发往 `image_encoder`、`audio_encoder`、`thinker`（+`talker_ar`）。
   动态路由由 `resolve_preprocessing_next_stages_speech` 决定：**没有图的请求不会空跑 encoder**。

### 阶段 B：双塔编码（GPU）

3. `image_encoder`（`components/image_encoder.py`）：Vision Tower + **Conv3d→Linear 的 PatchEmbed
   替换**（kernel==stride 时数学等价，7~15× 提速）。支持跨请求批量（`stages.py:_batch_image_encoder_payloads`）
   与同批去重、4GiB CPU LRU 输出缓存。
4. `audio_encoder`（`components/audio_encoder.py`）：Audio Tower + **层间 CUDA 图**
   （`_GraphedLayerStack` 让 32 层循环只走一次 Python，`_SegmentSplits` 消除每层一次的 D2H 同步）。
   同样支持批量与去重。

两塔完成后各自经 `project_encoder_to_mm_aggregate` / `project_encoder_to_talker_ar`
（后者剔除 deepstack 多尺度特征，talker 用不到）把 `encoder_outs` 单键 payload 发出。

### 阶段 C：Thinker（GPU，SGLang 引擎）

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

### 阶段 D：Talker（GPU，第二条 AR 引擎）

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

### 阶段 E：Code2Wav（GPU，流式声码器）

12. `Code2WavScheduler`（`components/code2wav_scheduler.py`）攒帧：每 `stream_chunk_size=10` 帧
    开一个滑窗（左上下文最多 `left_context_size=25` 帧），裁掉上下文对应的样本后输出增量音频；
    EOS 采用 **lazy scan**（批量扫描代替逐帧 `.item()` 同步）。
13. 开启 output-overlap 时：首个窗口同步 D2H（保证 TTFA），之后每个窗口 `non_blocking`
    拷进 **pinned 槽位** + event 记录，GPU 继续算下一窗，主机线程等上一窗 event 后再发——
    深度 2 的 D2H 流水线。
14. CUDA 图（`code2wav_cuda_graph.py`）：`GraphKey(batch_size, frames)` 精确形状捕获；
    batch=1 是"原子层"，失败即整体回滚；batch>1 尽力而为、逐键预算检查、失败缩容重试。

### 阶段 F：文本终端

15. `decode` stage 的 `StreamingDetokenizeScheduler` 逐 token 反分词：不完整 UTF-8
    （出现 U+FFFD）会挂起待下一个 token 补齐；`stream_done` 与零 token 竞态用
    `_done_seen` OrderedDict 兜底；流式终态 result 刻意**不带全文**防客户端重复拼接。

---

## 4. 核心抽象清单（读任何一篇前先背下来）

### 4.1 数据面三件套

- **`StagePayload`**（`proto/stage.py`）：跨阶段的信封 = `request_id + request + data`。
  `data` 是 **plain dict**（msgpack 可序列化），语义由各模型的 `PipelineState` 类解释。
- **`Qwen3OmniPipelineState`**（`payload_types.py:38`）：`payload.data` 的类型化视图，
  字段为 `raw_inputs / prompt / mm_inputs / encoder_inputs / encoder_outs / thinker_inputs /
  thinker_out / engine_outputs / stream_state`。`from_dict/to_dict` 负责与 dict 互转。
  **关键设计**：每个阶段的投影函数（`project_*`）本质是构造一个只含目标阶段所需字段的
  新 State——这是"IPC 开销裁剪"的全部手段。
- **`IncomingMessage / OutgoingMessage`**（`scheduling/messages.py`）：调度器 inbox/outbox 里的
  消息，`type ∈ {new_request, stream_chunk, stream_done, result, error}`。

### 4.2 控制面 / 数据面分离

- **控制面**（ZMQ）：`SubmitMessage`、`DataReadyMessage`（"数据在哪"的通知）、
  `DataAckMessage`、`CompleteMessage`、`AbortMessage`、`AdminMessage`。只有控制消息走
  Coordinator/Stage 的 `control_plane`。
- **数据面**（relay）：真实张量走 `CommEngine` 选定的传输
  （`TransportKind = local_object | cuda_ipc | shm | mooncake`，见 `comm/data_ref.py:11`），
  控制面只传 `DataRef`（object_id + transport + 元信息）。收发双方用
  DataReady→read→Ack 握手，配额由 ack 驱动。

### 4.3 Stage = IO 壳 + Scheduler

`Stage`（`pipeline/stage/runtime.py:100`）不做任何模型计算，只做五件事：
控制面收发、数据面读写、输入聚合（`InputHandler` + `wait_for`/`wait_for_fn`）、
流块路由（`StreamQueue` + `can_accept_stream_before_payload`）、把计算投递给 scheduler 的
inbox 并排空 outbox。Scheduler 分三类契约兼容的实现：

| 实现 | 用途 | 线程模型 |
|------|------|----------|
| `SimpleScheduler` | preprocessing / encoders（可选批量） | 专线程跑事件循环，`compute_fn` |
| `OmniScheduler`（及 `QwenTalkerScheduler`） | thinker / talker_ar（SGLang AR 引擎） | 专线程跑 `_event_loop_{async_decode,overlap,normal}` |
| `Code2WavScheduler` / `StreamingDetokenizeScheduler` | code2wav / decode（流式终端） | 专线程 + inbox/outbox |

### 4.4 一个必须内化的不变量

**"payload 是快照，流是通道"。** `wait_for + merge` 聚合的是各源最终 payload；
而 token / hidden / codec 帧 / 音频样本这些**时序数据全部走 stream 通道**
（`stream_to`、`OutgoingMessage(type="stream", target=...)`）。
这也解释了为什么 `project_talker_to_code2wav` 只留一个空 latch（`request_builders.py:164-170`）：
"code2wav 该处理这个请求了"这条事实走 payload，真正的码张量走 stream。

---

## 5. 模块地图（带文件锚点）

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

## 6. 阅读源码的推荐路径

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

## 7. 本系列后续篇目

| 篇 | 文件 | 核心问题 |
|----|------|----------|
| 01 | `Qwen3-Omni-01-流水线框架与Stage运行时.md` | Stage 如何聚合多源输入、路由流、处理崩溃 |
| 02 | `Qwen3-Omni-02-调度器契约与双调度器.md` | Simple/Omni 双调度器与 talker 就绪策略 |
| 03 | `Qwen3-Omni-03-配置系统与部署拓扑.md` | 三份配置、placement 校验、内存契约 |
| 04 | `Qwen3-Omni-04-预处理与多模态编码器.md` | CPU 预处理、双塔、批量与缓存 |
| 05 | `Qwen3-Omni-05-多模态融合与M-RoPE.md` | merge、pad 值哈希、三轴位置 |
| 06 | `Qwen3-Omni-06-Thinker引擎与嵌入注入.md` | SGLang Req 装配、注入、隐藏态捕获 |
| 07 | `Qwen3-Omni-07-Thinker-Talker流式耦合与部分启动.md` | partial start、prefill 重建、FIFO |
| 08 | `Qwen3-Omni-08-Talker模型与解码交接.md` | talker 前向、code predictor、自反馈 |
| 09 | `Qwen3-Omni-09-Code2Wav流式声码器与CUDA图.md` | 滑窗、重叠 D2H、精确形状图 |
| 10 | `Qwen3-Omni-10-decode阶段与流式反分词.md` | UTF-8 边界、竞态、终态契约 |
| 11 | `Qwen3-Omni-11-通信引擎与张量传输.md` | 传输选择、握手、KV 转移 |
