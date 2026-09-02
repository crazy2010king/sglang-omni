# SGLang-Omni code_doc 索引

> 本目录包含多个文档系列，索引分节列出，按主题检索。

## 一、Qwen3-Omni 源码深度解析系列（12 篇）

> 基于仓库源码逐行核对写成，是 `qwen3-omni.md` 的深化版。
> 阅读顺序即编号顺序：先总览（00）建立全局框架，再按 01→11 逐一深入。
> 每篇独立成文，但篇末都给出与前后篇的接口。

## 篇目

| # | 文件 | 一句话 |
|---|------|--------|
| 00 | [Qwen3-Omni-00-总览与代码导读.md](Qwen3-Omni-00-总览与代码导读.md) | 拓扑全景、请求生命周期、核心抽象、模块地图与阅读路径 |
| 01 | [Qwen3-Omni-01-流水线框架与Stage运行时.md](Qwen3-Omni-01-流水线框架与Stage运行时.md) | Stage IO 壳、多源 join、流路由、故障域 |
| 02 | [Qwen3-Omni-02-调度器契约与双调度器.md](Qwen3-Omni-02-调度器契约与双调度器.md) | inbox/outbox 契约、Simple/Omni 双调度器、partial start 调度策略 |
| 03 | [Qwen3-Omni-03-配置系统与部署拓扑.md](Qwen3-Omni-03-配置系统与部署拓扑.md) | 三份 PipelineConfig、placement 五道闸、共置内存契约 |
| 04 | [Qwen3-Omni-04-预处理与多模态编码器.md](Qwen3-Omni-04-预处理与多模态编码器.md) | Preprocessor、PatchEmbed Conv3d→Linear、音频层图、批量与缓存 |
| 05 | [Qwen3-Omni-05-多模态融合与M-RoPE.md](Qwen3-Omni-05-多模态融合与M-RoPE.md) | merge_for_thinker、pad 值哈希、逐位对齐的三轴位置编码 |
| 06 | [Qwen3-Omni-06-Thinker引擎与嵌入注入.md](Qwen3-Omni-06-Thinker引擎与嵌入注入.md) | arch override、chunked 注入游标协议、静态隐藏态捕获、prefill sidecar |
| 07 | [Qwen3-Omni-07-Thinker-Talker流式耦合与部分启动.md](Qwen3-Omni-07-Thinker-Talker流式耦合与部分启动.md) | TalkerPrefillBuilder、9 行 assistant 布局、PendingTextTensorQueue |
| 08 | [Qwen3-Omni-08-Talker模型与解码交接.md](Qwen3-Omni-08-Talker模型与解码交接.md) | before_decode 交接、feedback+text 合成、code predictor、自反馈闭环 |
| 09 | [Qwen3-Omni-09-Code2Wav流式声码器与CUDA图.md](Qwen3-Omni-09-Code2Wav流式声码器与CUDA图.md) | 滑窗数学、EOS lazy scan、深度-2 D2H 重叠、精确形状 CUDA 图 |
| 10 | [Qwen3-Omni-10-decode阶段与流式反分词.md](Qwen3-Omni-10-decode阶段与流式反分词.md) | UTF-8 边界、三通道对齐、slim 终态契约 |
| 11 | [Qwen3-Omni-11-通信引擎与张量传输.md](Qwen3-Omni-11-通信引擎与张量传输.md) | CommEngine、传输选择决策树、DataRef 取货单、KV 转移 |

## 与原文档（qwen3-omni.md）的对照与修正

- **阶段数**：原文档称"8 阶段语音流水线"；当前源码（`models/qwen3_omni/config.py`）
  为 **7 阶段**（preprocessing / image_encoder / audio_encoder / thinker / decode /
  talker_ar / code2wav），且语音拓扑**没有独立 mm_aggregate 阶段**——
  join 已内联进 thinker 与 talker_ar（详见 00 篇 §2、01 篇 §2）。
- **MIN_PARTIAL_START_CHUNKS**：常量确为 3（`config.py:21`），但它是
  `QwenTalkerScheduler` 构造时的**硬下限**；分散式部署实际传
  `partial_start_min_chunks=5`，共置式直接关闭 partial start（详见 02/03 篇）。
- **talker max_seq_len**：现值 **32768**（非 8192），原因是 V1 talker prefill
  重放完整 thinker prompt（30 帧视频约 22K 位置），8192 会触发 req_to_token 越界与
  FusedAddRMSNorm 非法内存访问（详见 03 篇 §2.6）。
- **code2wav / 输出重叠 / CUDA 图**：原文档概述了方向，本系列 09 篇给出
  槽位池、事件时序、tier0/tier1 捕获策略与逐键预算的完整实现细节。

## 核心源码入口速查

```
拓扑与配置     sglang_omni/models/qwen3_omni/config.py
阶段工厂       sglang_omni/models/qwen3_omni/stages.py
请求构造/投影   sglang_omni/models/qwen3_omni/request_builders.py
Stage 壳       sglang_omni/pipeline/stage/runtime.py
Coordinator    sglang_omni/pipeline/coordinator.py
AR 调度器      sglang_omni/scheduling/omni_scheduler.py
Thinker 装配   sglang_omni/models/qwen3_omni/bootstrap.py
Talker 交接    sglang_omni/models/qwen3_omni/talker_model_runner.py
声码器         sglang_omni/models/qwen3_omni/components/code2wav_scheduler.py
通信           sglang_omni/comm/{engine,router,data_ref}.py
```

## 二、同目录其他文档系列（并行会话产出，未在本系列范围）

- **Qwen3-TTS 系列（8 篇）**：`Qwen3-TTS-01-总体架构与全景导览.md` 至
  `Qwen3-TTS-08-与Qwen3-Omni深度对比.md`——qwen3_tts 模型的独立深度解析，
  第 08 篇与本系列互为参照。
- **跨模型框架**：`核心框架逐层深解.md`（core.md 的硬核展开，覆盖框架层）、
  `core.md`、`serve命令解析与运行全链路详解.md`、`api_cli.md`、
  `sglang-omni与qwen_tts的关系.md`。
- **原始摘要**：`qwen3-omni.md`、`qwen3-tts.md`（英文概览，本系列为其中文深化版）。
