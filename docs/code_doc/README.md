# SGLang-Omni code_doc 索引

> 本目录包含多个文档系列，索引分节列出，按主题检索。

## 一、Qwen3-Omni 源码深度解析（全一册）

**[Qwen3-Omni-源码深度解析全集.md](Qwen3-Omni-源码深度解析全集.md)**

> 基于仓库源码逐行核对写成，是 `qwen3-omni.md` 的深化版。
> 全集共 12 章（原 12 篇系列合并而成），章节即原篇目编号，
> 正文交叉引用（如"05 篇 §3.4"）与第 00 章末的导航表均按章号检索。
> 阅读顺序即章节顺序：先第 00 章建立全局框架，再按 01→11 逐一深入。

| 章 | 主题 |
|----|------|
| 00 | 总览与代码导读：拓扑全景、请求生命周期、核心抽象、模块地图 |
| 01 | 流水线框架与 Stage 运行时：多源 join、流路由、故障域 |
| 02 | 调度器契约与双调度器：Simple/Omni 双调度器、partial start 策略 |
| 03 | 配置系统与部署拓扑：三份 PipelineConfig、placement 五道闸、内存契约 |
| 04 | 预处理与多模态编码器：PatchEmbed Conv3d→Linear、音频层图、批量与缓存 |
| 05 | 多模态融合与 M-RoPE：merge_for_thinker、pad 值哈希、三轴位置编码 |
| 06 | Thinker 引擎与嵌入注入：arch override、注入游标协议、隐藏态捕获 |
| 07 | Thinker-Talker 流式耦合与部分启动：TalkerPrefillBuilder、PendingTextTensorQueue |
| 08 | Talker 模型与解码交接：feedback+text 合成、code predictor、自反馈闭环 |
| 09 | Code2Wav 流式声码器与 CUDA 图：滑窗、深度-2 D2H 重叠、精确形状图 |
| 10 | decode 阶段与流式反分词：UTF-8 边界、三通道对齐、slim 终态契约 |
| 11 | 通信引擎与张量传输：传输选择决策树、DataRef、KV 转移 |

### 与原文档（qwen3-omni.md）的对照与修正

- **阶段数**：原文档称"8 阶段语音流水线"；当前源码（`models/qwen3_omni/config.py`）
  为 **7 阶段**（preprocessing / image_encoder / audio_encoder / thinker / decode /
  talker_ar / code2wav），且语音拓扑**没有独立 mm_aggregate 阶段**——
  join 已内联进 thinker 与 talker_ar（详见全集第 00/01 章）。
- **MIN_PARTIAL_START_CHUNKS**：常量确为 3（`config.py:21`），但它是
  `QwenTalkerScheduler` 构造时的**硬下限**；分散式部署实际传
  `partial_start_min_chunks=5`，共置式直接关闭 partial start（详见第 02/03 章）。
- **talker max_seq_len**：现值 **32768**（非 8192），原因是 V1 talker prefill
  重放完整 thinker prompt（30 帧视频约 22K 位置），8192 会触发 req_to_token 越界与
  FusedAddRMSNorm 非法内存访问（详见第 03 章）。
- **code2wav / 输出重叠 / CUDA 图**：原文档概述了方向，第 09 章给出
  槽位池、事件时序、tier0/tier1 捕获策略与逐键预算的完整实现细节。

### 核心源码入口速查

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

## 二、同目录其他文档

- **[Qwen3-TTS-深度解析.md](Qwen3-TTS-深度解析.md)**——qwen3_tts 模型的独立深度解析
  （原 8 篇系列合并版），其对比篇与本全集互为参照。
- **跨模型框架**：`核心框架逐层深解.md`（core.md 的硬核展开，覆盖框架层）、
  `core.md`、`serve命令解析与运行全链路详解.md`、`api_cli.md`、
  `sglang-omni与qwen_tts的关系.md`。
- **原始摘要**：`qwen3-omni.md`、`qwen3-tts.md`（英文概览，上两册分别为其中文深化版）。
