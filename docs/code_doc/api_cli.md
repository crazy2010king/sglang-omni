API 服务器和 CLI
相关源文件
SGLang-Omni 提供了一个高性能的服务层，旨在处理多阶段多模态模型的复杂性。该层包括用于服务器编排的统一命令行界面 (CLI)、实现与 OpenAI 兼容的 REST 端点的 FastAPI 应用程序，以及用于低延迟交互式音频的实时 WebSocket API。

该服务架构的设计以计算为中心，其中 Omni 模型的不同阶段（例如，思考者、说话者、编解码器）可以分布在多个进程或 GPU 上，同时对最终用户呈现为一个统一的 API。

系统概述
下图展示了 CLI 入口点、API 服务器和底层管道协调逻辑之间的关系。

服务器入口点架构















资料来源： 
sglang_omni/cli/serve.py
69-72
 
sglang_omni/serve/launcher.py
48
 
sglang_omni/serve/openai_api.py
175-193
 
sglang_omni/serve/openai_api.py
5-18

命令行界面和服务器启动器
与 SGLang-Omni 交互的主要方式是通过sgl-omni serveCLI。CLI 负责解析模型配置、应用特定于硬件的覆盖设置以及管理多进程服务器的生命周期。

配置解析：根据模型路径或提供的 YAML 文件ConfigManager确定合适的配置。PipelineConfig
sglang_omni/cli/serve.py
14
硬件调优：CLI 提供标志来覆盖默认的阶段配置，例如--thinker-tp-size或--talker-tp-size 
sglang_omni/cli/serve.py
227-234
共置控制：该--colocate标志允许在同一进程中运行多个阶段（例如 Thinker 和 Talker），以减少进程间通信 (IPC) 开销，尤其针对特定目标。Qwen3OmniSpeechColocatedPipelineConfig 
sglang_omni/cli/serve.py
28
 
sglang_omni/cli/serve.py
90-109
优化开关：用户可以async通过启用解码模式--decode-mode，该模式可为 Higgs TTS、MOSS-TTS-Local 和 Fun-ASR 等受支持的模型启用一步前瞻调度。
sglang_omni/cli/serve.py
29-47
 
tests/unit_test/higgs_tts/test_cli_decode_mode.py
37-50
预填充合并：CLI 支持--prefill-coalesce-requests批量--prefill-coalesce-wait-ms处理多个预填充请求，从而提高 ASR 和 TTS 引擎各阶段的吞吐量。
sglang_omni/cli/serve.py
48-64
端口发现：启动器包含用于查找可用端口或强制执行严格端口绑定的逻辑。SGLANG_OMNI_STRICT_PORT 
sglang_omni/serve/launcher.py
93-116
有关详细信息，请参阅CLI 和服务器启动器。

兼容 OpenAI 的 REST API
该服务器为多个 OpenAI API 端点提供了即插即用的替代方案，并通过ChatCompletionRequest架构扩展以支持多模态输入和多阶段采样参数。
sglang_omni/serve/protocol.py
41-89

端点	目的	主要特点
/v1/chat/completions	多模态聊天	支持modalities、、、、和覆盖audios​images​videosstage_sampling
sglang_omni/serve/protocol.py
64-89
/v1/audio/speech	文本转语音	支持标准TTS合成、通过参考进行语音克隆以及原始PCM流媒体传输
sglang_omni/serve/openai_api.py命令行界面和服务器启动器
相关源文件
CLIsgl-omni serve是启动 SGLang-Omni 推理栈的主要入口点。它负责处理模型架构解析、声明式配置合并以及多进程管道生命周期的编排。

CLI架构和入口点
CLI 使用以下方式构建typer并注册：sglang_omni.cli:app 
sglang_omni/cli/__init__.py
7-14
该命令定义于sglang_omni.cli.serve:serve 
sglang_omni/cli/serve.py
440
它支持两种主要操作模式：

自动解析：通过提供--model-path（HF ID 或本地路径）启动，系统会推断出正确的PipelineConfig类别。
sglang_omni/config/manager.py
107-124
显式配置：通过 YAML/JSON 文件启动--config，从而可以对阶段放置和运行时参数进行精细控制。
sglang_omni/config/manager.py
127-145
自然语言到代码实体映射：CLI流程
下图将 CLI 概念映射到负责处理它们的底层代码实体。

资料来源： 
sglang_omni/cli/serve.py
440
 
sglang_omni/config/manager.py
34-43
 
sglang_omni/models/registry.py
135

配置管理
这ConfigManager是决定模型应如何执行的中央权威机构。
sglang_omni/config/manager.py
34-40

分辨率逻辑
模型路径解析：如果--model-path提供，resolve_config_cls_for_model_path则检查元数据以architectures使用AutoConfig 
sglang_omni/config/manager.py
16-24
它还支持 Mistral 格式配置的备用逻辑（params.json）和原始 JSON 解析。
sglang_omni/config/manager.py
25-31
 
sglang_omni/utils/hf.py
54-129
注册表查找：将架构字符串（例如， ）与存储实现映射的Qwen3OmniForCausalLM注册表进行匹配。PIPELINE_CONFIG_REGISTRYPipelineConfig
sglang_omni/config/manager.py
31
 
sglang_omni/models/registry.py
114-120
变体选择：类似这样的标志--text-only会触发变体查找（例如，选择多模态配置的“文本”变体）
sglang_omni/config/manager.py
112-123
CLI 标志覆盖
CLI 提供高级标志，这些标志会被转换为PipelineConfig对象内部的深层覆盖：

并行性：--thinker-tp-size并--thinker-gpus覆盖 Thinker 阶段的默认张量并行性和设备放置。
sglang_omni/cli/serve.py
365-375
内存管理：--mem-fraction-static设置全局 KV 缓存预算，同时--thinker-mem-fraction-static提供逐阶段的精细化管理。
sglang_omni/cli/serve.py
187-201
共置：此--colocate标志强制所有阶段共享单个 GPU，需要特定的Qwen3OmniSpeechColocatedPipelineConfigGPU 类和显式的内存预算验证。
sglang_omni/cli/serve.py
89-109
解码模式：该--decode-mode标志（异步/同步）允许通过enable_async_decode在阶段中修改来强制模型进入特定的执行模式factory_args。Higgs TTS、MOSS-TTS-Local 和 Fun-ASR 等模型主要支持此功能。
sglang_omni/cli/serve.py
29-46
 
sglang_omni/cli/serve.py
153-179
资料来源： 
sglang_omni/config/manager.py
106-124
 
sglang_omni/cli/serve.py
187-201
 
sglang_omni/utils/hf.py
26-51

服务器启动器生命周期
配置完成后，launch_server启动运行时环境
sglang_omni/serve/launcher.py
228-275

生命周期序列
端口发现：如果请求的端口被占用，_find_available_port则自动绑定到一个随机的空闲端口并记录警告。
sglang_omni/serve/launcher.py
93-115
管道初始化MultiProcessPipelineRunner：使用最终的运行器实例化A PipelineConfig，该运行器会生成各个阶段进程。
sglang_omni/serve/launcher.py
241-243
应用创建create_app：通过挂载与 OpenAI 兼容的 REST 端点、用于 TTS 和实时 API 的 WebSocket 路由以及可选的分析器端点来创建 FastAPI 应用程序。
sglang_omni/serve/launcher.py
244-250
故障监控：启动器进入一个循环，持续监控子进程的运行状况runner.check_live()。如果任何阶段的进程崩溃，整个服务器都会关闭，以防止部分进程故障。
sglang_omni/serve/launcher.py
260-273
实现图：启动服务器


资料来源： 
sglang_omni/serve/launcher.py
228-275
 
sglang_omni/serve/launcher.py
93-115

高级配置逻辑
部分启动政策
启动器支持partial-startQwen3-Omni 的一项策略，允许说话者阶段在思考者阶段完成其全部响应之前开始生成音频。此功能可通过以下方式切换：--talker-partial-start 
sglang_omni/cli/serve.py
205-220
CLI 利用create_talker_ar_executor_from_config此行为
sglang_omni/cli/serve.py
64-66

GPU 兼容性和自定义全规避
启动器会自动检测硬件环境是否支持自定义的 all-reduce 内核。如果使用多个 GPU 进行张量并行 (TP) 但缺少 P2P​​ 访问，系统会自动将disable_custom_all_reduce: True相关参数注入 SGLang 服务器，以防止程序挂起。
sglang_omni/cli/serve.py
276-300
此检查使用should_disable_custom_all_reduce_for_gpus 
sglang_omni/cli/serve.py
23

记忆分数分辨率
KV缓存的内存分配是通过层次结构来实现的：

显式阶段覆盖：--thinker-mem-fraction-static 
sglang_omni/cli/serve.py
187-201
全局静态覆盖：--mem-fraction-static 
sglang_omni/cli/serve.py
187-201
默认模式：StageRuntimeConfig.sglang_server_args.mem_fraction_static 
sglang_omni/config/schema.py
73-85
资料来源： 
sglang_omni/cli/serve.py
187-201
 
sglang_omni/config/schema.py
73-85

关键 CLI 标志参考
旗帜	目的	实施细节
--model-path	HF 型号 ID 或路径	用于架构解析ConfigManager 
sglang_omni/config/manager.py
107
--config	YAML/JSON 配置路径	覆盖默认模型注册表配置
sglang_omni/config/manager.py
127
--colocate	所有阶段都在单个 GPU 上运行	验证Qwen3OmniSpeechColocatedPipelineConfig 
sglang_omni/cli/serve.py
103-107
--thinker-tp-size	Thinker 张量并行	stage.tp_size思考者角色的覆盖
sglang_omni/cli/serve.py
365-375
--text-only	禁用多模态阶段	variant="text"配置加载期间的选择
sglang_omni/cli/serve.py
488
--decode-mode	强制异步/同步解码	注入enable_async_decode到工厂参数中
sglang_omni/cli/serve.py
153-179
--mem-fraction-static	SGLang KV 缓存预算	适用于runtime.sglang_server_args 
sglang_omni/cli/serve.py
187-201
--encoder-mem-reserve	为编码器预留GPU内存	设置encoder_mem_reserve工厂参数
sglang_omni/cli/serve.py
246-274
资料来源： 
sglang_omni/cli/serve.py
187-440
 
sglang_omni/config/manager.py
45-70
 
sglang_omni/serve/launcher.py
93-115
 
sglang_omni/config/schema.py
128-210
6
 
sglang_omni/serve/openai_api.py
194-205
/v1/audio/transcriptions	语音转文本	支持带有转录适配器的 ASRMOSS-Transcribe-Diarize 
sglang_omni/serve/openai_api.py
10
 
sglang_omni/serve/openai_api.py
124
/v1/audio/voices	语音管理	通过以下方式上传和列出持久的 TTS 参考语音SpeakerSampleStore 
sglang_omni/serve/openai_api.py
11-13
 
sglang_omni/serve/openai_api.py
115
管理端点	RL 控制	/update_weight_from_disk诸如/pause_generation用于强化学习工作流程中动态模型更新的端点
sglang_omni/serve/openai_api.py
92-97
 
sglang_omni/serve/protocol.py
74-101
详情请参阅OpenAI 兼容 REST API。

实时 WebSocket API
服务器实现了两个不同的 WebSocket 接口，用于低延迟音频交互：

OpenAI Realtime API：该/v1/realtime端点为全双工多模态交互提供有状态会话，可通过enable_realtime应用工厂中的标志启用。
sglang_omni/serve/openai_api.py
18
 
sglang_omni/serve/openai_api.py
184-187
有状态 TTS 流：该/v1/audio/speech/stream端点由 管理SpeechWebSocketSession，允许客户端流式传输文本片段并接收连续的音频响应。
sglang_omni/serve/openai_api.py
9
 
sglang_omni/serve/openai_api.py
116
详情请参阅实时 WebSocket API。

SGLang-Omni 路由器
在大规模部署中，它sgl-omni-router充当外部负载均衡器，为多个工作进程提供服务。

拓扑结构：路由器前端连接多个sgl-omni serve工作节点，以扩展吞吐量并提供高可用性。
docs/basic_usage/omni_router.md
15-23
路由策略：支持诸如 `<p>` round_robin、least_request`<p>` 或 `<p>`之类的策略random，以将请求分发到健康的 Worker 节点。
sglang_omni_router/config.py
41
托管启动器：包含LocalLauncher用于托管工作池的功能以及用于复杂部署的基于 YAML 的配置。
docs/basic_usage/omni_router.md
30-45
有关详细信息，请参阅SGLang-Omni Router。

API 数据流
此图显示了如何将传入的数据ChatCompletionRequest转换为OmniRequest管道协调器使用的内部类型。



资料来源： 
sglang_omni/serve/openai_api.py
32-193
 
sglang_omni/client/client.py
78-156
 
sglang_omni/serve/protocol.py
41-89

资料来源：
sglang_omni/serve/openai_api.py
1-19
 
sglang_omni/cli/serve.py
1-67
 
sglang_omni/serve/launcher.py
93-116
 
sglang_omni/serve/protocol.py
72-101
 
sglang_omni/client/client.py
37-168
 
docs/basic_usage/omni_router.md
1-45









命令行界面和服务器启动器
相关源文件
CLIsgl-omni serve是启动 SGLang-Omni 推理栈的主要入口点。它负责处理模型架构解析、声明式配置合并以及多进程管道生命周期的编排。

CLI架构和入口点
CLI 使用以下方式构建typer并注册：sglang_omni.cli:app 
sglang_omni/cli/__init__.py
7-14
该命令定义于sglang_omni.cli.serve:serve 
sglang_omni/cli/serve.py
440
它支持两种主要操作模式：

自动解析：通过提供--model-path（HF ID 或本地路径）启动，系统会推断出正确的PipelineConfig类别。
sglang_omni/config/manager.py
107-124
显式配置：通过 YAML/JSON 文件启动--config，从而可以对阶段放置和运行时参数进行精细控制。
sglang_omni/config/manager.py
127-145
自然语言到代码实体映射：CLI流程
下图将 CLI 概念映射到负责处理它们的底层代码实体。

资料来源： 
sglang_omni/cli/serve.py
440
 
sglang_omni/config/manager.py
34-43
 
sglang_omni/models/registry.py
135

配置管理
这ConfigManager是决定模型应如何执行的中央权威机构。
sglang_omni/config/manager.py
34-40

分辨率逻辑
模型路径解析：如果--model-path提供，resolve_config_cls_for_model_path则检查元数据以architectures使用AutoConfig 
sglang_omni/config/manager.py
16-24
它还支持 Mistral 格式配置的备用逻辑（params.json）和原始 JSON 解析。
sglang_omni/config/manager.py
25-31
 
sglang_omni/utils/hf.py
54-129
注册表查找：将架构字符串（例如， ）与存储实现映射的Qwen3OmniForCausalLM注册表进行匹配。PIPELINE_CONFIG_REGISTRYPipelineConfig
sglang_omni/config/manager.py
31
 
sglang_omni/models/registry.py
114-120
变体选择：类似这样的标志--text-only会触发变体查找（例如，选择多模态配置的“文本”变体）
sglang_omni/config/manager.py
112-123
CLI 标志覆盖
CLI 提供高级标志，这些标志会被转换为PipelineConfig对象内部的深层覆盖：

并行性：--thinker-tp-size并--thinker-gpus覆盖 Thinker 阶段的默认张量并行性和设备放置。
sglang_omni/cli/serve.py
365-375
内存管理：--mem-fraction-static设置全局 KV 缓存预算，同时--thinker-mem-fraction-static提供逐阶段的精细化管理。
sglang_omni/cli/serve.py
187-201
共置：此--colocate标志强制所有阶段共享单个 GPU，需要特定的Qwen3OmniSpeechColocatedPipelineConfigGPU 类和显式的内存预算验证。
sglang_omni/cli/serve.py
89-109
解码模式：该--decode-mode标志（异步/同步）允许通过enable_async_decode在阶段中修改来强制模型进入特定的执行模式factory_args。Higgs TTS、MOSS-TTS-Local 和 Fun-ASR 等模型主要支持此功能。
sglang_omni/cli/serve.py
29-46
 
sglang_omni/cli/serve.py
153-179
资料来源： 
sglang_omni/config/manager.py
106-124
 
sglang_omni/cli/serve.py
187-201
 
sglang_omni/utils/hf.py
26-51

服务器启动器生命周期
配置完成后，launch_server启动运行时环境
sglang_omni/serve/launcher.py
228-275

生命周期序列
端口发现：如果请求的端口被占用，_find_available_port则自动绑定到一个随机的空闲端口并记录警告。
sglang_omni/serve/launcher.py
93-115
管道初始化MultiProcessPipelineRunner：使用最终的运行器实例化A PipelineConfig，该运行器会生成各个阶段进程。
sglang_omni/serve/launcher.py
241-243
应用创建create_app：通过挂载与 OpenAI 兼容的 REST 端点、用于 TTS 和实时 API 的 WebSocket 路由以及可选的分析器端点来创建 FastAPI 应用程序。
sglang_omni/serve/launcher.py
244-250
故障监控：启动器进入一个循环，持续监控子进程的运行状况runner.check_live()。如果任何阶段的进程崩溃，整个服务器都会关闭，以防止部分进程故障。
sglang_omni/serve/launcher.py
260-273
实现图：启动服务器


资料来源： 
sglang_omni/serve/launcher.py
228-275
 
sglang_omni/serve/launcher.py
93-115

高级配置逻辑
部分启动政策
启动器支持partial-startQwen3-Omni 的一项策略，允许说话者阶段在思考者阶段完成其全部响应之前开始生成音频。此功能可通过以下方式切换：--talker-partial-start 
sglang_omni/cli/serve.py
205-220
CLI 利用create_talker_ar_executor_from_config此行为
sglang_omni/cli/serve.py
64-66

GPU 兼容性和自定义全规避
启动器会自动检测硬件环境是否支持自定义的 all-reduce 内核。如果使用多个 GPU 进行张量并行 (TP) 但缺少 P2P​​ 访问，系统会自动将disable_custom_all_reduce: True相关参数注入 SGLang 服务器，以防止程序挂起。
sglang_omni/cli/serve.py
276-300
此检查使用should_disable_custom_all_reduce_for_gpus 
sglang_omni/cli/serve.py
23

记忆分数分辨率
KV缓存的内存分配是通过层次结构来实现的：

显式阶段覆盖：--thinker-mem-fraction-static 
sglang_omni/cli/serve.py
187-201
全局静态覆盖：--mem-fraction-static 
sglang_omni/cli/serve.py
187-201
默认模式：StageRuntimeConfig.sglang_server_args.mem_fraction_static 
sglang_omni/config/schema.py
73-85
资料来源： 
sglang_omni/cli/serve.py
187-201
 
sglang_omni/config/schema.py
73-85

关键 CLI 标志参考
旗帜	目的	实施细节
--model-path	HF 型号 ID 或路径	用于架构解析ConfigManager 
sglang_omni/config/manager.py
107
--config	YAML/JSON 配置路径	覆盖默认模型注册表配置
sglang_omni/config/manager.py
127
--colocate	所有阶段都在单个 GPU 上运行	验证Qwen3OmniSpeechColocatedPipelineConfig 
sglang_omni/cli/serve.py
103-107
--thinker-tp-size	Thinker 张量并行	stage.tp_size思考者角色的覆盖
sglang_omni/cli/serve.py
365-375
--text-only	禁用多模态阶段	variant="text"配置加载期间的选择
sglang_omni/cli/serve.py
488
--decode-mode	强制异步/同步解码	注入enable_async_decode到工厂参数中
sglang_omni/cli/serve.py
153-179
--mem-fraction-static	SGLang KV 缓存预算	适用于runtime.sglang_server_args 
sglang_omni/cli/serve.py
187-201
--encoder-mem-reserve	为编码器预留GPU内存	设置encoder_mem_reserve工厂参数
sglang_omni/cli/serve.py
246-274
资料来源： 
sglang_omni/cli/serve.py
187-440
 
sglang_omni/config/manager.py
45-70
 
sglang_omni/serve/launcher.py
93-115
 
sglang_omni/config/schema.py
128-210



兼容 OpenAI 的 REST API
相关源文件
SGLang-Omni 服务器提供了一个高性能、兼容 OpenAI 的 REST API 层，该 API 层构建于 FastAPI 之上。它作为外部应用程序的主要入口点，将标准的 OpenAI 协议模式转换为由OmniRequest多阶段流水线处理的内部消息。
sglang_omni/serve/openai_api.py
2-17

该 API 支持文本、音频、图像和视频输入的多模态扩展，以及文本和语音 (TTS) 的高级流式传输功能。

API 端点概述
服务器公开以下主要端点：
sglang_omni/serve/openai_api.py
5-19
：

端点	方法	描述
/v1/chat/completions	邮政	文本、图像、视频和音频聊天完成
sglang_omni/serve/protocol.py
41-101
/v1/audio/speech	邮政	使用 SSE 或原始 PCM 流媒体进行文本转语音 (TTS) 合成
sglang_omni/serve/protocol.py
246-271
/v1/audio/speech/batch	邮政	高通量批量TTS合成
sglang_omni/serve/protocol.py
293-305
/v1/audio/speech/stream	WS	用于实时文本输入的有状态 WebSocket TTS
sglang_omni/serve/openai_api.py
9
/v1/audio/transcriptions	邮政	带有用于语音分割的转录适配器的自动语音识别（例如，MOSS-Transcribe-Diarize）
sglang_omni/serve/transcriptions.py
51-57
/v1/audio/translations	邮政	将音频语音翻译成英语
sglang_omni/serve/openai_api.py
7
/v1/audio/voices	GET/POST	持久性TTS参考语音的管理
sglang_omni/serve/openai_api.py
9-11
/v1/models	得到	列出可用模型（报告当前在建项目）
sglang_omni/serve/protocol.py
534-541
/health	得到	返回服务器状态和运行状况信息
sglang_omni/serve/openai_api.py
17
/v1/realtime	WS	兼容 OpenAI 的实时 API（启用时）
sglang_omni/serve/openai_api.py
18
资料来源：
sglang_omni/serve/openai_api.py
1-19
 
sglang_omni/serve/protocol.py
41-101
 
sglang_omni/serve/protocol.py
534-541
 
sglang_omni/serve/transcriptions.py
50-57

请求翻译生命周期
当请求到达端点时，它会在提交给服务器之前经过转换过程Coordinator。

1. 模式验证
传入的 JSON 有效负载会根据 Pydantic 中定义的模型进行验证。这些sglang_omni.serve.protocol模型包括标准的 OpenAI 字段（例如，，temperature）和 SGLang- Omnimax_tokens扩展（例如modalities，，，，，）。audiosimagesvideosstage_sampling
sglang_omni/serve/protocol.py
41-101

2. 内部请求构建
API 层将已验证的协议对象转换为GenerateRequest。关键转换逻辑包括：

聊天：_build_chat_generate_request将多模态字段和分阶段采样映射到GenerateRequest元数据和额外参数
sglang_omni/serve/openai_api.py
25
 
tests/unit_test/serve/test_openai_api.py
27
ASR（build_transcription_generate_request别名speech_to_text.build_speech_to_text_generate_request）处理音频输入并解析转录适配器
sglang_omni/serve/transcriptions.py
38-40
 
sglang_omni/serve/speech_to_text.py
122-135
TTS：SpeechRequestValidator降低CreateSpeechRequest到一个PreparedSpeechRequest，最终用于构建一个GenerateRequest与output_modalities=["audio"] 
sglang_omni/serve/speech_service.py
168-171
3. 客户提交
它Client接收一个值GenerateRequest，将其包装在一个OmniRequest容器中_build_omni_request，然后提交给另一个容器Coordinator。它既支持用于代码补全的一次性操作submit()，也stream()支持用于实时迭代器的迭代器。
sglang_omni/client/client.py
60-72

4. 响应聚合/流式传输
该函数Client管理异步迭代StreamMessage或完成结果。对于非流式调用，completion()它会聚合文本片段，并在进行np.concatenateBase64 编码之前（使用适当的声道对齐方式）连接音频块。
sglang_omni/client/client.py
78-156

数据流：请求到管道
标题：API请求翻译流程























资料来源：
sglang_omni/serve/openai_api.py
23-31
 
sglang_omni/serve/protocol.py
41-101
 
sglang_omni/serve/speech_service.py
168-171
 
sglang_omni/client/client.py
53-72
 
sglang_omni/serve/transcriptions.py
38-40
 
sglang_omni/serve/speech_to_text.py
122-135

聊天完成情况（/v1/chat/completions）
此端点支持标准文本交互和多模态扩展。

多模态扩展
图片/视频/音频：以本地路径或 URL 列表的形式传递，分别位于 ` <images> images`、videos`<videos>` 或 ` <audio>`audios字段中。
sglang_omni/serve/protocol.py
71-79
模式：用户可以["text", "audio"]在该modalities字段中指定触发同时文本和语音输出。
sglang_omni/serve/protocol.py
64
阶段采样：允许通过或来覆盖特定阶段（例如，thinkervs talker）的采样参数。stage_samplingstage_params 
sglang_omni/serve/protocol.py
87-88
这包括特定字段talker_temperature，例如talker_top_p，等等。
sglang_omni/serve/protocol.py
91-95
流式传输行为
如果stream: true设置了该参数，服务器将返回一个StreamingResponse包装_chat_stream生成器的函数。
sglang_omni/serve/openai_api.py
26
生成器生成ChatCompletionStreamResponse对象
sglang_omni/serve/protocol.py
141-150
音频数据以 base64 编码字符串的形式提供。choices[0].delta.audio 
sglang_omni/serve/protocol.py
125-132

资料来源：
sglang_omni/serve/protocol.py
41-101
 
sglang_omni/serve/protocol.py
141-150
 
sglang_omni/serve/openai_api.py
26
 
sglang_omni/client/client.py
162-198

音频语音（/v1/audio/speech）
TTS 端点支持多种流媒体模式和高吞吐量批处理。

响应格式和流式传输
响应格式：支持wav、mp3、flac、和opusaacpcm 
sglang_omni/client/audio.py
24-31
批量处理：允许在单个请求中/v1/audio/speech/batch处理多个输入（最多 100 个）。DEFAULT_TTS_BATCH_MAX_ITEMS
sglang_omni/serve/protocol.py
293-305
 
sglang_omni/serve/openai_api.py
73
PCM 流媒体：支持原始 PCM 流媒体。对于有状态的实时输入，/v1/audio/speech/stream使用 WebSocket 端点。
sglang_omni/serve/openai_api.py
9
 
sglang_omni/serve/speech_ws.py
116
音频处理工具
速度调整：apply_speed通过重新采样来调整音频np.interp 
sglang_omni/client/audio.py
138-160
参考音频：支持通过SpeechReference（音频路径、URL 或 base64 数据）进行语音克隆
sglang_omni/serve/protocol.py
273-283
语音管理：持久语音可以通过以下方式进行管理/v1/audio/voices，并由以下方式提供支持：SpeakerSampleStore 
sglang_omni/serve/openai_api.py
9-13
 
sglang_omni/serve/speech_voices.py
115
音频架构
标题：音频编码和客户端生命周期


















资料来源：
sglang_omni/client/audio.py
110-160
 
sglang_omni/client/audio.py
162-222
 
sglang_omni/client/client.py
127-156
 
sglang_omni/serve/protocol.py
293-305
 
sglang_omni/client/types.py
21-32

音频转录（/v1/audio/transcriptions）
该端点提供 ASR 功能，包括通过自动分块支持长音频。

音频分块和验证
持续时间探测：probe_audio_duration在处理之前测量上传字节的持续时间，使用专门的路径处理 WAV 文件，并回退到PyAV容器格式。
sglang_omni/serve/speech_to_text.py
37
 
tests/unit_test/serve/test_speech_to_text.py
85-111
块规划：如果持续时间超过max_audio_clip_s（定义见AudioChunkingConfig），plan_audio_chunks则在工作线程中调用。
sglang_omni/serve/transcriptions.py
113-122
RMSSplitter：利用能量水平寻找分裂点
sglang_omni/serve/transcription_chunking.py
11
 
tests/unit_test/serve/test_transcription_chunking.py
194-196
转录适配器
服务器使用TranscriptionAdapter（通过解析resolve_adapter）对模型输出进行后处理
sglang_omni/serve/speech_to_text.py
40-41
这有助于为 MOSS-Transcribe-Diarize 等分屏模型生成专门的输出。

资料来源：
sglang_omni/serve/transcriptions.py
50-180
 
sglang_omni/serve/speech_to_text.py
176-215
 
sglang_omni/serve/transcription_chunking.py
10-27

RL 和管理员控制
SGLang-Omni 为强化学习 (RL) 工作流程和体重管理提供管理端点，通常受到以下限制：admin_api_key 
sglang_omni/serve/openai_api.py
189
 
sglang_omni/http/admin_auth.py
51

管理端点
生成控制：/pause_generation允许暂停调度程序以执行权重更新
sglang_omni/serve/protocol.py
92
 
sglang_omni/serve/openai_api.py
345
体重管理：端点如动态模型体重同步/update_weights_from_disk，并促进其同步。/update_weights_from_distributed
sglang_omni/serve/protocol.py
96-97
 
sglang_omni/serve/openai_api.py
353-370
更新组：/init_weights_update_group并/destroy_weights_update_group管理用于同步更新的集合组
sglang_omni/serve/protocol.py
84-89
权重检查器：/weights_checker验证分布式进程中模型权重的一致性
sglang_omni/serve/protocol.py
100
资料来源：
sglang_omni/serve/protocol.py
74-101
 
sglang_omni/serve/openai_api.py
345-370
 
sglang_omni/http/admin_auth.py
51

实施细节
服务器启动器生命周期
该launch_server函数负责协调启动过程。
sglang_omni/serve/launcher.py
6

端口发现：_find_available_port确保请求的端口空闲，或者找到备用端口。
sglang_omni/serve/launcher.py
93-115
应用创建：create_app使用已连接的设备初始化 FastAPI 应用程序。Client 
sglang_omni/serve/launcher.py
48
信号处理：_PipelineUvicornServer捕获SIGINT/SIGTERM以便优雅地关闭子工作进程
sglang_omni/serve/launcher.py
62-85
错误处理
服务器区分客户端错误（例如，上下文溢出）和服务器错误。

is_bad_request_error检查ClientError是否存在指示用户输入错误的标记
sglang_omni/serve/openai_errors.py
18
VoiceUploadBodyLimitMiddleware：一个中间件，用于在语音内容完全解析之前拒绝上传过大的语音文件。
sglang_omni/serve/openai_api.py
136-173
speech_error_response映射SpeechAPIError到与 OpenAI 兼容的 JSON 响应
sglang_omni/serve/speech_errors.py
107
资料来源：
sglang_omni/serve/launcher.py
1-115
 
sglang_omni/serve/openai_api.py
136-173
 
sglang_omni/serve/openai_errors.py
1-20
 
sglang_omni/serve/speech_errors.py
102-109




实时 WebSocket API
相关源文件
实时 WebSocket API 提供了一个有状态、低延迟的多模态交互接口，主要面向与 OpenAI 兼容的实时协议和有状态文本转语音 (TTS) 流。它在sglang_omni.serve.realtime子包中实现，并支持诸如服务器端语音活动检测 (VAD) silero-vad、音频缓冲和基于会话的生命周期管理等功能。

架构概述
该实时系统将异步的、基于请求的管道协调器与持久的 WebSocket 连接连接起来。它管理封装了对话状态的会话，包括音频缓冲区、VAD 状态机和历史对话项。

关键组成部分
成分	角色	来源
RealtimeSessionManager	管理活动实例的集合RealtimeSession并处理会话的打开/关闭。	
sglang_omni/serve/realtime/manager.py
13-44
RealtimeSession	拥有一个 WebSocket 和一个 OpenAI-Realtime 音频输入会话；协调 VAD 和生成。	
sglang_omni/serve/realtime/session.py
71-143
StreamingVAD	使用会话状态机silero-vad检测 PCM 流中的语音开始/停止事件。	
sglang_omni/serve/realtime/vad.py
40-110
RealtimeAudioBuffer	一个滚动式、仅追加的原始 PCM16 字节缓冲区，支持切片和 base64 摄取。	
sglang_omni/serve/realtime/audio_buffer.py
22-86
Client	GenerateRequest会话用于向后端提交对象的高级接口。	
sglang_omni/serve/realtime/session.py
13
Event Models	用于协议验证的Pydantic 模式（例如SessionUpdate，，InputAudioBufferAppend）。	
sglang_omni/serve/realtime/events.py
62-109
数据流图：音频到响应生命周期
该图说明了原始音频块如何从 WebSocket 通过 VAD 系统传输到管道，最终生成响应/转录输出。


















资料来源：
sglang_omni/serve/realtime/session.py
226-260
 
sglang_omni/serve/realtime/audio_buffer.py
39-45
 
sglang_omni/serve/realtime/vad.py
57-103
 
sglang_omni/serve/realtime/session.py
72-83

语音活动检测（VAD）
该系统用于silero-vad提供服务器端转向检测。该类StreamingVAD封装了模型并管理连续音频流所需的状态机。

VAD 配置：turn_detection通过以下方式镜像 OpenAI 的参数VADConfig，包括threshold（prefix_padding_ms包​​含语音前的上下文）和silence_duration_ms（触发回合完成）。
sglang_omni/serve/realtime/vad.py
17-30
处理逻辑：音频以 16 kHz 采样率（32 毫秒帧）在 512 个采样点的窗口内进行处理。VAD 维护一个leftover_pcm缓冲区来处理来自客户端的未对齐数据块。
sglang_omni/serve/realtime/vad.py
12-14
 
sglang_omni/serve/realtime/vad.py
64-67
事件发射：当检测到语音时，它会计算sample_offset包含填充。当静音超过持续时间阈值时，它会发射一个事件SPEECH_STOPPED。
sglang_omni/serve/realtime/vad.py
81-100
音频推理：该infer方法用于torch.inference_mode()在 512 个样本帧上执行 Silero 模型，并返回语音概率。
sglang_omni/serve/realtime/vad.py
105-109
资料来源：
sglang_omni/serve/realtime/vad.py
40-122

会话生命周期和事件
该 API 遵循 [此处应填写具体定义] 中定义的严格事件驱动协议events.py。会话是有状态的，并且会持久化对话历史记录，以便在回合之间保持上下文。

实时会话管理
它RealtimeSessionManager充当活动连接的主要注册中心。

open(websocket)创建一个新的映射RealtimeSession并将其添加到内部sessions映射中。
sglang_omni/serve/realtime/manager.py
26-35
close(session_id)：触发session.teardown()​​并从管理器中移除会话。
sglang_omni/serve/realtime/manager.py
37-41
轮次序列化：如果 VAD在引擎仍在处理先前话语时发出响应，则RealtimeSession使用asyncio.Queue( response_queue) 来序列化响应。speech_stopped
sglang_omni/serve/realtime/session.py
133-134
音频缓冲
输入音频由该设备处理RealtimeAudioBuffer，该设备维护一个bytearray原始 PCM16 字节的数据。

append_b64(audio_b64)：解码 base64 并将其附加到缓冲区，检查是否超过硬性限制（默认 60 秒）以防止内存耗尽。
sglang_omni/serve/realtime/audio_buffer.py
39-44
 
sglang_omni/serve/realtime/audio_buffer.py
10-11
WAV 转换：缓冲区可以将切片转换为 WAV 数据 URI，wave以便io.BytesIO提交到多模态管道。
sglang_omni/serve/realtime/audio_buffer.py
68-80
在代码空间中实现
下图将逻辑实时实体映射到sglang-omni代码库中的特定类和方法。






























资料来源：
sglang_omni/serve/realtime/manager.py
13-24
 
sglang_omni/serve/realtime/session.py
71-143
 
sglang_omni/serve/realtime/events.py
62-65
 
sglang_omni/serve/realtime/audio_buffer.py
22-37

多模态交互逻辑
该方法RealtimeSession对检测到的每一个用户话语都采用两步处理策略：

响应生成：利用音频和历史conversation数据来流式传输response.text.delta（并可选择性地audio包含增量）事件。这能为用户提供即时反馈。
sglang_omni/serve/realtime/session.py
75-76
转录：重新播放同一音频，并使用预设的硬编码程序_TRANSCRIPTION_PROMPT生成逐字转录文本。此功能用于历史记录和用户界面日志记录。
sglang_omni/serve/realtime/session.py
37-41
 
sglang_omni/serve/realtime/session.py
77-79
笔录和助理的回复都会被附加到文本中self.conversation，作为下一轮的文本上下文。
sglang_omni/serve/realtime/session.py
80-83

强行闯入和中断
实时 API 支持“插话”功能，用户可以通过说话打断助手的回复。

轮次检测中断：如果turn_detection.interrupt_response启用，当通过 VAD 检测到新的语音时，服务器会自动取消当前正在生成的语音。
sglang_omni/serve/realtime/session.py
334-345
显式取消：客户端可以发送response.cancel事件来停止当前生成。
sglang_omni/serve/realtime/session.py
284-290
截断：如果在播放过程中助手响应被中断，客户端会发送conversation.item.truncate一个消息audio_end_ms。然后服务器会将该项在历史记录中标记为已截断。
sglang_omni/serve/realtime/session.py
292-310
资料来源：
sglang_omni/serve/realtime/session.py
284-310
 
sglang_omni/serve/realtime/events.py
85-90
 
tests/unit_test/serve/test_realtime_barge_in.py
93-146

游乐场融合
我们提供了一个“Wire Service”网络演示，其中playground/qwen-omni/realtime/展示了该协议：

麦克风捕获AudioWorklet：通过base64 编码捕获 16 kHz 单声道 PCM16信号。
playground/qwen-omni/realtime/app.js
192-212
协议：发送input_audio_buffer.append帧和句柄session.update以在纯文本模式和文本+音频模式之间切换。
playground/qwen-omni/realtime/app.js
103-119
播放控制器：一个RealtimePlaybackController（实例化为playback）使用 Web Audio API 管理 24 kHz PCM16 音频块的无缝播放。
playground/qwen-omni/realtime/app.js
58-59
 
playground/qwen-omni/realtime/playback.js
4-23
资料来源：
sglang_omni/serve/realtime/session.py
71-83
 
playground/qwen-omni/realtime/app.js
1-12
 
playground/qwen-omni/realtime/playback.js
4-23


SGLang-Omni 路由器
相关源文件
SGLang-Omni Router 是一个外部的、高性能的 HTTP 代理进程，旨在为多个sgl-omni serve工作实例提供前端支持。
docs/basic_usage/omni_router.md
3-5
它为客户端应用程序提供了一个统一的 OpenAI 兼容端点，同时通过健康跟踪、负载均衡、能力感知路由和自动化生命周期管理来管理工作节点池。
docs/basic_usage/omni_router.md
7-9

系统拓扑
路由器充当网关，将请求分发给同构或异构的工作节点池。与处理模型执行特定部分的内部流水线阶段不同，路由器后面的每个工作节点都是一个完整、独立的 Omni V1 服务器实例。
docs/basic_usage/omni_router.md
25-28

路由器-工作者关系
路由器不会将单个请求拆分到多个工作节点；相反，它会根据路由策略选择最合适的工作节点，并将整个请求转发出去。
docs/basic_usage/omni_router.md
26-28
它使用内部机制ProxyHandler来管理上游请求的生命周期，包括标头过滤和准入控制。
sglang_omni_router/app.py
155-161

实体映射：请求流

















资料来源：
docs/basic_usage/omni_router.md
15-23
 
sglang_omni_router/app.py
109-168
 
sglang_omni_router/proxy.py
108-146
 
docs/basic_usage/omni_router.md
32-34
 
sglang_omni_router/app.py
34-39

多进程 CP/DP 架构
对于高吞吐量部署，路由器支持多进程架构，该架构将控制平面 (CP)和数据平面 (DP)分开。 
sglang_omni_router/serve.py
163-172
--router-processes通过将值设置为 $\ge 2$ 来激活此模式。

控制平面（CP）
CP流程是工人登记、健康检查和行政管理的唯一所有者
sglang_omni_router/app.py
51-57

快照发布：CP 发布状态更改（健康状况、工作节点注册表）供 DP 使用。
准入管理：协调所有 DP 流程的全局在途请求限制，以防止工作进程池过载。
sglang_omni_router/proxy.py
108-132
数据平面（DP）
DP进程是无状态的中继，负责处理实际的HTTP/WebSocket流量。
sglang_omni_router/proxy.py
1-9

请求转发：分发点 (DP) 使用来自控制点 (CP) 的快照来执行路由决策，并将请求字节转发给工作节点。
sglang_omni_router/proxy.py
31-35
响应中继：分发点将响应流式传输回客户端，并在上游故障时处理 SSE 事件注入。
sglang_omni_router/proxy.py
80-86
资料来源：
sglang_omni_router/app.py
1-57
 
sglang_omni_router/proxy.py
1-146
 
sglang_omni_router/serve.py
163-172

托管工人池
路由器包含一个启动器（通过 调用--launcher-config），可自动部署本地工作副本。
docs/basic_usage/omni_router.md
32-34
它管理工作子进程，通过某种方式分配 GPU ，并在将它们暴露给流量之前CUDA_VISIBLE_DEVICES监控它们的端点。/health
docs/basic_usage/omni_router.md
62-67

启动器配置（YAML）
启动器通过 YAML 文件进行配置，该文件定义了模型路径、工作进程数和 GPU 分配。
docs/basic_usage/omni_router.md
49-60

钥匙	描述
backend	设置local为子进程管理
docs/basic_usage/omni_router.md
51
num_workers	sgl-omni serve要生成的实例总数
docs/basic_usage/omni_router.md
54
num_gpus_per_worker	分配给每个工作实例的 GPU 数量
docs/basic_usage/omni_router.md
55
worker_extra_args	直接传递给工作进程sgl-omni serve命令的 CLI 标志
docs/basic_usage/omni_router.md
58
worker_capabilities	支持的模态的明确列表（例如chat，，audio_output）
docs/basic_usage/omni_router.md
103-109
资料来源：
docs/basic_usage/omni_router.md
49-60

路由和健康检查
路由策略和元数据
路由器支持多种请求分发策略
sglang_omni_router/config.py
31
：

round_robin按顺序轮换使用可用的健康工人。
least_request：将转发转发给活动连接数最少的进程。
sglang_omni_router/proxy.py
194
random随机选择一名健康工人。
在路由之前，会ProxyHandler解析extract_route_metadata请求体或特定标头，以确定请求是否需要特定的工作进程功能，例如image_input、audio_input或。video_input 
sglang_omni_router/proxy.py
23-30

语音路由
该路由器支持有状态语音路由VoiceRoutingState 
sglang_omni_router/app.py
148-154

语音所有权：某些工作人员可以被指定为“语音所有者”，用于持续上传语音内容。
sglang_omni_router/config.py
29-30
请求关联性：在注册表中指定特定语音名称的请求将被路由到当前拥有该语音数据的工作进程。
sglang_omni_router/proxy.py
36-37
健康检查生命周期
路由器通过该类执行后台健康检查HealthChecker。
sglang_omni_router/app.py
142-146

间隔：定期执行检查（默认值：10 秒）。
故障阈值unhealthy：连续发生 $N$ 次故障后，工作进程将被标记。
sglang_omni_router/app.py
110
成功阈值：一名unhealthy或一名unknown工人必须通过 $M$ 次连续检查才能返回池中。
实体映射：健康状况和路由逻辑











资料来源：
sglang_omni_router/app.py
142-147
 
sglang_omni_router/proxy.py
75-78
 
sglang_omni_router/worker.py
42-47

管理与控制平面
该路由器支持强化学习 (RL) 和模型权重更新工作流程的管理广播。

广播行为
当收到管理员请求（例如，`add`/update_weights_from_disk或 `add` ）时，路由器会将该请求广播给所有工作节点。/pause_generation
sglang_omni_router/app.py
51-57

权重更新日志：路由器使用日志UpdateJournal来确保权重更新过程不会崩溃。如果路由器在更新过程中崩溃，它会“失败关闭”，即禁用目标直到验证通过。
sglang_omni_router/app.py
61-82
管理员锁定：路由器使用admin_update_lock串行化广播来传输权重更新信息，以防止出现竞态条件。
sglang_omni_router/app.py
181
身份验证：管理端点受通过配置的 Bearer 令牌保护。admin_api_key 
sglang_omni_router/app.py
195-201
资料来源：
sglang_omni_router/app.py
51-97
 
sglang_omni_router/app.py
181-193

CLI 用法
路由器通过sgl-omni-router命令行界面 (CLI)启动。
docs/basic_usage/omni_router.md
165

使用托管工作进程启动
受管工作进程通过 YAML 配置以子进程的形式启动。路由器会等待所有受管工作进程完成，/health然后才开始接受客户端流量。
docs/basic_usage/omni_router.md
65-67

sgl-omni-router \
  --port 8008 \
  --launcher-config examples/configs/qwen3_omni_router.yaml \
  --policy round_robin
资料来源：
docs/basic_usage/omni_router.md
36-40

利用现有员工启动
如果工作进程已经在运行，路由器可以直接指向它们的 URL。
docs/basic_usage/omni_router.md
162-168

sgl-omni-router \
  --port 8008 \
  --worker-urls http://127.0.0.1:8011 http://127.0.0.1:8012 \
  --model qwen3-omni
资料来源：
docs/basic_usage/omni_router.md
165-174

实施细节
代理和标头过滤
该ProxyHandler实现逻辑会在将请求转发给工作进程之前，逐跳剥离路由头和内部路由元数据，以保持透明度。
sglang_omni_router/proxy.py
218-232

请求头：诸如 `<header_name>` host、` content-length<header_name>` 和 `<header_name>`之类的标头accept-encoding会被移除，以便路由器httpx客户端能够为上游连接正确设置它们。
sglang_omni_router/proxy.py
53-61
响应头：诸如 `$($($($($($($($($($($($($($($( date) server...
sglang_omni_router/proxy.py
62-66
入场控制和泳池管理
路由器实现了AdmissionController对全局在途请求的约束。
sglang_omni_router/proxy.py
108-113

连接池大小：上游连接池的大小是根据max_connections以下因素确定的：effective_max_inflight 
sglang_omni_router/app.py
127-128
流式中继：路由器使用自定义的_RelayResponse（Starlette 子类StreamingResponse）来确保即使客户端在数据流开始传输之前断开连接，上游连接和准入槽位也能被释放。
sglang_omni_router/proxy.py
199-216
资料来源：
sglang_omni_router/proxy.py
53-66
 
sglang_omni_router/proxy.py
108-146
 
sglang_omni_router/proxy.py
199-216
 
sglang_omni_router/app.py
127-140
