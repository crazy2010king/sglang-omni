核心架构
相关源文件
SGLang-Omni 运行时构建为一个多阶段异步流水线，专为低延迟多模态交互而设计。与单体式 LLM 引擎不同，SGLang-Omni 将模型执行分解为离散的、相互连接的阶段，这些阶段通过高性能的控制平面和数据平面进行通信。
sglang_omni/pipeline/stage/runtime.py
2-8
这种架构支持复杂的拓扑结构，例如“思考者-说话者”模式，其中推理模型（思考者）将隐藏状态流式传输到生成模型（说话者），以同时进行语音和文本合成。
sglang_omni/models/qwen3_omni/config.py
132-207

系统概述
该系统采用以计算为中心的设计，每个阶段都运行自己独立的调度器，以应对其特定的瓶颈（例如，计算密集型思考者与内存密集型说话者）。
sglang_omni/pipeline/stage/runtime.py
7-8

层	责任	是否具备模型感知能力？
协调员	请求生命周期、路由到入口阶段、最终结果合并、中止广播
sglang_omni/pipeline/coordinator.py
42-51
不
阶段	IO shell：ZMQ 控制平面、传输数据平面、扇入聚合、流路由
sglang_omni/pipeline/stage/runtime.py
65-80
不
调度程序	批次选择、键值缓存管理、计算调度（例如OmniScheduler）
sglang_omni/scheduling/omni_scheduler.py
162-175
部分
模型运行器	模型前向传播、采样、模型特定接口（例如，多模态嵌入注入）
sglang_omni/model_runner/base.py
91-97
是的
类关系图
下图将高级系统组件与代码库中的特定类联系起来。




























资料来源：
sglang_omni/pipeline/coordinator.py
42-51
 
sglang_omni/pipeline/stage/runtime.py
65-110
 
sglang_omni/proto/request.py
34-48
 
sglang_omni/pipeline/mp_runner.py
1-7

管道配置和拓扑结构（#2.1）
执行计划由声明式配置解析得出。PipelineConfig 
sglang_omni/config/schema.py
26
定义了对象的顺序StageConfig，而ProcessTopologyPlan 
sglang_omni/pipeline/mp_runner.py
26
决定如何将这些阶段映射到物理操作系统进程和 GPU。此层处理阶段融合（将多个阶段放置在一个进程中）和张量并行 (TP)放置。
sglang_omni/pipeline/mp_runner.py
122-125
它MultiProcessPipelineRunner通过以下方式将这些声明式配置解析为具体的执行计划prepare_pipeline_runtime 
sglang_omni/pipeline/mp_runner.py
28-33

详情请参见管道配置和拓扑。

资料来源： 
sglang_omni/pipeline/mp_runner.py
70-120
 
tests/unit_test/pipeline/test_compile.py
24-89
 
sglang_omni/config/schema.py
26

协调员和舞台运行时（#2.2）
这Coordinator 
sglang_omni/pipeline/coordinator.py
42-51
是请求的全球入口点。它跟踪RequestState 
sglang_omni/proto/request.py
23
并确保如果请求涉及多个终端阶段，结果能够正确合并。它还管理诸如权重更新或生成暂停之类的管理操作。admin() 
sglang_omni/pipeline/coordinator.py
168-204

每个Stage 
sglang_omni/pipeline/stage/runtime.py
65-110
充当一个 I/O 外壳Scheduler。它处理：

控制平面：接收SubmitMessage，DataReadyMessage或AbortMessage通过 ZMQ
sglang_omni/pipeline/stage/runtime.py
34-48
输入聚合：wait_for在开始计算之前等待多个上游依赖项（ ），由……管理InputHandler 
sglang_omni/pipeline/stage/runtime.py
117
流程管理： MultiProcessPipelineRunner 
sglang_omni/pipeline/mp_runner.py
1-7
生成工作进程并监控其运行状况。
有关详细信息，请参阅协调器和阶段运行时。

资料来源： 
sglang_omni/pipeline/coordinator.py
53-114
 
sglang_omni/pipeline/stage/runtime.py
130-162
 
sglang_omni/pipeline/mp_runner.py
70-88

调度器（#2.3）
调度器负责管理阶段的“计算”环节。它们从收件箱中提取任务，并将结果推送到发件箱。

OmniScheduler：针对具有键值缓存管理、异步解码前瞻和隐藏状态流的自回归 (AR) 模型进行了优化
sglang_omni/scheduling/omni_scheduler.py
162-205
SimpleScheduler：用于非AR任务，例如预处理和编码器。
tests/unit_test/pipeline/test_scheduler.py
116-119
ThreadedSimpleScheduler：在线程池中并发运行请求
tests/unit_test/pipeline/test_scheduler.py
154-176
DLLMScheduler：专门用于基于扩散的语言模型，如 LLaDA2-Uni。
有关详细信息，请参阅调度程序。

资料来源： 
sglang_omni/scheduling/omni_scheduler.py
162-205
 
tests/unit_test/pipeline/test_scheduler.py
21-25
 
sglang_omni/pipeline/stage/runtime.py
68-77

阶段间沟通（#2.4）
SGLang-Omni 使用由以下系统管理的双平面通信系统：CommEngine 
sglang_omni/comm/engine.py
94-100
：

控制平面：使用ZMQ + msgpack 进行轻量级信令传输（例如DataReadyMessage，，，，）AbortMessageStreamMessageCompleteMessage
sglang_omni/proto/messages.py
34-48
数据平面（中继）：通过以下方式传输大型张量Relay 
sglang_omni/relay/base.py
49
它支持多种后端。
sglang_omni/comm/data_ref.py
22-23
：
shm：用于零拷贝同节点传输的共享内存。
nccl：用于跨 TP 等级的 GPU 到 GPU 传输。
nixl：RDMA/SHM 元数据专用传输。
mooncake：用于大规模部署的分布式传输引擎。
本地调度：通过同一流程中的融合阶段进行优化LocalStageDispatcher 
sglang_omni/pipeline/stage/runtime.py
105
详情请参见“阶段间沟通”。

资料来源： 
sglang_omni/comm/engine.py
160-202
 
sglang_omni/comm/router.py
131-141
 
sglang_omni/comm/data_ref.py
17-23
 
sglang_omni/pipeline/stage/runtime.py
130-146

模型运行器图层（#2.5）
这ModelRunner 
sglang_omni/model_runner/base.py
91-97
该层感知特定的模型权重和架构。它处理前向传播的生命周期，包括execute_launch异步execute_resolve解码前瞻。
sglang_omni/model_runner/base.py
106-116
自定义模型运行器QwenTalkerModelRunner通过在阶段工厂中实例化来接入管道。
tests/unit_test/qwen3_omni/test_talker.py
29

详情请参见模型运行器图层。

资料来源： 
sglang_omni/model_runner/base.py
91-116
 
tests/unit_test/pipeline/test_async_decode.py
35-52
 
sglang_omni/pipeline/stage/runtime.py
96-98

请求生命周期流程
下图展示了请求在典型的多阶段管道中流动时，代码实体之间的交互情况。



资料来源：
sglang_omni/pipeline/coordinator.py
216-240
 
sglang_omni/pipeline/stage/runtime.py
163-185
 
sglang_omni/comm/engine.py
191-202
 
sglang_omni/proto/messages.py
34-48


管道配置和拓扑
相关源文件
SGLang-Omni 流水线由声明式配置定义，该配置描述了逻辑阶段、它们的硬件需求以及它们之间的数据路由。系统使用“编译”阶段将这种高级意图解析为具体的执行计划，该计划涉及多个进程、GPU 内存分配和进程间通信 (IPC) 通道。

声明式配置
配置分为两个主要层级：PipelineConfig和StageConfig。这些类使用 Pydantic 进行验证，并作为模型特定定义和与模型无关的运行时之间的契约。
sglang_omni/config/schema.py
1-9
 
sglang_omni/config/schema.py
128-144

PipelineConfig
PipelineConfig定义模型执行流程的全局参数
sglang_omni/config/schema.py
211-214

model_path：通往拥抱脸模型或本地检查点的路径
sglang_omni/config/schema.py
216
stagesStageConfig定义计算图的对象列表
sglang_omni/config/schema.py
217
relay_backend用于数据传输的后端（例如shm，，，，）ncclnixlmooncake
sglang_omni/config/schema.py
220
fused_stages为了实现本地调度优化，应该位于同一操作系统进程中的阶段组
sglang_omni/config/schema.py
221
env_defaults应用于所有工作进程的全局环境变量
sglang_omni/config/schema.py
222
StageConfig
StageConfig描述一个单一的逻辑工作单元
sglang_omni/config/schema.py
128-191

标识和工厂：name标识阶段，并factory提供指向创建该阶段执行器的函数的点式导入路径。
sglang_omni/config/schema.py
149-152
并行度：tp_size定义张量并行度。如果gpu是一个列表，其长度必须与列表长度匹配。tp_size 
sglang_omni/config/schema.py
161-162
 
tests/unit_test/pipeline/test_compile.py
48-52
路由：next定义静态下游目标，同时route_fn允许动态的、请求感知的路由。
sglang_omni/config/schema.py
156-158
扇入：wait_for指定在此阶段触发之前必须完成的上游阶段。
sglang_omni/config/schema.py
174
merge_fn需要A来合并它们的输出
sglang_omni/config/schema.py
176
流式传输：stream_to标识接收中间数据块的阶段
sglang_omni/config/schema.py
179
进程分配：每个非TP阶段都可以声明一个process字符串来定义其操作系统进程组
sglang_omni/config/schema.py
164
资料来源： 
sglang_omni/config/schema.py
11-222
 
tests/unit_test/pipeline/test_compile.py
23-87
 
sglang_omni/config/topology.py
83-90

编译阶段
“编译”阶段将声明式转换PipelineConfig为 aProcessTopologyPlan和 aStagePlacementPlan 
sglang_omni/pipeline/runtime_config.py
32-33

工艺拓扑分辨率
该build_process_topology_plan函数决定哪些阶段共享操作系统进程。
sglang_omni/config/topology.py
36-57

显式分组：具有相同process字符串的阶段将被分组在一起。
sglang_omni/config/topology.py
116-121
融合：所列阶段fused_stages统一为单一工艺组
sglang_omni/config/topology.py
123-138
TP 隔离：张量并行阶段会自动分配给每个进程的唯一进程（例如，{process}_tp{rank}）
sglang_omni/config/topology.py
160-172
验证：该计划确保任何进程组都不会跨越多个物理 GPU。
sglang_omni/config/topology.py
175-215
GPU布局和内存预算
内存分配逻辑会计算每个进程所需的GPU内存总占比。
sglang_omni/config/schema.py
50-72

内存分数：阶段可以请求total_gpu_memory_fraction 
sglang_omni/config/schema.py
55-63
规划器会验证任何单个 GPU 上的总和不超过max_total_gpu_memory_fraction_per_gpu 
sglang_omni/config/schema.py
116-127
SGLang 参数：mem_fraction_static传递给底层 SGLang 引擎以预留 KV 缓存空间
sglang_omni/config/schema.py
73-87
IPC端点分配
运行时会创建一个用于 IPC 套接字的临时目录。
sglang_omni/pipeline/runtime_config.py
29-33
每个阶段都被分配一个唯一的套接字地址
sglang_omni/pipeline/mp_runner.py
62-63

资料来源： 
sglang_omni/config/topology.py
1-215
 
sglang_omni/config/schema.py
31-127
 
sglang_omni/pipeline/runtime_config.py
12-33

模型注册和自动分辨率
SGLang-Omni 支持基于模型架构的自动配置选择。

注册表系统
它PIPELINE_CONFIG_REGISTRY存储模型架构与其对应PipelineConfig类之间的映射关系。
sglang_omni/models/registry.py
88-136
它扫描包含一个config.py的子模块EntryClass 
sglang_omni/models/registry.py
35-72

配置发现
它ConfigManager提供用于解析配置的实用程序：

from_model_pathconfig.json通过检查 Hugging FaceAutoConfig来确定架构并获取已注册的信息。PipelineConfig 
sglang_omni/config/manager.py
107-124
from_file加载配置的 YAML/JSON 导出文件，允许用户通过以下方式覆盖默认值stage_overrides 
sglang_omni/config/manager.py
127-145
架构解析：如果AutoConfig失败，系统会尝试从params.json（Mistral 格式）或原始config.json解析中解析架构。
sglang_omni/utils/hf.py
38-51
 
sglang_omni/utils/hf.py
85-130
资料来源： 
sglang_omni/models/registry.py
1-136
 
sglang_omni/config/manager.py
1-186
 
sglang_omni/utils/hf.py
23-130

拓扑图和数据流图
从配置到流程执行
此图说明了如何PipelineConfig将解析为ProcessTopologyPlan用于MultiProcessPipelineRunner生成工作进程的。

标题：管道分辨率数据流




















资料来源： 
sglang_omni/config/manager.py
107-124
 
sglang_omni/pipeline/runtime_config.py
17-33
 
sglang_omni/config/topology.py
28-33
 
sglang_omni/pipeline/mp_runner.py
8-13

阶段间路由逻辑
此图显示了StageConfig属性如何定义阶段之间的消息流以及如何处理扇入。

标题：阶段路由和扇入逻辑





















资料来源： 
sglang_omni/config/schema.py
128-187
 
tests/unit_test/pipeline/test_compile.py
89-153

高级 CLI 覆盖
CLIsgl-omni serve提供了几个标志来覆盖模型中定义的默认拓扑结构。这些标志会在编译阶段之前PipelineConfig解析。sglang_omni/cli/serve.py

--isolate-stage [NAME]强制将特定阶段放入其自身的操作系统进程中，即使默认配置将其融合在一起。
sglang_omni/cli/serve.py
117-121
--stage-process [NAME] [PROCESS_NAME]：明确地将阶段分配给指定的进程组
sglang_omni/cli/serve.py
123-128
--colocate：一种简写形式，用于特定型号系列（例如 Qwen3-Omni），使用预定义的、针对单节点多 GPU 设置优化的共置拓扑结构。
sglang_omni/models/qwen3_omni/config.py
35-39
--thinker-tp-size [N]：覆盖阶段的设置tp_size，thinker这将触发重新计算ProcessTopologyPlan 
sglang_omni/cli/serve.py
105-110
--decode-mode [async|sync]：覆盖enable_async_decode受支持阶段（例如 Higgs TTS、Qwen3-ASR）的设置
sglang_omni/cli/serve.py
83-87
 
tests/unit_test/higgs_tts/test_cli_decode_mode.py
15-16
--async-lookahead-min-batch-size [N]调整异步解码调度器的前瞻阈值
tests/unit_test/higgs_tts/test_cli_decode_mode.py
59-63
资料来源： 
sglang_omni/cli/serve.py
100-150
 
sglang_omni/models/qwen3_omni/config.py
35-40
 
tests/unit_test/higgs_tts/test_cli_decode_mode.py
1-182

配置汇总表
班级	责任	关键字段
PipelineConfig	全球拓扑	stages，，，relay_backend​fused_stages​env_defaults 
sglang_omni/config/schema.py
211-222
StageConfig	逻辑单元	factory，，，，，next​gpu​tp_size​process​wait_for 
sglang_omni/config/schema.py
128-187
ProcessTopologyPlan	操作系统布局	groups，，stage_to_process​tp_stage_to_processes 
sglang_omni/config/topology.py
28-33
CommConfig	数据传输	slot_size_mb，，credits​cuda_ipc_slot_size_kb 
sglang_omni/config/schema.py
11-28
资料来源： 
sglang_omni/config/schema.py
11-222
 
sglang_omni/config/topology.py
28-33



协调员和舞台运行时间
相关源文件
协调器和阶段运行时构成了 SGLang-Omni 的执行骨干。协调器Coordinator管理整个管道中请求的高级生命周期，而阶段运行时则Stage充当一个独立的 IO shell，管理特定组件（例如，思考器、对话器或预处理器）的数据移动和控制信号。

协调员：请求生命周期管理
它Coordinator是跟踪请求状态、管理终端条件合并和广播中止信号的中央权威机构。它通过CoordinatorControlPlane使用 ZMQ 与各个阶段进行通信。msgpack 
sglang_omni/pipeline/control_plane.py
79-82

主要职责
注册：阶段通过以下方式注册其控制端点register_stage 
sglang_omni/pipeline/coordinator.py
115-123
提交：对象的入口点OmniRequest。它跟踪RequestState（待处理、运行中、已完成、失败、已中止）状态。
sglang_omni/proto/request.py
9-16
多终端合并：对于复杂的管道，协调器可以等待多个终端阶段完成后再解决请求。
sglang_omni/pipeline/coordinator.py
76-80
对于文本和音频输出都必须完成的多模态模型来说，这一点至关重要（例如，decode对于code2wavQwen3-Omni）。
中止广播：当请求中止时，协调器会AbortMessage通过广播向所有阶段发送中止通知。PubSocket 
sglang_omni/pipeline/coordinator.py
246-254
 
sglang_omni/pipeline/control_plane.py
166-172
管理操作：通过各种方式管理管理任务，例如跨阶段的模型信息查询或权重更新。AdminOperation 
sglang_omni/pipeline/coordinator.py
168-206
容量管理：如果max_in_flight启用此功能，当运行容量和排队容量的总和达到上限时，协调器将拒绝新的提交。
sglang_omni/pipeline/coordinator.py
62-88
数据流：从请求提交到完成
下图说明了它如何与实体Coordinator交互。StageControlPlane

协调员请求生命周期



资料来源：
sglang_omni/pipeline/coordinator.py
214-254
 
sglang_omni/pipeline/coordinator.py
286-320
 
sglang_omni/proto/messages.py
227-230
 
sglang_omni/proto/request.py
36-42

舞台运行时：IO Shell
AStage是管道的封装器Scheduler（例如OmniScheduler或SimpleScheduler）。它处理管道中的“脏活”：ZMQ 控制消息、中继（共享内存/NCCL）数据传输和输入聚合。
sglang_omni/pipeline/stage/runtime.py
65-80

实施细节
角色隔离：一个阶段可以是主控single、leader后端或follower终端。leader各个single阶段拥有自己的外部 I/O，而follower阶段（在 TP 组中）通过内部队列接收工作。
sglang_omni/pipeline/stage/runtime.py
82-109
 
sglang_omni/pipeline/stage_workers.py
111-121
输入聚合：（InputHandler通常DirectInput是或AggregatedInput）管理扇入。它会等待所有必需的上游阶段提供数据，然后才会触发调度程序。
sglang_omni/pipeline/stage/runtime.py
116-117
 
sglang_omni/pipeline/stage/input.py
21-25
中继 I/O：数据传输机制由CommEngine负责对位置进行分类并使用执行中继 I/O 的组件管理。DataRef 
sglang_omni/comm/engine.py
94-99
元数据通过以下方式传递DataReadyMessage 
sglang_omni/proto/messages.py
18-27
流管理：阶段负责StreamQueue将部分数据块路由到协调器或下游阶段。
sglang_omni/pipeline/stage/runtime.py
151-154
调度器线程：该阶段会生成一个专用线程来运行调度器，确保计算不会阻塞 asyncio 事件循环。
sglang_omni/pipeline/stage/runtime.py
171-177
舞台内部架构






















资料来源：
sglang_omni/pipeline/stage/runtime.py
171-205
 
sglang_omni/pipeline/stage/runtime.py
250-290
 
sglang_omni/comm/engine.py
191-202

多进程流水线运行器
它将MultiProcessPipelineRunner声明式配置转换为物理过程拓扑。
sglang_omni/pipeline/mp_runner.py
2-7

工艺拓扑规划
阶段被分组为StageWorkerProcessSpec对象。一个工作进程可以承载多个位于同一位置的非TP阶段，而TP等级则各自拥有自己的进程。
sglang_omni/pipeline/stage_workers.py
125-129
 
sglang_omni/pipeline/stage_workers.py
140-149

环境设置：运行器通过以下方式将环境变量和 GPU 特定的默认值注入到每个进程中。_patched_spawn_env 
sglang_omni/pipeline/stage_workers.py
152-191
进程生成：用于multiprocessing启动工作进程。每个进程都会执行_run_process初始化Stage对象的操作。
sglang_omni/pipeline/stage_workers.py
236-258
故障监控：StageGroup监控子进程；如果任何阶段崩溃，则会捕获故障域并通知协调器。
sglang_omni/pipeline/stage_workers.py
260-275
本地调度：对于共享同一流程的阶段，LocalStageDispatcher通过绕过 ZMQ/中继来优化通信。
sglang_omni/pipeline/stage_workers.py
20
MPS 优化：该运行器支持使用 CUDA MPS 的同 GPU 数据并行处理，以便在 GPU 资源充足时提高吞吐量。
docs/basic_usage/mps_dp.md
1-9
资源管理实体
实体	角色	来源
StageWorkerProcessSpec	定义一个操作系统进程运行一个或多个阶段所需的一切	
sglang_omni/pipeline/stage_workers.py
125-129
StageGroup	拓扑组中操作系统进程的生命周期管理器	
sglang_omni/pipeline/stage_workers.py
193-199
StageLaunchConfig	已解析逻辑阶段构建（工厂、路由、扇入）的元数据	
sglang_omni/pipeline/stage_workers.py
38-109
TPLeaderFanout	管理从TP等级0到追随者的工作分配	
sglang_omni/pipeline/tp_control.py
24-29
流程生成流


















资料来源：
sglang_omni/pipeline/stage_workers.py
125-149
 
sglang_omni/pipeline/mp_runner.py
70-191

控制平面消息
通信严格使用数据类进行类型化sglang_omni.proto.messages。

消息类型	方向	目的
SubmitMessage	协调 → 舞台	在入口阶段启动新请求
sglang_omni/proto/messages.py
227-230
DataReadyMessage	舞台 → 舞台	继电器中提供了输出数据的信号。
sglang_omni/proto/messages.py
18-27
StreamMessage	舞台 → 协调	提供部分数据块（例如，音频帧）
sglang_omni/proto/messages.py
203-211
CompleteMessage	舞台 → 协调	表示请求最终完成
sglang_omni/proto/messages.py
172-179
AbortMessage	坐标 → 全部	立即取消请求
sglang_omni/proto/messages.py
158-161
DataAckMessage	舞台 → 舞台	确认已收到并使用数据
sglang_omni/proto/messages.py
102-110
资料来源：
sglang_omni/proto/messages.py
1-240
 
sglang_omni/pipeline/control_plane.py
30-44


调度员
相关源文件
SGLang-Omni 中的调度器负责管道阶段内的执行逻辑。它们管理请求的生命周期、批处理策略以及与模型运行器的交互。虽然该类Stage充当 I/O shell 和进程管理器，但它Scheduler决定了数据如何在模型或函数中流动。

该系统支持多种调度器类型，以适应不同的工作负载：

OmniScheduler：自回归 (AR) 模型的主要调度器，具有 KV 缓存管理和异步解码前瞻功能。
SimpleScheduler：一个用于无状态任务（预处理、编码）的轻量级调度器。
ThreadedSimpleScheduler：SimpleScheduler 的一个变体，用于管理并发请求执行的线程池。
StreamingSimpleScheduler：扩展 SimpleScheduler，用于处理增量流块（例如，Code2Wav、Vocoders）的阶段。
DLLMScheduler：专为扩散语言模型（LLaDA2）而设计。
OmniScheduler
OmniScheduler专为高性能AR推理而设计。它采用组合策略封装上游SGLangScheduler类，无需直接继承。它能够处理诸如TP（张量并行）工作扇出、通过KV缓存分配req_to_token_pool以及连续批处理等复杂任务。

组成和上游集成
它不继承自 `<Object>` sglang.srt.managers.scheduler.Scheduler，而是OmniScheduler通过 `<Object>` 查找上游方法__getattr__并将其绑定到自身的实例。
sglang_omni/scheduling/omni_scheduler.py
4-12
这使得它能够重用 SGLang 的核心调度逻辑（例如get_next_batch_to_run），同时覆盖特定阶段的行为，例如run_batch和。recv_requests 
sglang_omni/scheduling/omni_scheduler.py
164-167

异步解码前瞻
为了降低延迟，OmniScheduler采用了一步前瞻异步解码。这使得调度器能够在 GPU 上启动解码步骤，并立即在 CPU 上开始处理下一批数据的元数据或预填充逻辑，而无需等待 CUDA 内核完成。

快速路径：对于小于 2 的批处理大小async_decode_min_batch_size（默认值），将跳过前瞻操作，以避免低并发性下的固定开销。
sglang_omni/scheduling/omni_scheduler.py
196-198
实施：发布后ModelRunner.execute_launch记录torch.cuda.Event
sglang_omni/model_runner/base.py
83-88
 ModelRunner.execute_resolve然后，在完成步骤之前查询或同步此事件。
sglang_omni/model_runner/base.py
109-116
令牌暂存：为避免阻塞 D2H 转账，令牌 ID 会使用乒乓策略复制到固定主机缓冲区。
sglang_omni/model_runner/base.py
121-145
预填充聚结
调度器支持等待多个预填充请求到达后再启动批处理，从而降低小请求的 GPU 开销。这由prefill_coalesce_requests以下因素控制：prefill_coalesce_wait_ms 
sglang_omni/scheduling/omni_scheduler.py
199-200

门逻辑：门会保持预填充状态，直到请求计数达到上限或最旧的请求超时为止。
sglang_omni/scheduling/omni_scheduler.py
63-66
CLI 控制：用户可以通过--prefill-coalesce-requests以下方式配置这些设置：--prefill-coalesce-wait-ms 
sglang_omni/cli/serve.py
14-31
OmniScheduler 的关键组件
成分	角色
inbox/outbox	用于阶段间通信的标准化 ZMQ 支持的队列
sglang_omni/scheduling/omni_scheduler.py
206-207
tree_cache	管理用于前缀共享的 RadixCache
sglang_omni/scheduling/omni_scheduler.py
180
req_to_token_pool	跟踪正在运行的请求的令牌到键值槽的映射关系
sglang_omni/scheduling/omni_scheduler.py
181
request_builder	转换StagePayload为特定于模型的请求数据
sglang_omni/scheduling/omni_scheduler.py
189
run_batch	调用主执行循环ModelRunner.execute 
sglang_omni/scheduling/omni_scheduler.py
160
资料来源：
sglang_omni/scheduling/omni_scheduler.py
1-207
 
sglang_omni/model_runner/base.py
64-145
 
tests/unit_test/pipeline/test_async_decode.py
1-171
 
sglang_omni/cli/serve.py
14-31

简单调度器和线程调度器
简易调度器
SimpleScheduler用于不需要键值缓存管理的阶段，例如图像编码器或音频编码器。它inbox使用提供的缓存来处理请求。compute_fn 
tests/unit_test/pipeline/test_scheduler.py
117-122

批处理batch_compute_fn：支持通过、max_batch_size和进行动态批处理。max_batch_wait_ms 
tests/unit_test/pipeline/test_scheduler.py
117-122
错误合约：在计算函数失败时，保留批量成功输出，同时发出每个请求的失败消息。
tests/unit_test/pipeline/test_scheduler.py
134-152
ThreadedSimpleScheduler
ThreadedSimpleScheduler提供了一种多线程模型，其中每个请求都由一个单独的线程处理，最多可达 1 个线程max_concurrency。这对于 I/O 密集型任务或需要并行运行的简单同步计算任务非常有用。
tests/unit_test/pipeline/test_scheduler.py
154-188

StreamingSimpleScheduler
SimpleScheduler这是为必须同时处理new_request消息和增量stream_chunk消息的阶段而设计的扩展。
sglang_omni/scheduling/streaming_simple_scheduler.py
34-40

声码器示例：它MossTTSLocalStreamingVocoderScheduler使用此功能来管理持久codec.streaming()会话
sglang_omni/models/moss_tts_local/streaming_vocoder.py
35-66
生命周期钩子：子类实现`LifecycleHooks` on_streaming_new_request、on_stream_chunk`LifecycleHooks` 和 ` on_stream_doneLifecycleHooks` 来管理有状态流。
sglang_omni/scheduling/streaming_simple_scheduler.py
81-120
批处理_can_batch_stream_chunks：如果启用此功能，可以将多个流块批处理在一起。
sglang_omni/scheduling/streaming_simple_scheduler.py
42-45
资料来源：
sglang_omni/scheduling/streaming_simple_scheduler.py
1-173
 
sglang_omni/models/moss_tts_local/streaming_vocoder.py
35-175
 
tests/unit_test/pipeline/test_scheduler.py
116-206

DLLMScheduler
这DLLMScheduler是一个专门用于 LLaDA2 等扩散语言模型的调度器。它处理扩散模型所需的迭代去噪过程，这与 AR 模型逐个标记的生成方式有很大不同。

资料来源：
sglang_omni/scheduling/dllm_scheduler.py
1-50

数据流和实体映射
以下图表说明了高级调度概念与实现这些概念的具体代码实体之间的关系。

应收账款调度管道（OmniScheduler）
此图显示了请求如何从源头流向Stage目标头OmniScheduler，最终到达终端ModelRunner。

















资料来源：
sglang_omni/scheduling/omni_scheduler.py
206-207
 
sglang_omni/scheduling/omni_scheduler.py
164-167
 
sglang_omni/model_runner/base.py
91-120

简单调度器流程与流式调度器流程
SimpleScheduler该图描绘了无状态系统和有状态系统之间的功能差异StreamingSimpleScheduler。






















资料来源：
sglang_omni/scheduling/streaming_simple_scheduler.py
158-173
 
tests/unit_test/pipeline/test_scheduler.py
116-206

错误处理和中止
所有调度器都实现了abort(request_id)停止处理和释放资源的方法。

OmniScheduler：中止事件会被跟踪_aborted_request_ids。如果运行失败，调度器会将上游中止通知转换为阶段输出。_UpstreamAbortSender 
sglang_omni/scheduling/omni_scheduler.py
105-131
StreamingSimpleScheduler：该abort方法清除内部状态并添加 ID，以_aborted_request_ids防止进一步处理延迟的数据块。
sglang_omni/scheduling/streaming_simple_scheduler.py
150-153
像 MOSS 声码器这样的子类会将流槽释放回会话池。
sglang_omni/models/moss_tts_local/streaming_vocoder.py
122-130
内存安全：OmniScheduler用于release_kv_cache确保在请求中止或完成时回收 GPU 内存。
sglang_omni/scheduling/omni_scheduler.py
39
资料来源：
sglang_omni/scheduling/omni_scheduler.py
105-131
 
sglang_omni/scheduling/streaming_simple_scheduler.py
150-153
 
sglang_omni/models/moss_tts_local/streaming_vocoder.py
122-130


舞台间沟通
相关源文件
SGLang-Omni 中的阶段间通信被分为用于信令的控制平面和用于高性能张量传输的数据平面。这种架构确保请求编排的开销不会成为多模态流水线高吞吐量需求的瓶颈。

通信架构概述
该系统采用解耦方式，其中小型控制消息通过 ZeroMQ (ZMQ) 路由，而大型数据负载（例如隐藏状态或音频张量）则通过专用中继后端传输。它CommEngine充当各阶段与这些系统交互的主要接口。
sglang_omni/comm/engine.py
94-99

系统流程图
Coordinator下图说明了运行Stage时和系统之间的交互Relay。

标题：阶段间控制和数据流



资料来源： 
sglang_omni/comm/engine.py
172-202
 
sglang_omni/relay/base.py
13-25
 
sglang_omni/proto/messages.py
18-27

控制平面（ZMQ + msgpack）
控制平面处理请求生命周期事件以及实例之间的信令Coordinator。Stage所有消息都使用序列化msgpack。msgspec 
sglang_omni/pipeline/control_plane.py
47-55

套接字基础架构
控制平面利用了由以下机构管理的几种 ZMQ 套接字类型ControlPlaneContext 
sglang_omni/pipeline/control_plane.py
58-78
：

PushSocket：用于舞台向特定目的地发送消息
sglang_omni/pipeline/control_plane.py
79-106
PullSocket：用于各阶段接收入站控制消息
sglang_omni/pipeline/control_plane.py
121-164
PubSocket/SubSocket：用于AbortMessage在所有阶段广播信号
sglang_omni/pipeline/control_plane.py
166-222
消息类型（sglang_omni.proto）
消息被定义为数据类，或者msgspec.Struct可以与字典相互转换以进行传输。
sglang_omni/proto/messages.py
1-203

消息类型	方向	目的
SubmitMessage	协调员 → 舞台	在入口阶段发起新的请求
sglang_omni/proto/messages.py
255-263
DataReadyMessage	舞台 → 舞台	表示上游级已向中继器写入数据。包含data_ref 
sglang_omni/proto/messages.py
18-27
CompleteMessage	舞台 → 协调员	表示终端阶段已完成处理或失败的信号
sglang_omni/proto/messages.py
172-179
StreamMessage	舞台 → 协调员	发送部分输出（例如，音频片段）以进行实时流传输
sglang_omni/proto/messages.py
203-211
AbortMessage	协调员 → 所有阶段	广播信号以停止处理特定数据。request_id 
sglang_omni/proto/messages.py
158-164
DataAckMessage	舞台 → 舞台	接收方完成一个数据平面对象的操作，以释放发送方资源
sglang_omni/proto/messages.py
102-111
KVTransferPrepareMessage	舞台 → 舞台	协商各阶段之间的分页 KV 缓存传输
sglang_omni/proto/kv_transfer.py
42-51
资料来源： 
sglang_omni/proto/messages.py
1-274
 
sglang_omni/pipeline/control_plane.py
1-44
 
sglang_omni/proto/kv_transfer.py
42-65

数据平面中继后端
基Relay类提供了一个抽象接口，用于跨进程边界传输张量。
sglang_omni/relay/base.py
13-25

支持的后端
CudaIpcRelay( cuda_ipc)：使用 CUDA IPC 句柄进行快速跨 GPU 传输。它使用有界发送端 GPU 插槽池，并通过 P2P 访问检查提供支持。_ensure_peer_access 
sglang_omni/relay/cuda_ipc.py
110-129
NixlRelay( nixl)：基于 RDMA 的高性能传输，用于NixlAgent跨节点或跨设备移动
sglang_omni/relay/nixl.py
175-218
ShmRelay( shm)：使用POSIX共享内存在同一主机上的进程之间进行基于CPU的数据传输。
MooncakeRelay（mooncake）：与 Mooncake 分布式 KV 缓存系统集成
sglang_omni/relay/mooncake.py
9-11
数据编组（sglang_omni.comm.stage_io）
张量在传输之前从复杂对象（如字典或列表）中提取出来，并在接收端恢复：

extract_cuda_tensors递归地查找 CUDA 张量并将其替换为占位符
sglang_omni/comm/stage_io.py
81-91
restore_tensors将接收到的张量重新插入到原始对象结构中
sglang_omni/comm/stage_io.py
110-120
serialize_direct_cuda_ipc_payload：一种优化方法，可将 CUDA 张量StagePayload及其相关的 CUDA 张量打包成单个二进制结构，用于进程间通信 (IPC)。
sglang_omni/comm/stage_io.py
138-163
资料来源： 
sglang_omni/relay/cuda_ipc.py
2-180
 
sglang_omni/comm/stage_io.py
56-120
 
sglang_omni/relay/nixl.py
175-218

本地调度和路线规划
它CommRouter根据平台拓扑结构和器件放置位置确定最佳传输机制。

实施：运输选择
解析CommRouter每条TransportKind边 [sglang_omni/comm/router.py]：

LOCAL_OBJECT用于同一流程中位于同一地点的阶段
sglang_omni/comm/data_ref.py
28
CUDA_IPC：用于P2P可用时同一节点GPU之间的数据传输
sglang_omni/comm/data_ref.py
30
SHM：CPU 目标或 CUDA IPC 不可用时的备用方案
sglang_omni/comm/data_ref.py
29
MOONCAKE用于远程（跨节点）边
sglang_omni/comm/data_ref.py
31
标题：通信路径解析

















资料来源： 
sglang_omni/comm/router.py
10-30
 
sglang_omni/comm/data_ref.py
27-32
 
sglang_omni/relay/cuda_ipc.py
110-129

KV缓存传输
对于多级LLM流水线，CommEngine支持专门的分页KV缓存传输，以避免冗余计算。这涉及KVPool注册和协调的页面移动。
sglang_omni/comm/kv_transfer.py
24-29

转移生命周期
准备：接收阶段需要prepare_kv_destination在其本地预留空间。KVPool 
sglang_omni/comm/engine.py
330-345
信号：AKVTransferPrepareMessage通过控制平面发送。
sglang_omni/proto/kv_transfer.py
42-51
运动：CommEngine执行put_kv_pages或get_kv_pages通过接力
sglang_omni/comm/engine.py
384-405
资料来源： 
sglang_omni/comm/engine.py
330-450
 
sglang_omni/comm/kv_transfer.py
1-60
 
tests/unit_test/pipeline/test_kv_transfer.py
47-98

代码实体映射
逻辑概念	代码实体	文件路径
通信引擎	CommEngine	
sglang_omni/comm/engine.py
94
控制平面插座	PushSocket/PullSocket	
sglang_omni/pipeline/control_plane.py
79-121
数据参考	DataRef	
sglang_omni/comm/data_ref.py
64
继电器基类	Relay	
sglang_omni/relay/base.py
13
RDMA 连接	Connection	
sglang_omni/relay/nixl.py
30
KV缓存池	KVPool	
sglang_omni/comm/kv_transfer.py
27
数据有效载荷	StagePayload	
sglang_omni/proto/request.py
60
CUDA IPC 管理器	CudaIpcRelay	
sglang_omni/relay/cuda_ipc.py
22
资料来源： 
sglang_omni/comm/engine.py
94-139
 
sglang_omni/pipeline/control_plane.py
79-222
 
sglang_omni/comm/data_ref.py
64-100
 
sglang_omni/proto/request.py
60-95
 
sglang_omni/relay/cuda_ipc.py
22-180


模型运行层
相关源文件
模型运行器层sglang_omni作为多模态模型的执行引擎。它封装了 SGLang 结构化运行时 (SRT)，ModelRunner为全模态特性提供专门支持，例如异步解码前瞻、多模态嵌入注入、音频编码预测以及对共置阶段的精确 GPU 内存管理。

概述和模型工作器
这ModelWorker是模型执行过程的入口点。它初始化模型配置，配置后端策略（如 FP8 或 MoE 内核），并实例化运行器。

主要职责
架构覆盖：通过交换和相关的注意力参数，将复杂的多模态架构（例如Qwen3OmniTalker，，，WhisperForConditionalGeneration）映射到其组成 LLM 配置。MossTTSLocalSGLangModelhf_text_config
sglang_omni/model_runner/model_worker.py
35-47
 
sglang_omni/model_runner/model_worker.py
115-148
后端策略：根据 GPU 架构和量化状态确定最佳的 MoE 和 GEMM 后端。
sglang_omni/model_runner/model_worker.py
149-169
内存管理：使用total_gpu_memory_fraction属性处理位于同一位置的进程的 GPU 内存分割，以确保多个 SGLang 引擎可以在同一设备上共存。
sglang_omni/model_runner/model_worker.py
27-32
 
sglang_omni/model_runner/model_worker.py
62
模型工作进程初始化流程
下图说明了如何ModelWorker引导执行环境并将自定义架构注册到全局 SGLang 注册表中。

“模型工作者初始化和注册”

















资料来源：
sglang_omni/model_runner/model_worker.py
50-81
 
sglang_omni/model_runner/model_worker.py
82-109
 
sglang_omni/model_runner/sglang_model_runner.py
158-180

SGLModelRunner 和注册
这SGLModelRunner是一个对 SGLang 的轻量级封装ModelRunner。它的主要作用是将sglang_omni模型实现与 SGLang 执行循环连接起来。

自定义模型注册
在初始化期间，将特定的模型类SGLModelRunner注入到 SGLang 的全局变量中。这使得 SGLang 可以使用HuggingFace 中的架构字符串（例如，`models.htaccess`、 ` models.htaccess` 和 `models.htaccess`）来实例化这些模型。sglang_omniModelRegistryQwen3OmniThinkerForCausalLMMossTTSDelaySGLangModelconfig.json 
sglang_omni/model_runner/sglang_model_runner.py
158-185

托管机房的GPU内存预算
当多个 SGLang 阶段位于同一 GPU 上时，标准内存分析（使用全局可用内存）会失败，因为并发进程可能会同时加载权重。SGLModelRunner使用以下方法_OmniKVCacheConfigurator实现专门的分析逻辑：

方法	逻辑
_profile_available_bytes	total_gpu_memory_fraction如果设置了该参数，则派发给进程感知或增量感知分析。
sglang_omni/model_runner/sglang_model_runner.py
71-74
过程感知	使用 NVML 算法get_process_gpu_memory_bytes查找当前 PID 使用的确切内存，并将其从分配的阶段预算中减去。
sglang_omni/model_runner/sglang_model_runner.py
135-155
三角洲感知	测量加载阶段的内存变化（post_model_load_memoryvs ），以估算键值缓存余量。pre_model_load_memory
sglang_omni/model_runner/sglang_model_runner.py
98-133
资料来源：
sglang_omni/model_runner/sglang_model_runner.py
44-70
 
sglang_omni/model_runner/sglang_model_runner.py
71-96
 
sglang_omni/model_runner/sglang_model_runner.py
98-155

基础模型运行器和异步解码
基类实现了一个适用于所有自回归（AR）模型的ModelRunner共享管道。它引入了一种异步解码（一步前瞻）机制，使GPU执行与主机端输出处理能够重叠进行。execute()

异步解码状态机
运行器维护一个_PendingStep跟踪 GPU 上已启动但尚未被主机使用的解码步骤的进程。
sglang_omni/model_runner/base.py
65-81

execute_launch：将前向传播和采样操作加入 GPU 队列，并记录设备事件。
sglang_omni/model_runner/base.py
83-90
execute_resolve：等待事件发生并处理结果。它使用固定的乒乓缓冲区策略（_pinned_pingpong）来确保主机读取操作永远不会与下一步的异步 D2H 复制操作发生冲突。
sglang_omni/model_runner/base.py
139-155
暂存：采样后，令牌会立即暂存到固定的主机内存中（_stage_token_ids），以避免在.tolist()调用期间阻塞可分页的 D2H 传输。
sglang_omni/model_runner/base.py
121-134
资料来源：
sglang_omni/model_runner/base.py
65-155
 
tests/unit_test/pipeline/test_async_decode.py
2-11

专业模型跑者
自定义模型运行器扩展了基础模型ModelRunner，以处理特定于模态的逻辑。

ThinkerModelRunner
Qwen3-Omni 的思考者阶段使用此功能来处理多模态输入。

嵌入注入：拦截custom_prefill_forward并替换占位符标记（图像、视频、音频）为相应的高维嵌入。
sglang_omni/model_runner/thinker_model_runner.py
39-54
多模态光标跟踪：管理_omni_consumed光标，确保在分块预填充期间正确切片嵌入内容。
sglang_omni/model_runner/thinker_model_runner.py
100-131
Deepstack 支持：处理 Deepstack 架构的残余视觉嵌入
sglang_omni/model_runner/thinker_model_runner.py
49-54
QwenTalkerModelRunner
由 Qwen3-Omni 说话器阶段用于生成语音代码。

反馈与文本融合：以先进先出 (FIFO) 的方式使用音频反馈和文本标记来驱动增强现实 (AR) 生成。
tests/unit_test/qwen3_omni/test_talker.py
88-114
预测器 CUDA 图：_PredictorDecodeGraph用于捕获和重放音频预测器前向传播（码本 1-7），以实现低延迟生成
sglang_omni/models/qwen3_omni/components/talker.py
58-91
“模型运行器执行路径”
















资料来源：
sglang_omni/model_runner/thinker_model_runner.py
23-55
 
sglang_omni/models/qwen3_omni/talker_model_runner.py
29-34
 
tests/unit_test/qwen3_omni/test_talker.py
48-54

FP8 后端配置
该ModelWorker策略根据平台功能和模型架构，为 FP8 应用特定的后端策略。

案件	政策结果
Qwen3OmniTalker (BF16)	默认设置flashinfer_cutlass为教育部
sglang_omni/models/qwen3_omni/components/talker.py
43-44
Qwen3OmniTalker（FP8）	cutlass为确保与FP8内核的兼容性，需要对MoE和triton密集型GEMM进行加权。
sglang_omni/model_runner/model_worker.py
149-169
Qwen3OmniThinker（FP8）	用于cutlassMoE 但保留了auto密集 GEMM
sglang_omni/model_runner/model_worker.py
149-169
资料来源：
sglang_omni/model_runner/model_worker.py
149-169
 
sglang_omni/quantization/fp8.py
1-50