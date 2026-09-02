Qwen3-Omni
相关源文件
Qwen3-Omni 是一款多模态大型语言模型，能够处理文本、图像、音频和视频输入，并生成同步的文本和语音输出。SGLang-Omni 将 Qwen3-Omni 实现为一个模块化的多阶段流水线，支持在单个或多个 GPU 上灵活部署，支持 FP8 精度，并通过“部分启动”策略实现低延迟的流式语音生成。

管道架构
Qwen3-Omni 的实现分为两种主要拓扑结构：一种是6 阶段纯文本流水线，另一种是8 阶段语音流水线。 
sglang_omni/models/qwen3_omni/config.py
195-215
。

8 阶段语音处理流程数据流
语音管道使用流式隐藏状态接口连接“思考者”（核心 LLM）和“说话者”（语音生成器）。

























资料来源：
sglang_omni/models/qwen3_omni/config.py
41-192
，
sglang_omni/models/qwen3_omni/request_builders.py
81-104
，
sglang_omni/models/qwen3_omni/stages.py
19-21

管道阶段定义
预处理：使用 CPU 端进行分词和媒体加载Qwen3OmniPreprocessor。它根据输入方式确定下一个活动阶段。
sglang_omni/models/qwen3_omni/config.py
41-64
。
image_encoder：使用 处理图像并将其转换为嵌入Qwen3OmniImageEncoder。它包含一项优化，将 替换Conv3d为 ，Linear从而实现约 7-15 倍的速度提升。PatchEmbed 
sglang_omni/models/qwen3_omni/config.py
67-78
。
audio_encoder：使用以下方式将音频片段处理成嵌入：Qwen3OmniAudioEncoder 
sglang_omni/models/qwen3_omni/config.py
81-92
。
mm_aggregate：将文本和多模态嵌入合并成一个统一的序列，供思考者使用merge_for_thinker 
sglang_omni/models/qwen3_omni/config.py
95-128
 
sglang_omni/models/qwen3_omni/merge.py
35
。
思考器：主要的自回归LLM（Qwen3-Omni-30B）生成文本标记和隐藏状态。在语音模式下，它同时向两者输出数据流talker_ar。decode 
sglang_omni/models/qwen3_omni/config.py
131-158
。
解码StreamingDetokenizeScheduler：用于向用户传输文本的终端阶段
sglang_omni/models/qwen3_omni/config.py
161-169
。
talker_ar：语音自回归阶段，用于根据思考者的隐藏状态预测声学编码。它使用QwenTalkerScheduler和QwenTalkerModelRunner 
sglang_omni/models/qwen3_omni/config.py
171-182
 
sglang_omni/models/qwen3_omni/talker_model_runner.py
16
。
code2wav：一个终端阶段，通过以下方式将声学代码转换为 PCM 音频：Code2WavScheduler 
sglang_omni/models/qwen3_omni/config.py
185-192
 
sglang_omni/models/qwen3_omni/components/code2wav_scheduler.py
90
。
关键组成部分
思考者-说话者界面
Thinker 和 Talker 通过流式机制耦合。Thinker 会hidden_states为每个生成的 token 发出状态。这些状态会被投影并传递给 Talker。

TalkerPrefillBuilder：通过将思考者的提示嵌入投影到说话者的潜在空间中，重构说话者的初始上下文。
sglang_omni/models/qwen3_omni/components/talker_prefill.py
17
。
部分启动策略：为了缩短首次音频播放时间 (TTFA)，当思考者数据块达到最低可用数量时，说话者即可开始生成音频代码。该MIN_PARTIAL_START_CHUNKS常量设置为 3。
sglang_omni/models/qwen3_omni/config.py
21
。
PendingTextTensorQueue：一个专门用于管理待 Talker 使用的隐藏状态张量的队列
sglang_omni/models/qwen3_omni/pending_text_queue.py
24-27
。
FIFO解码交接：QwenTalkerModelRunner实现一个before_decode钩子，以确保说话者在继续之前已接收到投影文本嵌入和声音反馈。
sglang_omni/models/qwen3_omni/talker_model_runner.py
45-66
。
M-RoPE 指数计算
Qwen3-Omni 采用多模态旋转位置嵌入 (M-RoPE)。该实现确保能够rope_scaling正确处理多模态输入，支持交错文本、图像和视频的时间和空间维度。它Qwen3OmniPipelineState负责管理整个流水线中的 M-RoPE 索引。
sglang_omni/models/qwen3_omni/payload_types.py
38
多模态旋转位置的计算逻辑封装在……中mrope_positions.py 
sglang_omni/models/qwen3_omni/mrope_positions.py
1-50
。

配置和部署
集中式与分散式
SGLang-Omni 支持 Qwen3-Omni 的两种主要部署模式：

共置：多个阶段共享单个 GPU。Qwen3OmniSpeechColocatedPipelineConfig这通过将thinker、talker_ar和放置code2wav在 GPU 0上来实现。
sglang_omni/models/qwen3_omni/config.py
210-215
OMP_NUM_THREADS=8应用环境默认值以防止 CPU 过度分配
sglang_omni/models/qwen3_omni/config.py
34-38
。
分散式：阶段分布在多个 GPU 上。将Qwen3OmniSpeechPipelineConfig任务分配thinker给 GPU 0，将talker_ar链分配给 GPU 1。
sglang_omni/models/qwen3_omni/config.py
202-207
。
FP8 支持
该模型支持 AR 阶段（思考者和谈话者）的原生 FP8 执行。

默认情况下，环境SGLANG_JIT_DEEPGEMM_PRECOMPILE变量已设置，0以防止首次请求期间出现过长的预编译延迟。
sglang_omni/models/qwen3_omni/config.py
28
。
该ModelWorker处理器处理子模型（例如Qwen3OmniTalker）的量化归一化和架构覆盖。Qwen3OmniThinkerForCausalLM 
sglang_omni/model_runner/model_worker.py
32-43
。
内存管理和 CUDA 图
AR 阶段的内存预算使用以下方式进行管理total_gpu_memory_fraction。

encoder_mem_reserve：对于共置部署，会预留一部分 GPU 内存给图像和音频编码器，以防止 SGLang 分配整个 KV 缓存。
sglang_omni/models/qwen3_omni/stages.py
133-152
。
Code2Wav CUDA 图：Code2WavCudaGraphRunner为流式声码器提供精确形状的图加速。
sglang_omni/models/qwen3_omni/components/code2wav_cuda_graph.py
143-158
total_gpu_memory_fraction分配图缓冲区需要明确的预算。
sglang_omni/models/qwen3_omni/components/code2wav_scheduler.py
108-120
。
输出重叠：Code2WavScheduler默认启用输出重叠，在 GPU 计算下一个数据块的同时，将音频窗口异步复制到固定的暂存缓冲区中。
sglang_omni/models/qwen3_omni/components/code2wav_scheduler.py
163-164
。
实施细节
请求构建和投影
当数据在各个阶段之间流动时，StagePayload对象会被“投影”，以去除不必要的张量并减少 IPC 开销。

project_thinker_to_decode：在将有效负载发送到文本反分词器之前，移除复杂的多模态嵌入和隐藏状态。
sglang_omni/models/qwen3_omni/request_builders.py
135-161
。
project_talker_to_code2wav：将说话者的输出转换为轻量级锁存器；实际的音频编码张量通过流数据平面到达。
sglang_omni/models/qwen3_omni/request_builders.py
164-170
。
Thinker 多模态嵌入注入
该函数ThinkerModelRunner在预填充阶段执行自定义多模态嵌入注入。它识别图像、视频和音频的占位符标记位置，并将其替换为编码器阶段生成的实际嵌入。
sglang_omni/model_runner/thinker_model_runner.py
77-178
。

代码实体协会





























资料来源：
sglang_omni/models/qwen3_omni/payload_types.py
38
，
sglang_omni/models/qwen3_omni/components/talker_prefill.py
17
，
sglang_omni/models/qwen3_omni/talker_model_runner.py
16
，
sglang_omni/models/qwen3_omni/pending_text_queue.py
24-27

使用示例
管道配置
管道是通过组合多个StageConfig对象来定义的。

# Example of defining the thinker stage
_thinker_stage(
    gpu=0, 
    speech_enabled=True, 
    process="thinker_process"
)
资料来源：
sglang_omni/models/qwen3_omni/config.py
131-158

说话者反馈交接
它QwenTalkerModelRunner负责管理投影文本和声音反馈的同步切换。

def before_decode(self, requests):
    # Ensure feedback and text are ready in FIFO queues
    if not self._requests_ready_for_decode(requests):
        raise RuntimeError("Talker decode reached model runner without ready feedback/text input")
    self.model.prepare_decode_buffers(requests)
    self._write_feedback_buffers(requests)
资料来源：
sglang_omni/models/qwen3_omni/talker_model_runner.py
45-66