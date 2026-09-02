Voxtral TTS 和 Qwen3-TTS
相关源文件
本页详细介绍了 SGLang-Omni 中两种主要的离散标记 TTS 系列的实现：Voxtral TTS和Qwen3-TTS。这两个模型都采用了多级流水线，其中自回归 (AR) 转换器生成声学标记，然后由声码器将其转换为音频波形。

1. Voxtral TTS
Voxtral-4B-TTS 是 Mistral AI 开发的一款基于 Ministral-3B 主干网的文本转语音模型。它使用一组预设的命名语音，可以生成 9 种语言的 24 kHz 语音。

1.1 流水线架构
Voxtral流程由三个阶段组成：preprocessing，，tts_generation和vocoder 
sglang_omni/models/voxtral_tts/config.py
27-31

阶段	责任	实体
预处理	文本规范化和铁拳分词。	create_preprocessing_executor 
sglang_omni/models/voxtral_tts/pipeline/stages.py
93-94
TTS 生成	使用 SGLang 生成 AR 声学标记。	create_generation_executor 
sglang_omni/models/voxtral_tts/pipeline/stages.py
168-169
Vocoder	基于声学代码的波形合成。	create_vocoder_executor 
sglang_omni/models/voxtral_tts/pipeline/stages.py
198
资料来源：
sglang_omni/models/voxtral_tts/config.py
27-31
 
sglang_omni/models/voxtral_tts/pipeline/stages.py
93-198

1.2. Voxtral 数据流和请求映射
该模块处理命名语音所需的专用嵌入逻辑。Voxtral 使用存储在检查点中的VoxtralTTSModelRunner特定语音（例如）的固定嵌入。cheerful_female

Voxtral 系统到代码实体映射
下图说明了如何将特定语音的自然语言请求映射到 SGLang 执行实体。

“Voxtral实体映射”













资料来源：
sglang_omni/models/voxtral_tts/pipeline/stages.py
102-142
 
sglang_omni/models/voxtral_tts/pipeline/engine_builder.py
12
 
sglang_omni/models/voxtral_tts/io.py
16

1.3. 主要实施细节
标记化：使用 Mistral 的Tekken标记器mistral_common 
sglang_omni/models/voxtral_tts/pipeline/stages.py
34-42
语音嵌入：通过定位提示信息中的位置VoxtralTTSModelRunner，将语音嵌入信息注入到输入流中。audio_token_id
tests/unit_test/voxtral_tts/test_pipeline.py
65-79
基数缓存隔离：为了防止不同语音之间发生缓存冲突，Voxtral 请求在SGLang 请求字段voxtral_voice:中使用命名空间进行隔离。extra_key
tests/unit_test/voxtral_tts/test_pipeline.py
89-91
Backbonellama ：基于带有自定义类的架构构建VoxtralSGLangTTSModel，用于 SGLang 集成
sglang_omni/models/voxtral_tts/pipeline/stages.py
199-200
GEMM 优化：Voxtral 能够inductor_config.max_autotune_gemm针对特定形状优化 Triton/ATEN 内核
sglang_omni/models/voxtral_tts/pipeline/stages.py
158-161
资料来源：
sglang_omni/models/voxtral_tts/pipeline/stages.py
34-207
 
tests/unit_test/voxtral_tts/test_pipeline.py
53-92

2. Qwen3-TTS
Qwen3-TTS 是一款 12Hz 帧速率的离散 TTS 模型，支持快速语音克隆、VoiceDesign 和 CustomVoice 模式。
sglang_omni/models/qwen3_tts/request_builders.py
44-46

2.1. 运行模式
Qwen3-TTS 支持在配置和预处理过程中解决多个任务：

基础（语音克隆）：使用上下文学习 (ICL) 和通过参考音频编码器提取 x 向量。
sglang_omni/models/qwen3_tts/request_builders.py
44
CustomVoice：使用模型路径中已识别的预置扬声器表
sglang_omni/models/qwen3_tts/request_builders.py
45
语音设计：根据文本指令生成语音
sglang_omni/models/qwen3_tts/request_builders.py
46
资料来源：
sglang_omni/models/qwen3_tts/request_builders.py
43-47
 
sglang_omni/models/qwen3_tts/config.py
12-17

2.2. 模型运行器和代码预测器
它Qwen3TTSModelRunner管理AR生成
sglang_omni/models/qwen3_tts/model_runner.py
17-18
Qwen3-TTS 的一个独特之处在于Qwen3TTSTalker，它能够根据第一层的语义标记和主干隐藏状态，分层地预测多个码本层。

Qwen3-TTS CUDA 图加速
使用以下方法加速代码预测器链（语言模型头部、采样和嵌入）：_PredictorDecodeGraph 
sglang_omni/models/qwen3_tts/sglang_model.py
67-75
这样就将每个令牌的预测器链捕获为一个 CUDA 图，从而最大限度地减少解码多个码本期间的主机端开销。
sglang_omni/models/qwen3_tts/sglang_model.py
134-146

Qwen3-TTS 执行流程
此图显示了从 HTTP 请求到专用请求的过渡Qwen3TTSModelRunner。

“Qwen3-TTS 代码执行流程”
















资料来源：
sglang_omni/models/qwen3_tts/stages.py
118-154
 
sglang_omni/models/qwen3_tts/sglang_model.py
153-158

2.3 取样和播种
Qwen3-TTS采用双种子策略。单个请求种子通过某种方式拆分为语义种子和子说话人采样种子，derive_qwen3_tts_sampling_seeds以确保在主AR骨干网和代码预测器上生成可重复的语音。
sglang_omni/models/qwen3_tts/request_builders.py
94-101
sample_from_sorted_logprobs_with_seed_small_k为了提高性能，当 top-k 值较小时，它采用自定义的 Triton 内核进行带种子 Gumbel 采样。
sglang_omni/models/qwen3_tts/sampling_kernels.py
95-100

资料来源：
sglang_omni/models/qwen3_tts/request_builders.py
94-101
 
sglang_omni/models/qwen3_tts/sampling_kernels.py
95-136

2.4. Logit 整形和重复惩罚
该Qwen3TTSModelRunner实现针对重复惩罚和词元抑制进行了优化的logit整形。与可能在主机上构建索引的基础运行器不同，它Qwen3TTSModelRunner维护设备驻留掩码（_shape_masks），以便直接在GPU上应用惩罚。
sglang_omni/models/qwen3_tts/model_runner.py
148-162
它使用一个方法_mask_fingerprint来跟踪各个步骤中的请求状态，并根据每一行是否都恰好增加了一个令牌来确定掩码是否需要重建。
sglang_omni/models/qwen3_tts/model_runner.py
119-146

资料来源：
sglang_omni/models/qwen3_tts/model_runner.py
119-200

3. 配置比较
这两个模型均通过 YAML 进行配置，并通过同一个 OpenAI 兼容端点提供服务。

特征	Voxtral TTS	Qwen3-TTS
分词器	铁拳（mistral-common）	Qwen3TTSTokenizer
语音克隆	否（仅限指定配音演员）	是的（x向量和ICL）
帧率	25帧/秒	12帧/秒
采样率	24 kHz	24 kHz
配置类	VoxtralTTSPipelineConfig	Qwen3TTSPipelineConfig
资料来源：
sglang_omni/models/voxtral_tts/config.py
15
 
sglang_omni/models/qwen3_tts/config.py
20-23
 
sglang_omni/models/voxtral_tts/pipeline/stages.py
97-101

4. 技术数据流概要
请求接收：/v1/audio/speech接收请求。
预处理阶段：
Voxtral：create_preprocessing_executor使用 Tekken 分词器对文本进行编码并准备VoxtralTTSState 
sglang_omni/models/voxtral_tts/pipeline/stages.py
102-142
Qwen3：create_preprocessing_executor解析任务类型并通过以下方式处理参考音频预处理preprocess_qwen3_tts_payload 
sglang_omni/models/qwen3_tts/stages.py
118-132
它用于ReferenceEncodeService对昂贵的参考音频 X 向量提取进行去重和缓存。
sglang_omni/models/qwen3_tts/request_builders.py
159-164
AR生成阶段：
用于与 SGLang 引擎交互的OmniScheduler用途VoxtralTTSModelRunnerQwen3TTSModelRunner
sglang_omni/models/voxtral_tts/pipeline/stages.py
168-187
 
sglang_omni/models/qwen3_tts/stages.py
135-154
生成的代码会被收集并存储在请求中。output_codes 
sglang_omni/models/qwen3_tts/model_runner.py
71-80
声码器阶段：
Voxtral：声学代码通过以下方式转换为波形create_vocoder_executor 
sglang_omni/models/voxtral_tts/pipeline/stages.py
198
Qwen3：Qwen3TTSStreamingVocoderScheduler根据预测的码本层合成最终波形
sglang_omni/models/qwen3_tts/stages.py
189-204
资料来源：
sglang_omni/models/voxtral_tts/pipeline/stages.py
93-221
 
sglang_omni/models/qwen3_tts/stages.py
118-204
 
sglang_omni/models/qwen3_tts/request_builders.py
159-164
 
sglang_omni/models/qwen3_tts/model_runner.py
71-80