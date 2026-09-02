# 03 配置系统与部署拓扑：三份 PipelineConfig、placement 校验与内存契约

> 主角：`sglang_omni/models/qwen3_omni/config.py`（403 行）、
> `sglang_omni/models/qwen3_omni/placement.py`（147 行）、
> `sglang_omni/config/schema.py`（EngineStageConfig/StageConfig/PlacementConfig 等）、
> `sglang_omni/models/qwen3_omni/stages.py` 的内存契约部分。

---

## 1. StageConfig：把一个阶段声明成数据

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

## 2. 七个阶段的逐字声明

### 2.1 preprocessing（`config.py:41-64`）

- 文本拓扑：`next = [有输入的 encoder..., mm_aggregate]`，路由
  `resolve_preprocessing_next_stages`；语音拓扑：`next = [encoders..., thinker]`，
  有音频输出时再追加 `talker_ar`（`resolve_preprocessing_next_stages_speech`）。
- 注意：**路由函数返回的是"逻辑上有出边的目标"**，真正发不发 payload 还要经
  project_payload；`_encoder_stages_with_model_inputs` 保证纯文本请求不会发往 encoder。
- `FactoryArgs(max_seq_len=8192)`：preprocessor 据此做 prompt 长度预检
  （`preprocessor.validate_prompt_seq_len`：`prompt + max_new_tokens >= max_seq_len` 直接拒）。

### 2.2 image_encoder / audio_encoder（`config.py:67-92`）

两者通过 `_encoder_join_edges(speech_enabled)` 共享出边定义：

- 文本：`next="mm_aggregate"`，投影 `project_encoder_to_mm_aggregate`；
- 语音：`next=[thinker, talker_ar]`，路由 `resolve_encoder_next_stages`
  （**不要音频输出时只发 thinker**），投影两张：join 用
  `project_encoder_to_mm_aggregate`，talker 用 `project_encoder_to_talker_ar`。

audio_encoder 特有的两个开关：
`factory.enable_layer_cuda_graph=True`（层间图，04 篇）与
`disable_direct_cuda_ipc_payload=True`。

### 2.3 mm_aggregate（仅文本拓扑，`config.py:95-108`）

`wait_for=[preprocessing, image_encoder, audio_encoder]` +
`resolve_mm_aggregate_wait_sources`（动态收窄）+ `merge_for_thinker` +
`next="thinker"`。它是个纯聚合 + 转发的 CPU/GPU 轻阶段（工厂 `_identity`）。

### 2.4 thinker（`config.py:131-158`，EngineStageConfig）

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

### 2.5 decode（`config.py:161-169`）

`terminal=True` + `can_accept_stream_before_payload=True`，工厂
`create_decode_executor` → `StreamingDetokenizeScheduler`。它同时接收：
payload（thinker 的 next）+ 每 token 流 + stream_done 信号，三者缺一不可
（10 篇详述三者如何对齐）。

### 2.6 talker_ar（`config.py:171-182`，EngineStageConfig）

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

### 2.7 code2wav（`config.py:185-192`）

`gpu_memory_fraction=0.02`（声码器很小）+ `terminal=True` +
`can_accept_stream_before_payload=True`，工厂
`components.code2wav_scheduler.create_code2wav_scheduler`。

---

## 3. 三份配置的差异矩阵

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

## 4. placement.py：Qwen 专属拓扑合法性

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

## 5. 内存契约：`_apply_colocated_ar_memory_contract`（stages.py:118-160）

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

## 6. 环境默认值与 FP8

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

## 7. 启动装配链（一图流）

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

## 8. 小结

- 拓扑 = 数据；阶段 = 数据；连边、join、投影、流边全是数据——这让"改部署"
  从改代码变成改配置，也让 placement 校验成为可能。
- 共置模式的三件事必须同时成立才安全：单卡绑定、显式内存预算、禁副本/TP；
  内存契约函数是唯一的预算推导点。
- 所有"魔法数字"（8192/32768/0.05/0.02/8/0）在源码注释里都有失败案例背书，
  读配置时务必连注释一起读。
