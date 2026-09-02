# `sgl-omni serve` 命令解析与运行全链路详解

> 剖析对象：
>
> ```bash
> uv run python3 -m sglang_omni.cli serve \
>   --model-path /home/zy0057/work_dev/wqq/nfs/test_cc_dev/git_source/Qwen3-TTS-12Hz-1.7B-Base \
>   --config examples/configs/qwen3_tts_1_7b.yaml \
>   --port 8000
> ```
>
> 本文基于对源码的逐行追踪，完整还原这条命令从进程启动到 HTTP 服务就绪的每一个环节。
> 配套总览文档见同目录 `api_cli.md`。

---

## 目录

1. [命令剖析：三段式结构](#1-命令剖析三段式结构)
2. [第一层：Typer CLI 解析](#2-第一层typer-cli-解析)
3. [第二层：配置解析（patch 体系）](#3-第二层配置解析patch-体系)
4. [第三层：launch_server 启动流水线](#4-第三层launch_server-启动流水线)
5. [衔接：worker 进程内的阶段装配](#5-衔接worker-进程内的阶段装配)
6. [第四层：三阶段工厂与 SGLang 引擎构建](#6-第四层三阶段工厂与-sglang-引擎构建)
7. [第五层：HTTP 服务与 uvicorn](#7-第五层http-服务与-uvicorn)
8. [全链路时序总览](#8-全链路时序总览)
9. [运行期：一条 TTS 请求怎么走](#9-运行期一条-tts-请求怎么走)
10. [失败模式与环境变量速查](#10-失败模式与环境变量速查)
11. [附录：关键源文件索引](#11-附录关键源文件索引)

---

## 1. 命令剖析：三段式结构

这条命令由三个层次组成，每层职责不同：

```
uv run  python3 -m sglang_omni.cli  serve  --model-path ... --config ... --port 8000
└─┬──┘  └───────────┬──────────────┘ └─┬──┘ └────────────────┬─────────────────┘
  │                 │                  │                     │
  环境启动器      Python 模块入口    Typer 子命令        serve 的参数
```

### 1.1 `uv run` —— 环境层

`uv run` 不参与任何业务逻辑。它读取项目 `pyproject.toml` 声明的虚拟环境（必要时自动同步依赖），在**该环境的解释器**中执行后面的命令。等价于"激活 venv 后运行 python3"。没有它，`sglang_omni` 与 `typer`、`pydantic` 等依赖可能不可导入。

### 1.2 `python3 -m sglang_omni.cli` —— 模块入口层

`-m` 触发 Python 的 runpy 机制：把 `sglang_omni.cli` 当作 `__main__` 执行，即运行
`sglang_omni/cli/__main__.py`：

```python
from . import app

if __name__ == "__main__":
    app()
```

`app` 定义在同包 `__init__.py`，是一个 **Typer** 应用：

```python
app = Typer()

app.add_typer(config_app, name="config")          # config 子应用
app.command("check-gpu")(check_gpu)               # check-gpu 子命令
app.command(
    "serve", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)(_serve)                                          # serve 子命令（关键！）
```

**注意 serve 的注册方式**：`allow_extra_args=True` + `ignore_unknown_options=True`。
这意味着 serve 命令**故意不穷举所有配置参数**——它只声明少量通用选项（`--model-path`、`--config`、`--port`、`--host` 等），其余形如 `--tts_engine.engine.mem_fraction_static 0.7` 的"点分覆盖参数"不会被 Click 拒绝，而是被收集进 `ctx.args`，稍后交给 `ConfigManager.parse_extra_args` 解析成配置补丁。这是"模型各异、参数无法枚举"这一设计约束的直接体现。

### 1.3 `serve` 子命令与参数分类

Typer/Click 解析时，本命令的参数分为两类：

| 参数 | 类别 | 去向 |
|---|---|---|
| `--model-path <本地路径>` | Typer 声明选项（`str`） | `serve()` 函数形参 `model_path` |
| `--config examples/configs/qwen3_tts_1_7b.yaml` | Typer 声明选项（`str`） | `serve()` 函数形参 `config` |
| `--port 8000` | Typer 声明选项（`int`，默认 8000） | `serve()` 函数形参 `port` |
| （无） | 未声明的点分参数 | `ctx.args`（本例为空列表） |

未给出的选项取默认值：`--host 0.0.0.0`、`--log-level info`、`--text-only False`、`--colocate False`、`--enable-realtime False`、`--mem-fraction-static None` 等。

---

## 2. 第一层：Typer CLI 解析

`serve()` 函数体（`sglang_omni/cli/serve.py`）的第一阶段：

```python
logging.basicConfig(
    level=getattr(logging, log_level.upper()),   # info
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

_validate_colocate_cli_request(colocate=False, config=..., text_only=False)
# colocate=False → 直接返回，无操作
```

随后进入配置解析主分支：

```python
if config:                                      # 本例命中
    config_manager = ConfigManager.from_file(config)
elif text_only:
    ...                                          # 未命中
else:
    config_manager = ConfigManager.from_model_path(model_path)
```

本例同时给出了 `--config` 与 `--model-path`，走 `from_file` 分支；`--model-path` 稍后以"CLI 层补丁"身份覆盖文件中的 `model_path`（见 §3.4，这是本命令最关键的一处语义）。

---

## 3. 第二层：配置解析（patch 体系）

这是整个启动过程中设计最精细的部分。核心思想：

> **所有配置来源（模型默认值 / YAML 文件 / CLI 标志 / 点分覆盖）统一归约为同一种表示——`ConfigPatch(path, value, source)`——合并规则只在一个地方执行（`ConfigResolver`），而不是散落在各解析器里。**

### 3.1 优先级模型（`config/patch.py`）

每个补丁携带来源与优先级，最终排序键为四元组：

```
(Layer, Specificity, path 深度, 插入顺序)
```

| 维度 | 取值（高者胜） | 含义 |
|---|---|---|
| **Layer** | `CLI=20` > `USER_FILE=10` > `MODEL_DEFAULT=0` | 跨来源：CLI 永远压过 YAML，YAML 压过模型内置默认 |
| **Specificity** | `EXPLICIT=20` > `ROLE=10` > `BROADCAST=0` | 同层内：点名某 stage 的路径压过广播型标志（如 `--mem-fraction-static`） |
| **path 深度** | 深者后应用 | 容器写先落、叶子写后落，`stages.x.env` 不会覆盖兄弟键 `stages.x.env.OMP_NUM_THREADS` |
| **插入顺序** | 后者胜 | 仅作为已被冲突检查放行的补丁的最后平局裁决 |

**冲突即报错**：同优先级、同路径、不同值（如 `--thinker.tp_size 4 --thinker.tp_size 8`）直接抛 `DuplicatePatchError`，绝不按"后写的赢"静默裁决——代码注释明确说明：任何发明的平局规则（参数顺序、字典序）都是用户必须学习的隐含规则。

### 3.2 YAML 文件解析（`sources_from_config_file`）

本例的 `examples/configs/qwen3_tts_1_7b.yaml` 内容极简：

```yaml
config_cls: Qwen3TTSPipelineConfig
model_path: Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

解析步骤：

1. **严格 YAML 加载**：`_DuplicateKeyRefusingLoader`（SafeLoader 子类）拒绝重复映射键。普通 `yaml.safe_load` 会静默保留最后一个重复键，使冲突检测永远看不到文件内的重复。
2. **形态检查**：顶层必须是 mapping；出现已废弃块（`stage_overrides` / `runtime_overrides` / `entry_stage`）则报错并给出迁移指引。
3. **解析 `config_cls`**：`PIPELINE_CONFIG_REGISTRY.get_config_cls_by_name("Qwen3TTSPipelineConfig")`。
   - 注册表在包导入时扫描 `sglang_omni.models` 下所有模型子包的 `config.py`，取各模块导出的 `EntryClass`，按其 `architecture` ClassVar（Qwen3-TTS 为 `"Qwen3TTSForConditionalGeneration"`）注册。
   - 按类名查找时还会遍历各模型模块的 `Variants` 字典（变体配置类）。
4. **弹出 `stages:` 与 `shared:` 块**（本例两者都不存在），剩余的 `model_path` 成为文件层覆盖项。
5. **基线构造循环**：先尝试 `config_cls()` 空参构造；`model_path` 是必填字段，会抛 `ValidationError(missing)`；把"缺失且文件里有"的字段补进构造参数后重建。这样**基线只包含构造必需的字段**，文件里的其余值全部以补丁形式施加一次——避免"基线已含值 + 补丁再写一次"导致来源追溯（provenance）混乱。
6. **叶子拆分**：`model_path` 是叶子 → 产生 1 个补丁：

```python
ConfigPatch(
    path=ConfigPath("model_path"),            # 编译期已校验可写、类型为 str
    value="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    source=ConfigSource(YAML_FILE, "examples/configs/qwen3_tts_1_7b.yaml"),
    layer=Layer.USER_FILE,                    # = 10
)
```

7. **`ConfigResolver.resolve`**：
   - `require_no_conflicts()`（单补丁，无冲突）；
   - `ordered()` 排序后逐个 `_apply` 到基线的 `model_dump()` 上（叶子直写，容器深合并）；
   - `config_cls(**data)` 重建配置，触发 Pydantic 完整校验与 `model_post_init`；
   - 重建成功后回读每个补丁路径的**实际生效值**记入 provenance（模型校验可能改写字段，追溯的必须是"生效值"而非"补丁值"）。

`model_post_init` 做的两件值得注意的事：

```python
self.config_cls = self.__class__.__name__   # 写回 "Qwen3TTSPipelineConfig"
if self.name is None:
    self.name = self.model_path             # name 默认取 model_path
```

### 3.3 基线配置：Qwen3-TTS 三阶段流水线

由于 YAML 没有写 `stages:` 块，完整阶段拓扑来自模型代码 `sglang_omni/models/qwen3_tts/config.py`：

```python
class Qwen3TTSPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "Qwen3TTSForConditionalGeneration"
    stage_config_types = {"tts_engine": EngineStageConfig}   # stage 名 → 类型化校验

    stages = [
        StageConfig(name="preprocessing", process="pipeline",
                    factory_path="...qwen3_tts.stages.create_preprocessing_executor",
                    next="tts_engine"),
        EngineStageConfig(name="tts_engine", process="pipeline",
                    factory_path="...qwen3_tts.stages.create_sglang_tts_engine_executor",
                    factory=FactoryArgs(dtype="bfloat16"),
                    gpu=0, next="vocoder", stream_to=["vocoder"]),
        StageConfig(name="vocoder", process="pipeline",
                    factory_path="...qwen3_tts.stages.create_vocoder_executor",
                    factory=FactoryArgs(dtype="bfloat16"),
                    gpu=0, terminal=True, can_accept_stream_before_payload=True),
    ]
```

| 阶段 | 类型 | GPU | 职责 | 路由 |
|---|---|---|---|---|
| `preprocessing` | 普通 Stage | 无（CPU） | 文本分词/参考音频编码等请求预处理 | → tts_engine |
| `tts_engine` | **EngineStage**（驱动 SGLang 引擎） | 0 | AR 主干（Qwen3TTSTalker）逐 token 生成语音 token | → vocoder，且**流式**直通 vocoder |
| `vocoder` | 普通 Stage | 0 | 语音 token → 波形（流式 vocoder），终点阶段 | terminal |

`preprocessing → tts_engine` 被声明为 `process_local_edges()`：预处理结果存放在模块级寄存器中，两个阶段必须同进程。

`model_post_init` 的 `_validate_general` 会校验：阶段名唯一、`next`/`terminal` 二者居其一、`wait_for` 必须配 `merge_fn`、路由目标必须存在、非 TP 阶段必须声明 `process` 等。任一失败 → 启动即报错。

### 3.4 `--model-path` 双源覆盖（本命令的关键语义）

回到 `serve()`。因为 **`--config` 和 `--model-path` 同时给出**，代码走了这条分支：

```python
if config and model_path is not None:
    flag_patches = flag_patches.merge(
        patches_from_model_path_flag(model_path, config_manager.config)
    )
```

`patches_from_model_path_flag` 生成一个 **CLI 层补丁**：

```python
ConfigPatch(
    path=ConfigPath("model_path"),
    value="/home/zy0057/.../Qwen3-TTS-12Hz-1.7B-Base",   # 本地路径
    source=ConfigSource(CLI_FLAG, "--model-path"),
    layer=Layer.CLI,                                      # = 20 > USER_FILE = 10
)
```

**结论**：文件里的 `model_path: Qwen/Qwen3-TTS-12Hz-1.7B-Base`（HF ID）被命令行的本地路径覆盖。最终生效的是本地路径。

> 为什么做成补丁而不是直接赋值？为了让"本次启动"与 `sgl-omni config resolve/explain` 的预览完全一致——来源追溯能说清"这个值来自 `--model-path` 标志"。

### 3.5 合并与最终裁决

```python
extra_args = config_manager.parse_extra_args(ctx.args)   # 本例 ctx.args == [] → []
merged_config = config_manager.merge_config(extra_args, extra_patches=flag_patches)
```

`merge_config` 内部：

1. `patches_from_dotted_cli([])` → 空补丁集（若有点分参数，此函数会把 `--tts_engine.tp_size 2` 规范化为 `stages.tts_engine.tp_size`：`stages.` 前缀在命令行上隐含；`compat.canonicalize_dotted_key` 负责补前缀与歧义拒绝——首段既是阶段名又是顶层字段名时直接报错）；
2. 合并 CLI 标志补丁；
3. `ConfigResolver.resolve` 完成排序、冲突检查、按序应用、重建校验；
4. `_validate_dotted_gpu_override_conflicts`：拒绝用 `stages.<x>.gpu` 覆盖已声明 `replica_devices` 的进程的 GPU 放置（本例无点分参数，跳过）。

若给出 `--mem-fraction-static`，它会被翻译成**每引擎阶段一个**的 `BROADCAST` 优先级补丁（`patches_from_broadcast_flags`），因此显式的点分路径（`EXPLICIT`）仍然能对单个 stage 覆盖它。

### 3.6 TP 派生与收尾

```python
merged_config = apply_tensor_parallel_engine_overrides(merged_config)
```

遍历所有阶段：本例三个阶段 `tp_size` 均为 1，`tensor_parallel_server_args_overrides` 返回空 → **无写入**，配置原样返回。该函数的设计意图：只有当 TP > 1 时，才把"TP 设置隐含的引擎参数"（如 `disable_custom_all_reduce`，且按 GPU 拓扑 P2P 探测结果放宽）补写进缺失的 `engine.*` 键——**派生只填空格，不覆盖用户显式写的值**。

`_should_print_merged_config(colocate=False, log_level="info")` → False，跳过配置打印（仅 `--colocate` 或 `--log-level debug` 时打印）。

至此，`serve()` 把三样东西交给 `launch_server`：**完全解析校验的 PipelineConfig**、`host="0.0.0.0"`、`port=8000`（以及 model_name、log_level 等）。

---

## 4. 第三层：launch_server 启动流水线

`launch_server`（`sglang_omni/serve/launcher.py`）是阻塞入口：

```python
def launch_server(pipeline_config, *, host="0.0.0.0", port=8000, ...):
    apply_gpu_compat_env_defaults()          # 按当前 GPU 型号设置兼容性环境变量默认值
    asyncio.run(_run_server(pipeline_config, host=host, port=port, ...))
```

`_run_server` 是 async 主流程，按顺序做六件事：

### 4.1 端口探测（`_find_available_port`）

```python
port = _find_available_port(host, port)     # ("0.0.0.0", 8000)
```

- 用临时 socket `bind(("0.0.0.0", 8000))` 试探：成功 → 用 8000；
- 失败（被占用）→ 打印警告，绑定 `(host, 0)` 让内核分配随机空闲端口并使用；
- **例外**：环境变量 `SGLANG_OMNI_STRICT_PORT=1` 时，被占用直接抛错，不许回退——供健康检查固定端口的监督方（如同 GPU DP 启动器）快速失败。
- 注意此探测发生在**模型加载之前**，避免白等几十秒才发现端口冲突。

### 4.2 运行时编译（`prepare_pipeline_runtime`）

```python
mp_runner = MultiProcessPipelineRunner(pipeline_config)
await mp_runner.start(timeout=float(os.environ.get("SGLANG_OMNI_STARTUP_TIMEOUT", "600")))
```

`start()` 第一步是 `prepare_pipeline_runtime(config)`（`pipeline/runtime_config.py`），把声明式配置编译成可执行计划：

1. **`compile_logical_processes`**：按 `process` 字段把阶段归组。本例三个阶段全部 `process="pipeline"` → 单个逻辑进程 `"pipeline"`，三阶段同进程共存。
2. **`expand_replica_stages`**：副本展开（`processes.<name>.num_replicas`，本例全为默认 1，无展开）。
3. **`validate_device_assignment`**：`tts_engine`/`vocoder` 声明 `gpu=0`，对照 `torch.cuda.device_count()` 校验设备号合法。
4. **`create_ipc_runtime_dir`**：在 `/tmp/sglang_omni/` 下创建按 pipeline 名命名的本次运行专属目录，并做 **Unix socket 路径长度预算校验**（zmq `ipc://` 路径不得超过约 100 字符，超限则要求缩短 `endpoints.base_path` 或阶段名）。
5. **`build_stage_placement_plan`**：GPU 放置计划——GPU 0 上放 `tts_engine` + `vocoder`，并按 `gpu_memory_fraction` 校验每卡内存预算总和。
6. **`build_process_topology_plan`**：进程拓扑计划（哪些阶段共享哪个进程、TP 阶段到进程的映射）。
7. **`allocate_endpoints`**：为协调器与每个阶段分配 ZeroMQ IPC 端点：

```
ipc://<runtime_dir>/completion.sock            # 请求完成事件
ipc://<runtime_dir>/abort.sock                 # 请求中止
ipc://<runtime_dir>/stage_preprocessing.sock
ipc://<runtime_dir>/stage_tts_engine.sock
ipc://<runtime_dir>/stage_vocoder.sock
ipc://<runtime_dir>/comm_<stage>_rank<N>.sock  # 阶段间数据通道（本例各 1 rank）
```

### 4.3 Coordinator（请求协调器）

```python
self._coordinator = Coordinator(
    completion_endpoint=..., abort_endpoint=...,
    entry_stage="preprocessing",            # resolved_entry_stage = stages[0]
    terminal_stages=["vocoder"],
    max_in_flight=...,                      # 来自 generation_admission_defaults()
)
```

- Coordinator 持有全局请求表：请求从 `entry_stage` 进入，沿 `next`/`stream_to` 路由，到达 `terminal` 阶段后经 completion socket 回传。
- `Qwen3TTSPipelineConfig.generation_admission_defaults()` 返回 `max_running_requests=16, max_queued_requests=16`——协调器层的在途请求数上限（运行中 + 排队）。
- `await coordinator.start()` 后启动 `run_completion_loop()` 协程消费完成事件。

### 4.4 spawn 子进程

```python
ctx = multiprocessing.get_context("spawn")
...
for group in self._groups:
    group.spawn(ctx)
await asyncio.gather(*(g.wait_ready(timeout) for g in self._groups))
```

- 使用 **spawn** 启动方式（非 fork）：子进程重新导入模块，规避 CUDA 与 fork 的兼容性问题。
- 每个**逻辑进程**一个 worker 子进程。本例只有一个 worker 进程，其内部按阶段工厂构造三个执行器（见 §6）。
- `wait_ready` 等待每个进程通过控制端点报告就绪；等待期间任何进程死亡 → `RuntimeError: Stage process(es) died during startup`，并走 `_cleanup_on_failure` 清理。
- （本例 `mps="off"`，MPS 相关逻辑全部跳过。）
- 全部就绪后，把各阶段控制端点注册进 Coordinator，并启动 `_monitor_children()` 守护协程：每 5 秒轮询子进程存活与 MPS 健康，发现异常即置位 fatal 事件、失败所有在途请求。

### 4.5 客户端与 FastAPI 应用

```python
client = Client(coordinator)
app = create_app(client, model_name=..., ...)
```

`Client`（`sglang_omni/client/client.py`）把协调器的 ZMQ 协议封装成进程内 API，供 HTTP 层调用。`create_app` 见 §7。

### 4.6 uvicorn 启动与故障联动

```python
config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info", timeout_keep_alive=120)
server = _PipelineUvicornServer(config)
await _serve_with_failure_watch(server, [mp_runner.wait_failed()])
```

- `_PipelineUvicornServer` 重写了 `capture_signals`：uvicorn 收到 SIGTERM 后默认会 re-raise 终止解释器，导致流水线子进程来不及清理；此处捕获信号后恢复原 handler 但**吞掉**已处理的信号，让流水线启动器负责子进程收尾。
- `_serve_with_failure_watch` 用 `asyncio.wait(..., FIRST_COMPLETED)` 同时盯两个任务：
  - HTTP server 任务正常退出 → 正常返回；
  - `wait_failed()` 先完成（流水线运行时崩溃/子进程死亡）→ 置 `server.should_exit=True` 优雅停 HTTP，再把异常抛出；
  - finally 中取消剩余 watcher。
- 最后 `finally: await mp_runner.stop()`——无论正常退出还是异常，都停止全部子进程并删除 IPC 运行目录。

---

## 5. 衔接：worker 进程内的阶段装配

worker 进程启动后按配置执行各阶段工厂。工厂路径由 `factory_path` 给出，构造参数来自三方的合并：`factory.*`（用户可配）> `stage_factory_kwargs()`（模型代码在启动时求值的接线参数）> 工厂函数自身默认值。本例 `enable_deterministic_inference=False`，`stage_factory_kwargs` 返回空。

---

## 6. 第四层：三阶段工厂与 SGLang 引擎构建

### 6.1 preprocessing —— `create_preprocessing_executor`

```python
ThreadedSimpleScheduler(
    preprocess_qwen3_tts_payload,        # 单请求预处理函数
    max_concurrency=8,                   # 必须并发：串行会让 speech_tokenizer 批处理永远凑不满批
    abort_callback=cleanup_prepared_qwen3_tts_request,
)
```

纯 CPU 线程池执行器：对进入的 TTS 请求做分词、参考音频的 speech-tokenizer 编码等，产出 `tts_engine` 可消费的载荷。`abort_callback` 保证请求中止时清理已登记的预处理产物。

### 6.2 tts_engine —— `create_sglang_tts_engine_executor` → `Qwen3TtsEngineBuilder.build()`

这是最重的一步——在 GPU 0 上拉起一个完整的 SGLang 推理引擎。`build()` 流程（`scheduling/engine_factory.py` + `models/qwen3_tts/engine_builder.py`）：

1. **checkpoint 解析**：`resolve_checkpoint(model_path)` → 本地权重目录；
2. **前置补丁**：`pre_infra_setup` 应用 qwen-tts 的 transformers 兼容补丁，并把 `Qwen3TTSConfig` 注册进 transformers `AutoConfig`（`AutoConfig.register("qwen3_tts", ...)`）；
3. **上下文长度**：`Qwen3TtsEngineBuilder.context_length = 8192`；
4. **引擎默认参数**（`generation_defaults(dtype="bfloat16")`）：

   | 参数 | 值 | 说明 |
   |---|---|---|
   | `mem_fraction_static` | 0.85 | 静态显存占比（KV cache 等） |
   | `max_running_requests` | 16 | 引擎内并发请求上限 |
   | `cuda_graph_max_bs` | 32 | CUDA Graph 最大 batch |
   | `disable_overlap_schedule` | True | 关闭 SGLang 重叠调度 |
   | `max_prefill_tokens` | 8192 | |
   | `sampling_backend` | pytorch | |
   | `trust_remote_code` | True | |

   若用户在 YAML/CLI 写了 `stages.tts_engine.engine.*`，会与这些默认值合并（用户值优先）；
5. **构造 ServerArgs**：`sglang_backend.build_sglang_server_args(checkpoint_dir, context_length=8192, **overrides)` → SGLang 原生 `ServerArgs`；
6. **基础设施**：`create_sglang_infrastructure_defer_cuda_graph(...)` 创建 model_worker（SGLang ModelRunner）、RadixCache、token 池与 KV cache 分配器；CUDA Graph 延后到模型装载后；
7. **模型级装配**（`setup_model`）：把 `speech_tokenizer`（从权重目录的 `speech_tokenizer/` 子目录加载 `Qwen3TTSTokenizer`）注入模型；加载 `AutoProcessor`；用官方 `qwen_tts.Qwen3TTSModel` 包装引擎模型并读取 `generation_config.json` 默认生成参数；登记预处理上下文；
8. **运行时装配**：`SGLangOutputProcessor` + `make_adapters`（request_builder / result_adapter / stream_output_builder）+ `_build_runtime` 组装出调度器；`request_build_max_workers=4`；
9. **CUDA Graph 初始化与校验**：`init_sglang_cuda_graphs` + prefill 图批量校验（Qwen3-TTS 支持可中断 prefill CUDA Graph 契约）；
10. 返回调度器实例——`tts_engine` 阶段的执行器就绪。

### 6.3 vocoder —— `create_vocoder_executor`

```python
device = resolve_device_spec(None, 0)                 # cuda:0
tokenizer = _load_qwen3_tts_tokenizer(...)            # 再加载一份 Qwen3TTSTokenizer（反离散化用）
scheduler = Qwen3TTSStreamingVocoderScheduler(
    tokenizer, device=device,
    stream_stride=..., stream_left_context_frames=...,   # 流式分帧参数
    max_batch_size=8, max_batch_wait_ms=2,
    initial_max_batch_size=32, initial_batch_wait_ms=2,  # 首块/后续块差异化批处理
    initial_cuda_graph=True, followup_cuda_graph=True,   # CUDA Graph 加速
)
scheduler.warmup_now()   # 预热
```

- 从 `tts_engine` 流式接收语音 token（`stream_to=["vocoder"]` + `can_accept_stream_before_payload=True` 允许音频流先于最终载荷到达）；
- 按 `stream_stride` 分帧、分"首块/后续块"两档批处理节奏解码为波形；
- **`warmup_now()` 在工厂构造完成后、就绪上报之前执行**——这是刻意安排：CUDA capture 不能与共置阶段的请求期 GPU 工作重叠。

---

## 7. 第五层：HTTP 服务与 uvicorn

### 7.1 `create_app` 组装 FastAPI

```python
app = FastAPI(title="sglang-omni", version=__version__)
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)        # 全开放 CORS
app.add_middleware(VoiceUploadBodyLimitMiddleware, max_bytes=...)   # 语音上传体积上限
```

`app.state` 注入运行时依赖：

| state | 值（本例） | 说明 |
|---|---|---|
| `client` | Client(coordinator) | HTTP 层 → 流水线的唯一通道 |
| `model_name` | pipeline `name`（= model_path） | `/v1/models` 与默认 model 字段 |
| `architectures` | `["Qwen3TTSForConditionalGeneration"]` | |
| `speaker_sample_store` | SpeakerSampleStore() | 参考音色上传的持久存储 |
| `speech_service` | SpeechRequestValidator(...) | TTS 请求校验器 |

两个与模型相关的行为由 `Qwen3TTSPipelineConfig` 的路径检测决定（`_is_qwen3_tts_base_model`：路径中含 `qwen3_tts`（容忍 `-`/`_` 混写）、含 `_base` 标记、且不含 `custom_voice`/`voice_design` 等标记）——本例路径 `.../Qwen3-TTS-12Hz-1.7B-Base` 命中 **Base 模型**：

- `requires_uploaded_voice_for_named_voice=True`：具名音色必须是已上传音色；
- `supports_uploaded_voice_references=True`：上传音色可下沉为参考音频。

### 7.2 路由注册清单

```python
_register_health(app)            # GET  /health（含 favicon）
_register_models(app)            # GET  /v1/models
_register_admin(app, key)        # 管理端点（RL 权重更新/生成暂停等）
_register_chat_completions(app)  # POST /v1/chat/completions（多模态）
_register_voices(app)            # GET/POST/DELETE /v1/audio/voices（音色管理）
_register_generate(app)          # POST /v1/generate（内部通用入口）
_register_speech(app)            # POST /v1/audio/speech（TTS，支持流式 PCM）
_register_speech_batch(app)      # POST /v1/audio/speech/batch
_register_speech_ws(app)         # WS   /v1/audio/speech/stream（有状态 TTS 流）
register_transcriptions(app)     # POST /v1/audio/transcriptions（ASR）
register_translations(app)       # POST /v1/audio/translations
# enable_realtime=False → 不挂载 /v1/realtime（OpenAI Realtime API）
```

随后 launcher 再挂载 **profiler 路由**：`POST /start_profile`、`/start_request_profile`、`/stop_profile`、`/stop_request_profile`（透传给各阶段 worker 的控制端点，`SGLANG_TORCH_PROFILER_DIR` 未设时 torch trace 请求会 400）。

### 7.3 服务就绪

uvicorn 在 `0.0.0.0:8000` 上开始监听。启动日志（`_placement_log_summary`）会打印解析后的放置计划：拓扑类名、pipeline 名、进程组（`pipeline`: [preprocessing, tts_engine, vocoder]）、每 GPU 阶段与显存预算、模型能力摘要（`reference_audio`、`streaming_vocoder`、`breakable_prefill_cuda_graph` 等）。

---

## 8. 全链路时序总览

```
uv run
 └─ python3 -m sglang_omni.cli            __main__.py → Typer app()
     └─ serve(ctx, model_path, config, port=8000, ...)          cli/serve.py
         ├─ logging.basicConfig(info)
         ├─ ConfigManager.from_file(qwen3_tts_1_7b.yaml)
         │    └─ sources_from_config_file
         │         ├─ 严格 YAML 加载（拒绝重复键）
         │         ├─ 注册表按名查 Qwen3TTSPipelineConfig
         │         ├─ 基线构造（仅必需字段）→ 三阶段默认拓扑
         │         └─ 文件值 → USER_FILE 层补丁（model_path）
         │    └─ ConfigResolver.resolve → 校验 + model_post_init
         ├─ parse_extra_args(ctx.args == [])                  → 无点分补丁
         ├─ --model-path 与 --config 同给
         │    └─ patches_from_model_path_flag                 → CLI 层补丁（胜出）
         ├─ merge_config → ConfigResolver（CLI > FILE）
         │    最终 model_path = /home/.../Qwen3-TTS-12Hz-1.7B-Base
         ├─ apply_tensor_parallel_engine_overrides            → tp 全 1，无写入
         └─ launch_server(config, host, port=8000)            serve/launcher.py
              ├─ apply_gpu_compat_env_defaults()
              └─ asyncio.run(_run_server)
                   ├─ _find_available_port(0.0.0.0, 8000)     # 模型加载前探测
                   ├─ MultiProcessPipelineRunner(config)
                   │    └─ start(timeout=600)
                   │         ├─ prepare_pipeline_runtime
                   │         │    ├─ compile_logical_processes   → 1 个进程 "pipeline"
                   │         │    ├─ expand_replicas / validate_devices
                   │         │    ├─ create_ipc_runtime_dir(/tmp/sglang_omni/...)
                   │         │    ├─ placement plan / process plan
                   │         │    └─ allocate_endpoints(zmq ipc:// *.sock)
                   │         ├─ Coordinator(entry=preprocessing, terminal=[vocoder],
                   │         │             max_in_flight=16+16)
                   │         ├─ spawn worker（spawn 上下文）
                   │         │    ├─ preprocessing → ThreadedSimpleScheduler(8 并发)
                   │         │    ├─ tts_engine    → Qwen3TtsEngineBuilder.build()
                   │         │    │                   （ServerArgs(8192, 0.85, …) →
                   │         │    │                    SGLang infra → 模型 → CUDA Graph）
                   │         │    └─ vocoder       → StreamingVocoderScheduler + warmup
                   │         ├─ wait_ready × N（任一进程死亡即失败并清理）
                   │         ├─ 注册 stage 端点 → Coordinator
                   │         └─ _monitor_children（5s 轮询）
                   ├─ Client(coordinator)
                   ├─ create_app(...)                          serve/openai_api.py
                   │    ├─ CORS + 上传体积中间件
                   │    ├─ Base 模型检测 → 具名音色须先上传
                   │    └─ 注册 /v1/* 全套路由 + profiler 路由
                   └─ _PipelineUvicornServer(0.0.0.0:8000)
                        └─ _serve_with_failure_watch(server, wait_failed)
                             HTTP 与流水线运行时，任一退出/崩溃 → 优雅联动停机
                             finally → mp_runner.stop()（收子进程、删 IPC 目录）
```

---

## 9. 运行期：一条 TTS 请求怎么走

服务就绪后，以 `POST /v1/audio/speech` 为例：

1. **HTTP 层校验**（`SpeechRequestValidator`）：model 解析（默认 = model_path）、具名音色必须来自 `SpeakerSampleStore`（Base 模型约束）、参考音频数量/文本要求、输入长度上限；
2. **构造 OmniRequest** 交给 `Client` → 写入 Coordinator 的入口队列；
3. **Coordinator** 按 `entry_stage=preprocessing` 派发到（同进程的）preprocessing 执行器；
4. `preprocess_qwen3_tts_payload` 完成文本/参考音频预处理，载荷沿 `next` 路由到 `tts_engine`；
5. `tts_engine`（SGLang 引擎，Qwen3TTSTalker 架构）逐 token 生成语音 token，**边生成边流式**推给 `vocoder`（`stream_to`）；
6. `vocoder` 流式解码为波形块；整段完成后经 `completion.sock` 回到 Coordinator → `Client` → HTTP 响应（或流式 PCM / WS 音频帧）；
7. 全程在途上限：`max_running_requests=16 + max_queued_requests=16`（协调器层）与引擎自身的 `max_running_requests=16`（引擎层）双层把关。

---

## 10. 失败模式与环境变量速查

| 现象/需求 | 机制 | 出处 |
|---|---|---|
| 端口被占但悄悄换了端口 | 启动日志 warning + 实际端口变更 | `_find_available_port` |
| 必须固定 8000，被占即失败 | `SGLANG_OMNI_STRICT_PORT=1` | 同上 |
| YAML 里同一个键写两次 | 解析期报 `duplicate mapping key` | `_DuplicateKeyRefusingLoader` |
| 两个同优先级来源写同一路径且值不同 | `DuplicatePatchError`（要求 Pick one） | `ConfigPatchSet.require_no_conflicts` |
| 点分参数首段既非阶段名也非顶层字段 | 报错并列出本 pipeline 的合法阶段名 + 相近建议 | `canonicalize_dotted_key` |
| 对非引擎阶段写 `engine.*` | 路径编译期拒绝（`engine_stage=False`） | `StageConfig` / `ConfigPath` |
| 阶段路由（next/terminal/stream_to）非法 | 配置重建时 `model_post_init` 校验失败 | `PipelineConfig._validate_general` |
| 子进程启动期死亡 | `wait_ready` 抛 `Stage process(es) died during startup` 并清理 | `mp_runner.start` |
| 运行中 worker 崩溃 | monitor 检测 → 失败在途请求 → 联动停 HTTP → 进程退出 | `_monitor_children` / `_serve_with_failure_watch` |
| SIGTERM 不杀子进程 | uvicorn 信号被捕获吞掉，由 launcher 统一收尾 | `_PipelineUvicornServer.capture_signals` |
| 启动超时 | `SGLANG_OMNI_STARTUP_TIMEOUT`（默认 600 秒） | `_run_server` |
| torch profiler 目录 | `SGLANG_TORCH_PROFILER_DIR`（未设时 torch trace 的 /start_profile 400） | `_mount_profiler_routes` |
| qwen-tts 包缺失 | 报错并打印完整安装指引（apt sox + uv pip --no-deps qwen-tts==0.1.1） | `_QWEN_TTS_INSTALL_HINT` |
| Qwen3-TTS 上 `enable_torch_compile` | `adjust_overrides` 直接拒绝 | `Qwen3TtsEngineBuilder` |

---

## 11. 附录：关键源文件索引

| 文件 | 职责 |
|---|---|
| `sglang_omni/cli/__init__.py` | Typer app 与子命令注册（serve 允许额外参数） |
| `sglang_omni/cli/__main__.py` | `python -m` 入口 |
| `sglang_omni/cli/serve.py` | serve 命令体：参数声明、配置解析编排、双源覆盖、TP 派生、调用 launch_server |
| `sglang_omni/config/sources.py` | YAML/点分 CLI/`--model-path`/`shared:` → 补丁；严格 YAML loader；`dump_user_config` |
| `sglang_omni/config/patch.py` | 补丁数据结构、Layer/Specificity 优先级、冲突检测 |
| `sglang_omni/config/compat.py` | 点分键规范化（隐含 `stages.` 前缀、歧义拒绝） |
| `sglang_omni/config/path.py` | 针对类型化 schema 编译配置路径（可写性、类型 coerce、叶子判定） |
| `sglang_omni/config/resolver.py` | 唯一的合并执行点：排序、应用、重建校验、provenance |
| `sglang_omni/config/schema.py` | PipelineConfig / StageConfig / EngineStageConfig / EngineArgs / FactoryArgs 与全部校验 |
| `sglang_omni/models/registry.py` | 架构 → 配置类注册表（扫描 models/*/config.py 的 EntryClass） |
| `sglang_omni/models/qwen3_tts/config.py` | Qwen3-TTS 三阶段拓扑与 Base 模型行为声明 |
| `sglang_omni/models/qwen3_tts/stages.py` | 三个阶段工厂（preprocessing / tts_engine / vocoder） |
| `sglang_omni/models/qwen3_tts/engine_builder.py` | Qwen3TtsEngineBuilder：引擎默认参数、speech_tokenizer 注入、适配器 |
| `sglang_omni/scheduling/engine_factory.py` | `build()` 通用骨架：ServerArgs → SGLang 基础设施 → 模型 → CUDA Graph → 调度器 |
| `sglang_omni/pipeline/runtime_config.py` | 逻辑进程编译、副本展开、GPU 放置计划、IPC 端点分配 |
| `sglang_omni/pipeline/mp_runner.py` | MultiProcessPipelineRunner：spawn、就绪等待、子进程监控、协调器接入 |
| `sglang_omni/serve/launcher.py` | launch_server/_run_server：端口探测、装配、uvicorn、故障联动、信号处理 |
| `sglang_omni/serve/openai_api.py` | create_app 与全部 OpenAI 兼容路由 |

---

*本文基于 2026-09 时点的工作区源码（分支 `feat-dev_tts`，HEAD `d6670de4`）逐文件追踪整理。*
