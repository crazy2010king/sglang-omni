# 11 通信引擎与张量传输：CommEngine、传输选择、DataRef 与 KV 转移

> 主角：`comm/engine.py`（1136 行）、`comm/router.py`（399 行）、`comm/data_ref.py`（212 行）、
> `comm/stage_io.py`（818 行）、`relay/`（cuda_ipc / shm / mooncake / nccl / nixl）。

---

## 1. 三个枚举定义了一切（data_ref.py:11-33）

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

## 2. CommRouter：传输选择的决策树（router.py:130-297）

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

## 3. CommEngine：发送/接收的执行体（engine.py:96-…）

### 3.1 发送路径

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

### 3.2 接收路径

Stage 收到 `DataReadyMessage` 后：`CommEngine.read_data(relay, request_id, data_ref)`
→ relay 层反序列化 → Stage `_send_data_ack`。01 篇 §1 的三分支
（direct IPC / inline / relay）就是在这一层前分流，绕过 relay 的两个捷径
由 `stage_io` 的谓词与反序列化器支持：
`deserialize_direct_cuda_ipc_payload`（CUDA IPC 句柄直接 open 成 GPU 张量）与
`deserialize_inline_stream_chunk`（小张量随消息走，省一次取货）。

### 3.3 KV 转移（KV_PAGES）

engine.py 的后半是**离散式部署的 KV cache 迁移**：`register_kv_pool` 注册本端池、
`prepare_kv_receive` 协商对端池布局（`KVPoolLayout.compatible_with`，
proto/kv_transfer.py）、`send_kv_pages` 按 `req_to_token` 页表搬运。
thinker→talker 虽然在当前 Qwen 拓扑中不共享 KV（talker 重放投影 prompt），
但这条通道是"同一引擎族共享 KV"愿景的基础设施
（Ming-Omni 等模型使用）。

---

## 4. Relay 实现（relay/）

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

## 5. 端到端一次 payload 传输（时间线）

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

## 6. 与 Qwen3-Omni 性能相关的三个事实

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

## 7. 小结

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
