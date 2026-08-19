<!-- markdownlint-disable MD013 MD024 MD036 MD060 -->

# PyPTO L1 单算子与 ACLGraph 完整设计文档

## 0. 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档性质 | 当前实现的最终态设计说明、接口契约与维护指南 |
| 更新时间 | 2026-08-19 |
| PyPTO 基线 | `e38f2d028c5c1b15c30ebb5e3e47b6930e3b77cb`，`main` |
| Simpler 基线 | `4922d5933e2937790aa5b01e737986114ac28d1d`，`main` |
| 当前验收平台 | A2/A3 onboard |
| 当前 L1 runtime | `tensormap_and_ringbuffer`（TRB）与 `host_build_graph`（HBG） |
| 当前入口 | `@pl.program` 编译产物，经 `pypto_init` 暴露为 PyTorch 可调用单算子 |
| 关联文档 | [实现过程记录](./PyPTO_L1与ACLGraph实现过程记录.md)、[原始实现计划](./pypto_l1_aclgraph_implementation_plan.md) |

本文是本次工作的完整设计文档。它不是阶段性计划，也不是测试流水账。出现信息冲突时，采用以下优先级：

1. 当前 PyPTO 与 Simpler 源码；
2. 已记录的 A2/A3 实机证据；
3. 本文明确写出的最终契约；
4. 实现过程记录中的历史判断；
5. 原始实现计划中的预案。

原始实现计划保留了大量设计推演和风险分析，仍然值得阅读；实现过程记录保留了故障、反例和上板证据。本文将二者收敛成一套可直接用于维护、评审和后续演进的最终模型。

### 0.1 相对早期计划的三项关键修正

当前实现已经超出早期 Phase 1 的最小范围，以下三点必须以当前实现为准：

1. HBG 已具有正式的 L1 调用、ACLGraph capture/replay 和错误恢复路径，不再是“暂不支持”的占位能力。
2. 正常调用最终采用 AICore-first 的 Host enqueue 顺序，同时以共同 Start event 保持单算子边界，并为 AICPU enqueue 失败补充 Host cancel 与 join 闭包；不再采用早期文档中的简单 AICPU-first 描述。
3. HBG 的 slot/callable registry 已由 resident AICPU DSO global 状态迁移为 Context-owned device registry。resident DSO 仅临时 latch 当前 context registry 地址，不再拥有跨 context 的 registry 内容或 generation 状态。

### 0.2 规范用语

- “必须”表示违反后会破坏正确性、资源所有权或 ACLGraph 契约。
- “当前”表示本文对应上述两个提交的实现事实。
- “v1”表示当前单 context、无并发、静态调用签名的 L1 产品边界。
- “外部”表示 PyTorch、torch_npu、ACLGraph 或直接 C ABI 调用方。
- “Host enqueue 成功”不等于设备执行完成。

---

## 1. 背景与问题定义

PyPTO 原有两类运行形态：

- L3：单机多卡，由 PyPTO/Simpler 管理分布式执行、设备资源和跨卡协同；
- L2：单卡全资源掌控，由 PyPTO/Simpler 创建执行 stream、分配内存，并在运行生命周期内拥有设备级资源。

这两种形态都允许框架把设备视为自己的执行域。L1 的目标完全不同：

> 把一个 PyPTO program 表现成普通 AscendC 自定义算子：调用方传入当前 stream，PyPTO 只向该 stream 提交一个有界的异步算子序列，不切换设备、不接管 caller stream、不做内部同步、不感知 ACLGraph capture/replay，也不越过单算子边界。

因此，L1 不是把 L2 接口外面包一层 Python，而是重新划分所有权：

- caller 拥有设备、当前 stream、输入输出 tensor、ACLGraph 与外部同步；
- PyPTO context 拥有预先准备的内部执行状态、隐藏 AICore stream、内部 workspace、注册后的 binary/handle 和持久 device 元数据；
- CANN runtime 拥有已 enqueue task 的快照和 capture 后的图节点；
- PyPTO 不拥有 ACLGraph，也不查询 capture 状态。

### 1.1 为什么仍需要隐藏 AICore stream

PyPTO 的一个 program 由 AICPU orchestration 与 AICore worker 协同执行。对外必须只有一个普通算子形态，但内部仍需双分支：

- caller stream：承载与 PyTorch 前后算子的顺序、状态清理、AICPU task 和最终 join；
- hidden AICore stream：承载预注册 AICore kernel。

这两个内部执行分支通过 event 严格包在同一个算子边界中。隐藏 stream 是实现细节，不能被用户看到，也不能通过 model attach 偷跑到算子边界之外。

### 1.2 为什么 ACLGraph 透明是硬约束

如果 PyPTO 查询 capture 状态、取得 graph/model handle 或把内部 stream 手工挂入 model，单算子就会变成依赖 ACLGraph 私有语义的复合执行器。这样会带来：

- eager 与 capture 两套行为；
- graph owner 与资源 owner 混乱；
- 私有 stream task 可提前于 caller stream 的算子位置启动；
- 很难接入 taskQueue、vLLM 和普通 PyTorch 调度。

最终方案只提交 runtime 原生可捕获的 stream/event/kernel task。图是否在 capture、何时 replay、何时销毁，都由外部决定。

---

## 2. 目标、非目标与不变量

## 2.1 功能目标

1. 对外提供带 caller stream 的底层 L1 API。
2. 提供 PyTorch convenience wrapper，使调用形态接近普通自定义算子。
3. 默认适配 torch_npu taskQueue，不通过 Python 属性直接旁路队列。
4. 支持 `@pl.program` 的 TRB 与 HBG 编译产物。
5. 支持显式 `prepare -> warmup -> external sync -> capture -> replay`。
6. 支持同一 context 声明多个 program，且每个 callable 有独立 `func_id` 命名空间。
7. 保持 L2/L3 的 API、资源模型和 wire ABI 兼容。
8. init/prepare/launch/close 的失败都必须保持可诊断、可拒绝后续调用，并在可行时支持显式重试清理。

## 2.2 当前非目标

1. 不允许同一设备上多个 live L1 context。
2. 不允许 L1 调用并发执行；PyPTO 仍占用全部 AICore。
3. 不允许同一 context 内跨 stream 未 quiesce 切换。
4. 不提供运行期动态扩容 workspace、slot、callable table 或 HBG package capacity。
5. 不提供 capture 后修改 tensor 地址、scalar 或 HBG 拓扑的 graph update API。
6. 不提供 L1 内部 stream/device synchronize。
7. 不承诺外部 `from_blob`/自定义 allocator storage 在调用方提前销毁时仍安全。
8. 不把完全未 report 的硬件 core 失联恢复纳入算子内协议；该类故障交给 CANN op timeout、driver fault containment 或外部 device/context recovery。
9. 当前验收只以 A2/A3 为准；A5 与 A5 simulator 不作为本次完成条件。

## 2.3 十条硬不变量

1. **外部 stream 是唯一入口顺序源。** L1 不切换 current device，也不替 caller 选择 stream。
2. **launch 路径不分配或释放 device memory。** 所有 device 资源在 prepare 阶段固定。
3. **launch 路径不做 stream/device sync，不做 reset。**
4. **不查询 capture 状态，不取得 graph/model handle，不调用 `rtStreamAddToModel`。**
5. **AICPU task 位于 caller stream。** hidden stream 只承载 AICore 分支。
6. **两个分支必须在单算子边界内 fork/join。** caller 尾部必须等待 hidden AICore 完成。
7. **每次 Host 调用都有不可变参数快照。** 下一次 Host 调用不能覆盖尚未消费的 task args。
8. **图可见地址在 capture 前固定。** capture 期间禁止 lazy register、H2D staging、arena 增长和 registry 变更。
9. **close 不猜测设备是否空闲。** caller 必须先保证所有 eager/captured work quiescent 并销毁/reset graph。
10. **L1 与 L2/L3 模式互斥。** 借用设备的 L1 context 不能触发 L2 的 device reset/aclFinalize 路径。

---

## 3. L1、L2、L3 的资源模型

| 维度 | L3 | L2 | L1 |
| --- | --- | --- | --- |
| 设备范围 | 单机多卡 | 单卡全资源 | 单算子借用当前设备 |
| stream | 框架创建并管理 | 框架创建并管理 | caller 传入；仅内部创建 hidden AICore stream |
| device current | 框架可设置 | 框架可设置 | 必须由 caller 预先设置，PyPTO 不切换 |
| 输入输出内存 | 框架可分配/搬运 | 框架可分配/搬运 | caller tensor 地址，底层强校验 |
| workspace | 框架内部 | 框架内部 | 当前仍由 context 内部固定管理 |
| 同步 | 框架可做 | 框架可做 | 禁止内部 sync；全部由 caller 负责 |
| teardown | 可 reset/aclFinalize | 可 reset/aclFinalize | 禁止 reset/aclFinalize，只释放自有资源 |
| 并发 | 按原模式协议 | 框架全掌控 | v1 禁止并发；跨流切换需前序完成 |
| ACLGraph | 非核心边界 | 非核心边界 | 必须作为普通算子透明 capture/replay |

L1 没有把 workspace 改成外部参数，原因是当前 PyPTO 一次调用占用全部 AICore，v1 本就不允许并发。共享 context workspace 不会在合法调用中踩踏。将 workspace 外置会立刻扩大 Python/C ABI、容量协商和 graph lifetime 的复杂度，却不能解决当前并发问题，因此留作后续 slot 化演进。

---

## 4. 总体架构

```text
Python user / torch.compile integration
        |
        | pypto_init(programs, device, config)
        v
pypto.runtime.l1
  L1Context ---- L1Operator
        |
        | default: queue capsule + Tensor keepalive
        v
_torch_npu_l1 adapter
  getCurrentNPUStream().stream(false)
  RunOpApiV2(...)
  NPUCachingAllocator::recordStream(...)
        |
        | deferred C++ callback, no Python object/GIL
        v
Simpler ChipWorker / L1DispatchState
  immutable prepare/launch snapshot
  close-vs-callback serialization
        |
        | Simpler L1 C ABI
        v
DeviceRunnerBase / L1ExecutionState
  validate -> prepare persistent state -> enqueue fixed launch sequence
        |
        +---------------- caller stream ----------------+
        | clear/start/AICPU/join/tail                   |
        +---------------- hidden stream ----------------+
                         wait/start/AICore/done
        |
        +--> TRB: task window / ring buffer orchestration
        |
        +--> HBG: immutable graph package -> restore working slot -> execute
```

### 4.1 组件职责

| 组件 | 主要职责 | 明确不负责 |
| --- | --- | --- |
| `pypto.runtime.l1` | 用户 API、编译产物一致性、shape/dtype/layout/scalar 校验、prepare/warmup/close 协议 | capture 探测、内部 sync、设备切换 |
| `_torch_npu_l1` | taskQueue 排队、raw stream 获取、C++ Tensor lease、allocator stream 记录 | 编译、设备资源创建、Python callback |
| `ChipWorker` | 加载 runtime DSO、L1 dispatch lease、队列 callback 与 close 互斥、可重试 owner | tensor 语义、ACLGraph owner |
| Simpler L1 C ABI | 稳定的 init/prepare/launch/finalize 边界 | PyTorch 类型、taskQueue 策略 |
| `DeviceRunnerBase` | native 强校验、持久状态、stream/event 编排、TRB/HBG 分派 | caller sync、graph 句柄管理 |
| `L1ExecutionState` | L1 phase、hidden stream/events、device claim、fail-closed close | 析构时隐式 runtime 调用 |
| AICPU runtime | orchestration、HBG restore、scheduler 协调、错误汇合 | 跨算子提前启动 |
| AICore runtime | 执行 kernel、报告每核启动状态、响应 pre-window cancel | 直接感知 Python/ACLGraph |

### 4.2 模块边界为何放在 Simpler C ABI

PyPTO 负责编译语义和 Python 使用体验，Simpler 负责平台执行与设备资源。L1 所需的 stream/event/HostArgs/handle launch 都属于平台 runtime 能力，因此必须落在 Simpler，而不是在 Python 中拼接 ACL API。

反过来，torch_npu taskQueue 是可选的 PyTorch 集成策略，不应让 Simpler core 链接 torch。最终使用独立 `_torch_npu_l1` adapter 消费版本化 capsule，使 core ABI 与 PyTorch ABI 解耦。

---

## 5. 用户 API 与使用契约

## 5.1 Python API

```python
from pypto.runtime.l1 import L1Config, pypto_init

ctx = pypto_init(
    programs=[compiled_add, compiled_mul],
    device=0,
    config=L1Config(use_task_queue=True),
)

add = ctx.operator(compiled_add)
mul = ctx.operator(compiled_mul)

ctx.prepare()            # 一次性准备全部声明的 program
add.warmup(x, y, out=z)  # 仅表示成功 enqueue
torch.npu.synchronize()  # caller 明确完成 warmup

# 随后按 torch_npu ACLGraph API capture/replay。
# graph reset/destruction + 外部 quiescence 后才允许：
ctx.close()
```

API 约束：

1. `programs` 必须非空，且全部属于同一 onboard platform、同一 Simpler runtime。
2. `device` 必填，并且必须等于 torch_npu 当前设备；初始化不会替用户切设备。
3. 同一 context 声明的 program 必须在第一次 launch 前全部 prepare。`L1Operator.prepare()` 等价于 `L1Context.prepare()`。
4. 普通 eager 模式可以在首次调用时自动 prepare；ACLGraph 流程必须显式 prepare 与 warmup。
5. `prepared` 表示 prepare task 已成功 enqueue，`warmed` 表示至少一次 invocation 已成功 enqueue；二者都不表示 device complete。
6. pure Out 参数必须显式通过 `out=` 传入。wrapper 不替用户分配输出 tensor。
7. context 和 operator 线程亲和；跨线程使用直接拒绝。
8. context 不实现 Python context-manager 的自动 close，因为 `__exit__` 无法证明 graph 已销毁和设备已 quiesce。

## 5.2 taskQueue 默认路径

默认 `use_task_queue=True`。adapter 必须：

1. 用 `c10_npu::getCurrentNPUStream().stream(false)` 取得 raw stream，禁止使用会 drain taskQueue 的 `.stream()`；
2. 通过 `at_npu::native::OpCommand::RunOpApiV2` 把 L1 callback 作为普通队列 op 排入；
3. callback 只捕获 C++ POD、`shared_ptr` lease 和 `at::Tensor`，不得捕获 Python 对象；
4. 为每个唯一默认 NPU allocator storage 调用 `recordStream`；
5. 在真正 callback 执行前保留 tensor handle，防止队列尚未消费时 Python 引用释放；
6. callback 与 context close 通过共享 dispatch mutex 串行，避免 DSO unload 后调用悬空函数指针。

直接模式 `use_task_queue=False` 只用于 bring-up/debug。它读取 Python `current_stream().npu_stream`；全局 taskQueue 开启时该属性可能先 drain Host queue，因此不能作为生产路径的“零副作用 raw stream”方案。

## 5.3 外部 storage 契约

默认 NPU caching allocator 的 tensor 可通过 `recordStream` 把 storage 生命周期延长到设备使用完成。对于 `from_blob`、外部 allocator 或自定义 deleter，torch_npu allocator 可能无法记录该 storage；调用方必须自行保活到 stream 完成，capture 后还必须保活到 graph 最后一次 replay 完成并销毁。

## 5.4 底层 C ABI

核心接口为：

```text
simpler_l1_supported
simpler_l1_init
simpler_l1_prepare_callable
simpler_l1_launch
finalize_device
```

设计要点：

- caller stream 是显式参数；
- C ABI 只接收 POD/序列化 blob/device pointer，不接收 PyTorch 对象；
- init 声明 platform、runtime、线程/窗口参数和 context generation；
- prepare 上传 callable-local 静态状态并冻结容量；
- launch 只消费已准备的 callable 与调用快照；
- finalize 只释放该 L1 context 自己创建的资源，绝不 reset device 或调用 `aclFinalize`。

---

## 6. 状态机与所有权

## 6.1 execution mode

```text
Uninitialized
    | init L2                       | init L1
    v                               v
L2Owned                         L1Borrowed
    | finalize                      | explicit close
    +---------------+---------------+
                    v
                  Closed
```

`L2Owned` 与 `L1Borrowed` 互斥。L1 使用进程内 per-device claim，v1 同一设备只允许一个 live context。

## 6.2 L1 phase

```text
New -> Initializing -> Collecting -> ReadyEnqueued -> Sealed
          |               |              |              |
          +---------------+--------------+--------------+
                                  failure -> Poisoned

Collecting/ReadyEnqueued/Sealed/Poisoned
                  | begin_close
                  v
               Closing --retry--> Closing --success--> Closed
```

关键语义：

- `Collecting`：允许追加 prepare，但 launch 尚未 seal；
- `ReadyEnqueued`：prepare task 已排入 caller stream；
- `Sealed`：第一次 launch 后 callable/capacity 冻结，拒绝新增 prepare；
- `Poisoned`：执行期错误后拒绝 prepare/launch，但仍允许 close；
- `Closing`：第一项 destructive teardown 前即粘性进入，任何 prepare/launch 都 fail-closed；close 失败可重试；
- `Closed`：资源释放完成。

析构函数不调用 ACL/runtime API。若用户忘记 close，宁可告警并保留资源，也不能在未知 graph/stream 状态下隐式销毁。

## 6.3 operator 状态

每个 operator 保存：

- 对应的 compiled program/callable identity；
- 是否至少成功 enqueue 过一次；
- 首次成功 enqueue 后绑定的 tensor shape、dtype、stride 和方向；
- scalar 声明与 bit packing 规则。

失败的首次 enqueue 不得提交 `warmed` 或 layout binding。只有 adapter/native launch 返回成功后才原子提交候选 metadata。

## 6.4 资源所有权表

| 资源 | owner | 创建时机 | 可见于 graph | 释放条件 |
| --- | --- | --- | --- | --- |
| caller stream | PyTorch/caller | 外部 | 是 | 外部管理 |
| hidden AICore stream | L1 context | init | 是 | 外部 quiescence 后 close |
| Start/AicoreDone/PrepareTail/SerialTail event | L1 context | init | 是 | 外部 quiescence 后 close |
| AICPU/AICore binary handle | L1 context/runtime loader | init/prepare | 是 | close，失败保留 handle 重试 |
| Runtime/KernelArgs | L1 context | prepare | 是 | close |
| callable-local function table | L1 callable state | prepare | 间接可见 | close |
| workspace/arena/register windows | L1 context | prepare | 是 | close |
| `L1AicoreReport[]` | L1 context | prepare | 是 | close |
| queue call snapshot | taskQueue entry | 每次 Host 调用 | Host 队列可见 | callback 完成后释放 |
| CANN HostArgs copy | CANN task/graph node | enqueue/capture | 是 | task 或 graph owner 管理 |
| HBG host GraphPlan | 单次 HBG Host build | 每次 Host 调用 | 否 | 生成独立 launch blob 后释放 |
| HBG serialized launch blob | 单次 Host launch | 每次 Host 调用 | 被 CANN 深拷贝 | API 返回后可释放 |
| HBG working slot | L1 context | prepare | 是 | close |
| HBG ContextRegistry | L1 context | prepare | 是 | close |
| input/output tensor storage | caller；默认 allocator 可记录 stream | 外部 | 是 | task/graph 使用完毕 |

---

## 7. 初始化与 prepare

## 7.1 初始化前检查

Python 层在调用 native init 前完成：

1. adapter capsule ABI 版本校验；
2. torch 与 torch_npu build version 兼容检查；
3. device 与 current device 一致；
4. program 的 platform/runtime 一致；
5. scalar dtype 是否在 L1 bit-exact packer 支持表内；
6. assembled `ChipCallable` 的 tensor/scalar/direction metadata 与 Python 编译 metadata 一致；
7. HBG program 不包含 L1 禁止的 host tensor data access requirement。

init 在 native 建立部分资源后失败时，可能仍有必须显式清理的 owner。`L1InitializationError` 因此可携带 close-only `cleanup_context`。Python 两层 owner adoption 捕获 `BaseException`，包括 `KeyboardInterrupt`，防止 native context 已建立但 Python owner 丢失。

## 7.2 context 初始化

native init 依次建立：

1. 进程内 per-device L1 claim；
2. borrowed-device execution mode；
3. hidden AICore stream；
4. ACLGraph 可捕获的 event；
5. AICPU binary/entry；
6. runtime-specific resident/init 状态；
7. context generation 与基础 ownership table。

初始化不会设置设备、不会 reset、不会同步，也不会偷偷 prepare program。

四个执行 event 使用 `aclrtCreateEventExWithFlag(..., ACL_EVENT_SYNC)` 创建。A2/A3 上板表明普通 event 在 capture 内 record 可能返回错误；`ACL_EVENT_SYNC` 是当前跨 stream fork/join 可捕获的必要创建属性，不是额外 Host 同步。

## 7.3 为什么 prepare 全量声明 program

第一次 launch 会把 context seal。若 `op_a.prepare()` 只准备 a，而 b 在 graph capture 前才发现未准备，b 就无法补注册。最终规定：

> 任一 `L1Operator.prepare()` 都委托给 `L1Context.prepare()`，按确定顺序准备 `pypto_init(programs=[...])` 中的全部 program。

这同时保证 capture 前资源容量、binary handle、callable table 和 HBG slot 都已经固定。

## 7.4 prepare 的职责

prepare 可以 enqueue 异步 H2D/registration task，但必须在 caller stream 上建立 PrepareTail 边界。主要工作包括：

- 精确解析 callable blob 长度，拒绝截断、越界与重复 identity；
- 加载/注册 AICPU 与 AICore binary，缓存可直接 launch 的 handle；
- 为每个 callable 建立独立 `func_id -> device function address` 快照；
- 创建 Runtime、KernelArgs、arena、register windows、workspace；
- 分配每核独占 cache line 的 `L1AicoreReport`；
- 初始化 HBG working slot 与 ContextRegistry；
- 上传并回读关键 device pointer/metadata 做完整性检查；
- 冻结 worker count、容量、地址和 ABI metadata；
- 在 prepare caller stream 记录 PrepareTail。

所有纯 Host 输入错误应尽量在 allocation/upload 前返回，且不 poison context。发生部分资源创建后的失败则进入可清理状态，不能假装 init/prepare 从未发生。

## 7.5 prepare 与 capture stream 不同

推荐流程允许默认 stream prepare/warmup、另一条 stream capture。第一次 launch 会消费 PrepareTail；后续 capture 不再等待图外 PrepareTail。caller 必须在 warmup 后显式同步，保证 capture stream 切换前前序任务完成。

CANN 对 capture 中等待一个“在图外 record 过”的 event 会返回 capture dependency/isolation 错误，即使该 event 已经同步完成。因此实现不能简单地在每次调用中 wait 历史 tail，而是结合一次性 PrepareTail 消费、同 stream FIFO 与非阻塞 tail 完成查询处理 stream 切换。

---

## 8. L1 通用 launch 协议

## 8.1 launch 前置校验

底层在持有 `l1_operation_mutex` 的情况下再次检查：

- execution mode 必须是 `L1Borrowed`；
- phase 必须允许 dispatch，不能是 Poisoned/Closing/Closed；
- callable 已 prepare，identity 唯一且地址属于当前 context；
- caller stream 非空、device 匹配、线程/current-device 契约成立；
- tensor 个数、scalar 个数、方向、shape、dtype、stride 与已编译 metadata 一致；
- device pointer 确实是当前设备可访问的 device memory；
- descriptor 的 `extent_elem_cache`、contiguous 标记和实际 shape/stride 可达范围一致；
- `start_offset + reachable_bytes` 不越过 storage size，所有乘加做溢出保护；
- Runtime、KernelArgs、workspace、binary handle、report 区和 HBG slot 已准备；
- 同一 context 的 stream 切换满足前序 tail 已完成。

校验失败发生在任何 enqueue 之前时，不提交调用 metadata，也不改变 warmed/layout 状态。若调用已经部分 enqueue，再发生错误，则必须 poison context 并完成可表达的异步错误闭包。

## 8.2 最终 enqueue 顺序

当前固定序列如下：

```text
caller stream                              hidden AICore stream
-------------                              --------------------
[optional consume PrepareTail]
[stream-switch gate; no graph-external wait]
async clear launch state / handshake / report
record Start
                                           wait Start
                                           launch prepared AICore handle
                                           record AicoreDone
launch AICPU with HostArgs
wait AicoreDone
record SerialTail
return to Host (asynchronous)
```

三个容易误解的点：

1. **AICore-first 指 Host enqueue 顺序，不表示设备越过算子边界提前执行。** AICore 与 AICPU 都受 caller stream 的 Start event 约束。
2. **AICPU 仍在 caller stream。** caller 前序 torch op 一定先于 Start/AICPU，caller 后序 torch op 一定后于 AicoreDone wait。
3. **Host 返回只表示序列成功提交。** 设备可能仍在运行，tensor/context/graph 生命周期不能据此结束。

最终选择 AICore-first，是因为 custom AICPU scheduler 可能在等待 AICore startup report 时占据调度资源；若 AICore SQE 还排在后面，真实设备上可能形成调度环或长时间 stall。独立 report 修正了 cache ownership 后，AICore-first 仍然有更好的失败闭包：hidden wait、AICore launch 和 done-record 都在 AICPU task 入队前完成 Host API 提交检查。

## 8.3 正常路径为何可被 ACLGraph 捕获

整个序列只包含：

- async memset；
- event record/wait；
- 已注册 AICore handle launch；
- AICPU `aclrtLaunchKernelWithHostArgs`；
- caller/hidden stream 上的普通 runtime task。

它不包含：

- device/stream synchronize；
- `aclrtSetDevice`、device reset、`aclFinalize`；
- binary lazy register；
- device alloc/free；
- capture status query；
- graph/model handle query；
- `rtStreamAddToModel` 或等价 attach；
- capture 分支专有逻辑。

因此 CANN 只看到一组正常 stream task，并自行把 event 依赖和两条 stream 的节点纳入图。

## 8.4 hidden done-record 或 AICPU enqueue 失败的 Host cancel 闭包

AICore-first 带来两个必须显式处理的异常：hidden AICore 已成功 enqueue 后，AicoreDone event 的首次 record 可能失败，或 AICPU Host launch 同步返回失败。没有有效 done 节点或 AICPU 时，AICore 可能永远等不到 register window/device cancel，caller 也没有可 join 的 completion 证明。

当前错误闭包为：

1. hidden AICore launch 成功后，立即 enqueue `record(AicoreDone)`；
2. 首次 done-record 失败时，先发布 Host cancel，再重试一次 done-record 以建立 join 节点；
3. done-record 成功但 AICPU launch 同步失败时，同样发布 Host cancel；
4. caller stream 对 launch-state/legacy handshake span 执行 `0xFF` async memset；
5. `0xFFFFFFFF` 被 AICore pre-window poll 识别为 Host cancel；
6. AICore 退出，hidden stream 完成 AicoreDone record；
7. caller stream wait AicoreDone，并记录 SerialTail；
8. 返回原始错误，同时 context 进入 Poisoned。

这里使用 byte-fill `0xFF`，不能用数值 2 的普通 byte memset，否则首个 `uint32_t` 会变成 `0x02020202`。失败补偿本身仍是异步 enqueue，不做内部 sync。若补偿 enqueue 也失败，context 必须保留 Poisoned/Closing ownership，不能宣称资源已安全回收。

该协议建立在 CANN 的常规假设上：Host launch API 同步失败表示 AICPU task 没有部分入队。若未来 runtime 明确允许“返回失败但 task 已入队”，需要加入 task identity/ack 级别的新协议。

## 8.5 跨 stream 串行化

v1 不支持同一 context 的并发调用。Host mutex 只能串行 enqueue，不能证明两个不同 device stream 上的前后调用已经完成，因此还需 device-side约束：

- 同一 raw stream：依赖 FIFO，无需等待历史 SerialTail；
- raw stream 变化：先非阻塞查询旧 SerialTail 是否 complete；未完成则直接拒绝；
- warmup 后换 capture stream：caller 先外部 sync，查询得到 complete，允许切换；
- 不把旧 SerialTail 的 wait 导入新 capture 图，避免 107024；
- graph replay 后切回 eager、两张 graph 交替或并发 replay：caller 必须在切换前外部 quiesce。

event status query不是 capture query，也不是同步；它只用于 fail-closed 地证明 eager 前序 tail 已完成。capture clone/replay 对原 event 的可见性有限，因此不能把它误写成通用并发检测器。

---

## 9. 调用参数、scalar 与 tensor 生命周期

## 9.1 为什么每次调用必须有 task-owned snapshot

Host 连续异步调用时，下一次调用可能在设备消费前到来。以下内容都不能放在一块会被覆盖的共享 Host 缓冲区中：

- callable id；
- tensor device 地址与 shape/stride/offset；
- scalar bit pattern；
- runtime-specific invocation metadata；
- TRB callable-local function binding；
- HBG graph package identity 与 inline payload。

`aclrtLaunchKernelWithHostArgs` 会复制传入的完整 args bytes。TRB 因此把每次调用的 `L1AicpuInvocationArgs` 作为 HostArgs 快照交给 runtime；HBG 把序列化 graph package作为 inline HostArgs payload交给 runtime。共享 workspace 可以只有一份，但 task 参数快照必须逐 task 独立。

## 9.2 TRB invocation ABI

TRB L1 invocation 包含：

- versioned header；
- persistent `KernelArgs` 的值拷贝；
- callable id；
- `ChipStorageTaskArgs`，包括 tensor/scalar 调用快照；
- runtime/callable identity 的必要字段。

每个 callable 另有 prepare-time `L1RegisterCallableArgs`，持有自己的 `func_id -> device function address` 快照。两个 program 即使都从 `func_id=0` 开始，也不会落入一张 context-global 表产生冲突。

CANN task args pool只保证普通对齐，不能假设满足 `alignas(64)`。AICPU entry 先以小 prefix/memcpy 读取 invocation，再复制到对齐的 context snapshot，禁止直接把 runtime args pointer 强转成 over-aligned struct。

## 9.3 scalar ABI

PyPTO orchestration signature 历史上只包含 tensor，scalar 不在 `sig_count` 中。当前 `scalar_count` 作为独立 metadata 沿编译、assemble、ChipCallable 和 native state 贯通，并放在兼容旧 ChipCallable layout 的 header tail padding 中，避免破坏 L2/L3 wire offset。

Python 按声明 dtype 做 bit-exact packing：

- integer/unsigned/bool 按目标宽度与符号规则；
- FP16 的 `1.0` 打包为低 16 位 `0x3c00`；
- BF16 的 `1.0` 打包为低 16 位 `0x3f80`；
- FP32/FP64 使用对应 IEEE bit pattern；
- 不支持的 FP4/FP8/HF 等类型在 init 阶段 fail-fast，而不是首次 launch 时静默误读。

capture 时 scalar 是否变化是 ACLGraph 使用者的问题。v1 capture 的节点持有当次调用快照，replay 不重新进入 Python 读取新 scalar。

## 9.4 tensor metadata 绑定

当前 v1 是静态调用签名：

- 编译得到的 tensor shape/dtype/direction 固定；
- 首次成功 enqueue 后进一步绑定实际 stride/layout；
- 后续调用必须匹配；
- 输入输出地址可以在不同 eager 调用中变化；
- capture 后 replay 使用 capture 节点持有的地址，除非 ACLGraph 自身提供并正确使用地址更新机制，PyPTO 不感知更新。

这里的“静态”不否认 PyPTO 编译器内部可用动态 shape 表达。正如 AscendC tiling 一样，capture 的是某次已经完成 tiling/参数固化的 task；PyPTO L1 不需要知道用户如何产生这组静态调用参数。未来若开放运行期 args 更新，必须先定义 graph update 与 HBG package 重建协议。

## 9.5 Tensor lease 的两个阶段

taskQueue 下 tensor 生命周期有两个不同窗口：

1. **Host queue 等待窗口**：adapter 捕获 `std::vector<at::Tensor>`，防止 callback 执行前 storage 被释放；
2. **device 异步执行窗口**：adapter 对默认 allocator storage 调 `recordStream`，防止 callback 返回后、device 使用完成前 storage 被 allocator 复用。

只做其中一项都不够。capture 后还需 caller 保留 graph/context/tensor owner 到 graph 销毁，因为 allocator 记录解决的是 stream task 使用期，不等价于拥有 graph 对象。

---

## 10. AICore startup report 与 no-reset 错误协议

## 10.1 为什么不能继续复用 legacy Handshake report 字段

旧 Handshake 把以下字段放在同一个 64B cache line：

- AICore 写的 `physical_core_id/core_type/aicore_done`；
- AICPU 写的 task/window/CANCEL 字段。

L1 中 AICPU 与 AICore 真正并行启动。AICPU 对混合所有权 cache line 做 `dc civac` 轮询时，可能把自己的旧脏 cache line 回写并覆盖 AICore 刚发布的 report。L2 的 AICore-first 历史顺序通常让 AICPU 首读即看到 report，不能证明这种共享 cache ownership 正确。

## 10.2 独立 `L1AicoreReport`

最终方案为每个 worker 分配一个独占 64B cache line：

```text
L1AicoreReport[0]  -- only AICore 0 writes, AICPU reads
L1AicoreReport[1]  -- only AICore 1 writes, AICPU reads
...
```

要求：

- 每核一整条 cache line，不能把多个核压在同一 64B line；
- AICore 是唯一 writer，AICPU 只读并按平台协议 invalidate；
- report base 只存一份，作为 `uint64_t` device address 放进 Runtime device-visible image；
- AICPU 与 AICore 从同一 trusted Runtime 取得地址，避免双源 split-brain；
- L2/L3 Runtime constructor 显式把该地址初始化为 0，走 legacy Handshake；
- 每次 launch 的 async clear 必须同时覆盖 legacy handshake 与 report span，并发生在 Start event 前；
- report 容量按实际 worker count 计算并做 64B alignment/overflow 检查。

legacy Handshake 仍保留 task 发布与 CANCEL 通道；L1 不再依赖其中的 report 字段。completion/DFX 使用 scheduler 初始化时复制出的稳定 `core_type_compact_`，避免后续再读已清零的旧 report。

## 10.3 pre-window cancel

AICore 在 register window 尚未打开时必须能退出，否则 borrowed-device L1 不能依赖 reset 回收。当前协议复用 legacy `aicpu_ready`：

- `0`：WAIT；
- `2`：AICPU device-side CANCEL；
- `0xFFFFFFFF`：Host enqueue-failure CANCEL；
- 正常成功仍以 register window 非零为唯一放行条件，不恢复旧的 AICPU→AICore ready round-trip。

AICore 先快速轮询 register window，只以较低频率 invalidate/check CANCEL，避免把历史上约数微秒的 cache round-trip重新带回正常 preamble。CANCEL 保持到整个 hidden AICore kernel 完成；AicoreDone event 是 collective completion/ack，v1 不需要每核另加 ACK。

该协议覆盖：

- physical core id 越界；
- physical id 在范围内但 register mapping 为 0；
- scheduler init/assign 失败；
- HBG prelaunch validation 失败；
- AICPU Host launch 同步失败。

它不覆盖“某 AICore 完全没有进入且从不 report”的硬件失联，因为 AICPU 无法区分慢启动与永久丢失。对这种情况使用 CANN op timeout/driver fault containment，不在 L1 内部擅自 reset。

---

## 11. binary、callable 与 workspace

## 11.1 AICore binary

当前 PyPTO/Simpler 自己管理 incore `kernel_bin`，没有把全部生命周期委托给公开 `aclrtRegisterBin` 风格的 runtime owner。L1 prepare 提前完成注册并缓存 launch handle；launch 只允许“handle 已存在”分支，不能触发 lazy register。

v1 对 kernel binary device memory 采用 context-lifetime 累积策略：在 close 前不复用/回收单个 binary。由于 context/program 数量有上限且不允许并发，这比在尚无 task completion owner 的情况下实现错误复用更安全。

## 11.2 callable identity 与函数地址

`func_id` 是单个 compiled program 内的局部编号，不是 context-global id。每个 callable 持有自己的函数地址快照和 identity hash。HBG invocation 还携带 callable、argument snapshot 和 function binding hash，防止 package 与错误 slot/callable 配对。

prepare 后 registry append-only，第一次 launch 后 context seal。任何 duplicate identity、同 id 不同地址、缺失 handle 或 ABI 不一致都在 enqueue 前拒绝。

## 11.3 workspace

当前 workspace：

- 由 PyPTO context 内部分配；
- prepare 时固定地址与容量；
- TRB/HBG 调用共享；
- 依赖 v1 无并发契约防止踩踏；
- capture/replay 期间不能扩容或换地址；
- close 前不释放。

未来要支持并发，不能只把 workspace 参数外置；还必须把 Runtime、KernelArgs、handshake/report、HBG working slot 和 completion fence 一起 slot 化。单独外置 workspace 会给用户造成“已经可并发”的错误暗示。

---

## 12. teardown、失败所有权与重试

## 12.1 close 前置条件

调用 `L1Context.close()` 前，caller 必须：

1. 停止新的 L1 enqueue；
2. 等待所有 eager task 完成；
3. 等待所有 graph replay 完成；
4. reset/destroy 持有 L1 node 的 ACLGraph；
5. 保持 context、binary、tensor storage 和 HBG source owner 到上述步骤完成。

close 自身不做 synchronize，也不查询 graph owner。用户层不暴露 reset 是因为 reset 不是正常生命周期必需动作，且错误 reset 会破坏同进程其他框架资源。

## 12.2 fail-closed teardown

native close 在第一项 destructive teardown 前进入 `Closing`。随后：

- prepare/launch 全部拒绝；
- binary unload、device free、event/stream destroy 逐项记录错误；
- 某项失败时保留其 handle/ownership table，不清空后伪装成功；
- hidden execution state 与 per-device claim 只在所有外层资源成功释放后关闭；
- `ChipWorker` 只有 native finalize 成功后才 dlclose runtime DSO；
- close 失败保留 dispatch state 和 DSO，可显式重试；
- Python wrapper 不因异常丢失 cleanup owner。

L2 的历史 teardown 允许 best-effort 后 reset device。L1 的“失败保留 owner”语义不能无条件改变 L2 loader/allocator 的析构行为，否则可能在 RTS 已 reset 后再次调用 unload/free。实现按 execution mode 区分二者。

## 12.3 callback 与 close 竞争

taskQueue callback 可能晚于 Python `op()` 返回。`L1DispatchState` 使用共享 mutex：

- callback 取得 mutex，检查 `closed`、context/function pointer，然后执行 native launch；
- finalize 取得同一 mutex，先把 dispatch 标成 closed，再执行 native teardown；
- finalize 不会在 callback 正在使用 DSO 时 dlclose；
- callback 不会在 finalize 后调用悬空函数指针。

直接 C ABI 不能依赖这一层保护，因此 native phase 自身也必须在 operation mutex 下拒绝 Closing/Poisoned。

---

## 13. L2/L3 隔离与兼容

L1 改造遵循“新增 borrowed mode，不重写 owned mode”的原则。

### 13.1 ABI 兼容

- `ChipCallable.scalar_count` 放入旧 header 尾部 padding；旧字段和 storage offset 保持不变；
- 保留旧 C++ factory overload；
- legacy blob padding 解读为 `scalar_count=0`；
- L3 payload/version 不因 L1 增量被静默改义；
- L1 queue capsule、TRB invocation、HBG package各自有独立 ABI/version，不复用 L2 wire。

### 13.2 资源隔离

- L1 AICPU entry/symbol只在 L1 mode 使用，不强迫 L2 loader解析 L1-only symbol；
- L1 大型 metadata按需分配，不无条件嵌入每个 L2 callable；
- `L1AicoreReport` 地址在普通 Runtime constructor 中为 0，L2/L3继续 legacy handshake；
- L1 hidden stream/events/context registry只在 `L1Borrowed` 创建；
- L2/L3 finalize 保留各自既有 reset/allocator 语义；
- runtime artifact/cache key包含 TRB/HBG runtime，显式输出目录有 runtime owner guard，避免两个 runtime 覆盖同名延迟加载产物。

### 13.3 当前回归结论边界

A2/A3 L3 回归已经通过记录中的完整集合。L2 大规模回归中存在一个可独立复现的历史高性能 case stall，不能写成“L2 全绿”，也没有证据把它归因于 L1。清晰的结论应是：已覆盖的普通 L2/L3 路径未发现 L1 引入的 ABI/资源回归；异常 case 保留独立问题跟踪。

---

## 14. TRB L1 设计

TRB 是第一阶段最小可用 runtime，也是通用 L1 stream/event/task snapshot 协议的基础。

## 14.1 prepare-time 静态状态

TRB prepare 固定：

- worker count 与 AICPU affinity；
- task window、heap、dependency pool 容量；
- Runtime 与 KernelArgs device 地址；
- AICore register window；
- callable-local function binding；
- AICore binary handle；
- workspace/arena；
- per-core `L1AicoreReport`。

这些状态可被多次 eager 调用和 captured node复用，但不能在 context 内并发消费。

## 14.2 每次调用

每次 TRB Host 调用只构造 fixed-size invocation snapshot，交给 AICPU `WithHostArgs`。AICPU 从 snapshot 取得 tensor/scalar/callable id，再使用 prepare-time Runtime 与 function table 启动 orchestration。AICore 从 prepare-time device KernelArgs 取得同一个 Runtime 地址。

共享 Runtime 是可变执行状态，因此每次 launch 在 Start 前清理 handshake/report，AICPU completion gate 在本轮所有参与者 arrive、唯一 finalizer 完成、所有线程读取最终状态、所有线程 depart 后，才允许最后离开者 deinit/reset本轮共享字段。这样避免迟到线程读取已被下一代清零的 error/runtime 状态。

## 14.3 失败汇合

所有已经进入本轮的有效 AICPU participant 必须 exactly-once 进入 completion gate。scheduler init 失败不能让部分线程绕过 arrive；orchestrator 也不能在 scheduler 最终裁决前进入可能无限 submit 的函数。最终协议让 scheduler 完成 handshake/assign 后汇合，唯一 leader发布 init verdict，orchestrator在进入 program前 acquire 等待 verdict。

这一协议的目的不是提高模拟器测试覆盖，而是在 borrowed-device 模式下确保错误后仍能让 AICPU/AICore task退出，不能借助 device reset掩盖漏 arrival 或永不退出的 core。

---

## 15. HBG L1：graph 作为 tiling-like task 参数

HBG 的核心不是“多一种 runtime 字符串”，而是把 Host 动态构建出的 scheduler graph 安全地变成一个可被 CANN task/ACLGraph node持有、可重复 replay、且不会被后续调用覆盖的参数包。

可以用 AscendC tiling 理解它：

- tiling_data 描述某次 kernel invocation 的执行参数；
- HBG package 描述某次 PyPTO invocation 的调度图、初始 scheduler memory 和函数/参数绑定；
- capture 时它必须随 task 固化；
- replay 时 Host 不重新 build，但设备必须从同一 pristine image恢复可变执行状态；
- package 的生命周期至少覆盖 task，capture 后覆盖 graph node。

但 HBG 比普通 tiling 多一个关键难点：图像在执行中会被原地消费和修改，不能把 runtime-owned source直接当工作区执行。

## 15.1 为什么“上传一次 graph 地址并永久复用”不正确

HBG scheduler 会修改：

- wake list；
- task state/completion flags；
- ready queue；
- completed-subtask 计数；
- watermark；
- scheduler queue pointer；
- Runtime mailbox、SM handle 等运行时字段。

一次执行结束后，working image已经不是初始图。若 ACLGraph 第二次 replay 只复用同一个 device graph 地址而不 restore，会读取已消费或已 destroy 的状态。因此必须把“不可变初始模板”和“可变执行槽”分开。

## 15.2 五层生命周期模型

```text
Layer 1: Host immutable HbgGraphPlan
           |
           | serialize fresh writable bytes per invocation
           v
Layer 2: Host writable serialized launch blob
           |
           | aclrtLaunchKernelWithHostArgs + placeholder
           v
Layer 3: CANN task/graph-node-owned immutable device args snapshot
           |
           | AICPU leader validates + restores every invocation/replay
           v
Layer 4: Context-owned mutable HBG execution slot
           |
           | scheduler/AICore execute and mutate
           v
Layer 5: Context/caller lifetime roots
         binaries, registry, workspace, tensors, events, graph owner
```

### Layer 1：不可变 Host GraphPlan

每次 HBG host build产生一个 `HbgGraphPlan`。它深拷贝所有 pristine region，并保存 canonical、未 patch 的序列化表示。对象不暴露 mutable byte view，可以生成任意多个彼此独立的 writable launch snapshot。

它是 plan hash 的 Host trust root，但不是 CANN task owner。构建失败事务化返回，不改变旧 owner。

### Layer 2：单次 writable launch blob

CANN 的 placeholder patch可能修改传入 HostArgs，因此不能把 canonical plan本身直接交给 launch API。每次 invocation 都从 plan深拷贝一份 writable blob。

该 blob 在 Host API 调用期间有效；一旦 CANN 完成 HostArgs copy，PyPTO 可以释放这份临时 Host bytes。它不能被下一次 Host build复用或覆盖。

### Layer 3：CANN runtime-owned task/node snapshot

`aclrtLaunchKernelWithHostArgs` 复制完整 `[hostArgs, hostArgs + argsSize)`。`aclrtPlaceHolderInfo{addrOffset, dataOffset}` 把 header 中的 `inline_payload_addr` patch 为 runtime-owned device args base 加 `header_size`，使 device header内的 source pointer指向同一份 args copy里的 inline pristine payload。

这与 AscendC 的“指针字段 + inline tiling bytes + runtime patch”同类：

- eager task由 CANN 持有自己的 args snapshot；
- captured node由 ACLGraph持有自己的 args snapshot；
- 不需要 PyPTO猜测 CANN 内部 task args pool 的环大小；
- 不依赖约 2048 次 launch 或任何内部固定上限；
- package总长度用 `uint32_t` wire边界校验，但不把源码中的内部最大值当公开规格。

runtime-owned args只拥有这份内联 bytes，不自动拥有 tensor storage、binary、workspace、working slot 或 context。后者仍由 caller/context保活。

### Layer 4：Context-owned mutable execution slot

working slot包含固定地址的：

- full shared-memory image destination；
- full runtime-arena destination；
- optional GM initializer destination；
- outer Runtime/KernelArgs；
- heap/workspace；
- scheduler execution state与completion fence。

每次 invocation/replay，AICPU唯一 restore leader都从 Layer 3 pristine source把完整必要 region恢复到该 slot，验证并发布成功后，其他 participant才允许 classify/dispatch。

v1 只有一个 working slot。图中连续两个 L1 node因 caller stream join自然串行，可以复用；两张图并发 replay不允许。未来并发必须引入多个完整 `HbgExecutionSlot`，不能只复制 graph blob。

### Layer 5：Context 与 caller lifetime roots

context保活：

- working slot、Runtime、KernelArgs、workspace；
- ContextRegistry；
- AICPU/AICore binary；
- event与hidden stream；
- callable function binding。

caller保活：

- ACLGraph owner；
- graph绑定的 input/output storage；
- L1Context本身；
- 外部 allocator owner。

只有 graph replay结束、外部 sync、graph reset/destroy后，才能 close context。

## 15.3 HBG launch blob wire format

```text
HbgLaunchBlobHeader (160 B, ABI 1.2)
HbgLaunchRegion[region_count] (40 B each)
8-byte canonical padding
immutable inline payload
```

### Execution binding（64 B）

包含：

- shared memory base/capacity；
- runtime arena base/capacity；
- GM heap base/capacity；
- runtime offset；
- slot generation。

package是 destination-bound。其 pristine image内包含重定位后的绝对 device address，不能在不重新 relocation的情况下复制到另一组 slot地址。binding与当前注册 slot不一致必须拒绝。

### Invocation identity（40 B）

包含：

- callable hash；
- argument snapshot hash；
- function binding hash；
- tensor/scalar count；
- host total task count；
- context-global callable id。

三个 hash属于不同 trust domain，不能合并成“看起来唯一”的单个 Host id。device restore前必须逐项互证。

### Regions

当前 region kind：

1. `SharedMemoryImage`；
2. `RuntimeArenaImage`；
3. 可选 `GmHeapInitializer`。

full SM 与 full runtime arena是强制 region，且必须覆盖 frozen capacity。只恢复本次看起来使用过的前缀是不安全的，因为未覆盖尾部可能残留上次 replay的 completion/queue状态。

验证包括：

- magic、ABI major/minor、header/total size；
- canonical alignment与 region count上限；
- binding窗口非空、加法不溢出；
- source/destination range不越界、不重叠；
- full-image要求；
- identity匹配；
- generation非零且匹配；
- plan hash覆盖 identity、region descriptor和完整 payload；
- patched/unpatched address mode正确。

## 15.4 Host build 能读取什么

HBG L1 host builder发生在调用/capture线程，caller stream上的前序 torch task可能尚未执行。即使 device tensor地址可以被 Host mapping，也不表示前序写已经按 stream语义完成。因此 HBG L1明确禁止 host build读取或写入外部 device tensor数据。

允许用于构图的信息：

- tensor metadata；
- device address作为不解引用的标识/参数；
- Host scalar；
- 编译后的拓扑与function binding。

禁止：

- `get_tensor_data`；
- `set_tensor_data`；
- 任何依赖外部 tensor运行时数值决定拓扑的 Host 行为。

编译器给 orchestration SO导出 requirements metadata。HBG L1 prepare读取并 fail-closed；TRB/L2不因该检查改变原行为。

## 15.5 Context-owned HBG registry

每个 HBG L1 context在 device memory中拥有一个 cache-line aligned `HbgContextRegistry`：

```text
HbgContextRegistry
  magic / ABI / struct_size / context_generation
  HbgExecutionSlotRegistry
  HbgCallableRegistry
```

DeviceRunner在 prepare创建、初始化和注册它，并持有到显式 close成功。AICPU resident DSO只保存当前 registry device address，不能保存：

- slot内容；
- callable内容；
- 上一个 context generation；
- 跨 context conflict状态。

这一设计取代了早期 resident-global registry。后者在第一个 context close后仍随 resident scheduler DSO存活，第二个 context可能遇到 stale/conflict；用 Host generation反复 reset resident global既难证明所有线程不再访问，也把资源 owner放错了层级。

ContextRegistry让生命周期重新对齐：谁创建 context，谁拥有 registry；context close释放它；下一 context得到全新的地址与内容。v1仍限定单 Host进程、单 live context。跨 Host进程顺序复用 resident DSO不在当前 generation唯一性保证内。

## 15.6 HBG 每次 invocation/replay 的 device 协议

```text
1. AICPU entry取得当前 ContextRegistry地址
2. 校验registry header/context generation
3. 按callable_id取得不可变CallableRegistration
4. 取得唯一ExecutionSlotRegistration
5. 校验HostArgs header/binding/identity/hash/region
6. 所有participant汇合；唯一leader restore完整pristine regions
7. leader发布restore verdict（release）
8. peers acquire读取；失败共同进入error epilogue
9. scheduler init/assign最终裁决
10. orchestration classify/dispatch
11. shutdown/runtime destroy
12. arrive -> unique finalize -> snapshot result -> depart
13. last-depart cleanup本轮可变executor状态
```

replay时步骤完全相同。Host不会再次 build图，但 CANN重放 AICPU task，leader从 graph-node-owned source重新 restore工作区，所以第二次及后续 replay不会继承已消费 scheduler状态。

## 15.7 多 callable 与多 node

- callable registry按 context-global callable id索引；
- 每个 callable registration拥有自己的 function table/hash；
- graph内部 function id仍是 callable-local；
- add与mul都可以有 `func_id=0`；
- 每个 captured node拥有自己的 HostArgs graph package与argument snapshot；
- 两个 node可以共享同一个 context working slot，但必须由 caller stream顺序和内部 join保证不重叠；
- 同一 callable在一张图中出现两次，也必须有两个独立 task/node snapshot，不能让后一个 Host build覆盖前一个。

## 15.8 HBG prelaunch control 与错误闭包

HBG 在 AICPU完整进入 runtime generation前仍有 slot、blob、ABI、affinity、KernelArgs、platform bridge等校验失败点。hidden AICore已经 enqueue后，这些错误不能直接 `return`，否则 AICore会永久等window。

当前使用独立 prelaunch control/cache line和 trusted Runtime override：

- caller在Start前清控制线；
- AICPU在可信 registry/slot存在时向对应control发布CANCEL并flush；
- AICore从Host提供的 trusted Runtime地址取得同一control/report视图，不信任可能损坏的 package内地址；
- slot registry acquire失败仍可使用 init阶段 latch 的 fallback control地址；
- affinity输入越界、Runtime地址不一致、physical core mapping失败都进入统一cancel/epilogue；
- 不允许平台入口静默 early return。

HBG participant错误汇合还区分“测试注入的合成错误”和真实 shutdown/runtime destroy错误。真实 teardown错误优先，不能因为本次是controlled fault就吞掉 unexpected cleanup failure。

## 15.9 HBG 容量与回收

prepare固定 working slot capacity。若某次 host build需要更大的 full SM/arena/package：

- v1直接拒绝；
- 不在 capture内扩容；
- 不原地迁移已有 destination-bound package；
- 用户创建更大容量的新 context，且先按协议销毁旧 graph/context。

CANN task args容量通过真实 probe验证，不把约 2048 launch或源码内某个 256 MiB常量写入产品规格。捕获后的 package无法由 PyPTO通过原 event query判断“graph永远不会再 replay”，所以回收由 ACLGraph owner负责；没有通用 retain/release回调时，宁可保留到 graph destroy/context close，也不能按 launch计数提前复用。

---

## 16. ACLGraph 生命周期

## 16.1 标准流程

```text
compile TRB or HBG programs
        |
pypto_init(all programs, current device)
        |
context.prepare() on eager stream
        |
operator.warmup(...) on eager stream
        |
caller external synchronize
        |
create/use capture stream and begin capture
        |
torch predecessor -> L1 node(s) -> torch successor
        |
end capture
        |
replay N times; validate results
        |
external synchronize
        |
graph.reset()/destroy
        |
context.close()
```

## 16.2 capture 时发生什么

TRB：

- Python/adapter创建本次 tensor/scalar snapshot；
- CANN捕获 caller/hidden stream event依赖与两类 kernel task；
- AICPU HostArgs成为 node-owned task args。

HBG：

- Host builder在 capture调用时运行一次；
- 生成 immutable plan和fresh writable blob；
- CANN复制整个 blob并patch inline payload pointer；
- captured AICPU node持有该 pristine graph source；
- replay不再调用 Host builder，只做 device restore和execute。

## 16.3 replay 时能变与不能变的内容

当前 v1 replay固定：

- tensor device address；
- scalar bit pattern；
- callable/function binding；
- HBG topology、task count与pristine memory image；
- working slot binding与capacity。

外部可以在同一 tensor storage中写入新数值，再 replay图；这不会改变地址。若 ACLGraph提供地址更新能力，PyPTO当前也没有同步更新HBG argument hash/package的协议，不能默认它对HBG安全。

## 16.4 多图规则

同一个 context可以先后 capture多张图，但：

- capture/replay必须串行；
- 两张图切换前由 caller保证前一张 quiescent；
- graph node各自持有自己的 HostArgs source；
- context working slot/workspace仍只有一份；
- graph destroy前不得 close context；
- 不支持两条 stream同时 replay两个 graph。

## 16.5 为什么没有 `reset()` 用户 API

正常 warmup到capture只需要 caller同步与 stream-switch gate，不需要 PyPTO reset内部状态。显式 reset API会诱导用户在 graph仍持有地址时清理binary/slot/event，或在错误后尝试复用半拆状态。当前正确动作只有：正常继续调用，或停止调用并在外部quiescence后close；Poisoned context不能 reset回Ready。

---

## 17. 与 pto2 历史实现的对比

本节对比的历史代码位于工作区 `/mnt/workspace/inductor/pto2/pypto`。它已经实现过类似 AscendC kernel launch 的基本 L1 和 ACLGraph适配，因此提供了重要工程参考；但其性能策略和资源边界与本次目标不同，不能直接移植。

## 17.1 pto2 的核心调用拓扑

pto2 的 `StreamContext` 内部创建：

- AICore stream；
- scheduler AICPU stream；
- controller AICPU stream。

在 L1/capture 路径中，它大体采用：

```text
caller/capture stream
  |
  | query capture info / get model
  | attach private streams to capture model
  |
  +---- private ctrl AICPU stream
  +---- private sched AICPU stream
  +---- caller/current stream launches AICore
```

`DeviceLauncher::SetCaptureStream` 会查询 stream capture信息，并调用 `RuntimeStreamAddToModel`/`rtStreamAddToModel` 一类接口把私有 stream挂入 capture model。capture early模式还会跳过 caller到私有AICPU stream的 pre-sync，让 orchestration有机会提前于 caller stream上该“kernel”位置展开。

这能带来性能收益：AICPU提前准备，AICore到达时可能减少等待。但它越过了单个 kernel的架构边界，因此不符合本次 L1定义。

## 17.2 逐项对比

| 维度 | pto2 历史实现 | 当前 PyPTO L1 | 结论 |
| --- | --- | --- | --- |
| 对外形态 | 已有 external/current stream 的 kernel-like launch | `pypto_init` + `L1Operator`，底层显式 caller stream | 继承“外部 stream”原则 |
| AICPU stream | 内部 ctrl/sched 私有 stream | 直接使用 caller stream | 当前边界更像普通算子 |
| AICore stream | caller/current stream | hidden AICore stream | 内部实现不同，但对外仍单算子 |
| capture 感知 | 查询 capture/model | 完全不查询 | 明确不继承 |
| 私有 stream 入图 | `rtStreamAddToModel` | 只靠普通 event依赖被CANN捕获 | 明确不继承 |
| early orchestration | capture下可跳过pre-sync、提前运行 | 两分支都受本次Start gate约束 | 明确反对跨算子边界优化 |
| eager sync | 某些路径可 device/stream sync | launch内禁止任何sync | 当前契约更严格 |
| launch顺序 | AICPU后AICore或私有AICPU提前 | Host enqueue AICore后AICPU，共同Start，尾部join | 当前有完整失败闭包 |
| HostArgs | 已使用AICPU HostArgs | TRB/HBG都使用runtime-owned快照 | 继承并版本化 |
| AICore launch | handle/binary launch | prepare注册handle，launch直接使用 | 继承，禁止lazy register |
| workspace | 外部传入/调用侧参与 | 当前context内部固定 | 本次有意不同 |
| taskQueue | 非当前生产级边界 | `.stream(false)` + `RunOpApiV2` + Tensor lease | 当前更适合PyTorch生产接入 |
| callable id | 历史模型不同 | callable-local func_id表与identity hash | 当前支持多program隔离 |
| teardown | 与历史上下文/同步耦合 | borrowed mode、无reset、失败可重试owner | 当前更严格 |
| HBG生命周期 | 不具备当前五层package/slot模型 | task-owned pristine source + context working slot | 当前面向大图/replay |

## 17.3 从 pto2 继承的有效经验

以下不是被否定，而是被吸收到更严格边界中：

1. **外部传入 stream。** L1 不能自己决定用户算子排在哪条流。
2. **AICPU `WithHostArgs`。** 让 runtime持有每次task参数快照，比共享Host buffer安全。
3. **AICore handle launch。** prepare-time完成binary注册，launch-time只用稳定handle。
4. **固定、POD化的device task args。** 避免launch路径分配和Python对象。
5. **event表达内部多stream依赖。** 普通runtime task本身可以被ACLGraph捕获。
6. **workspace可作为显式执行资源建模。** 本次暂不外置，但未来并发slot化会重新采用这一思想。

## 17.4 明确不采用的做法

1. **不查询 capture状态。** eager/capture不得有两套launch语义。
2. **不获取或依赖 capture model。** PyPTO不知道自己属于哪张graph。
3. **不调用 `rtStreamAddToModel`。** 私有stream是否入图由event依赖和CANN正常capture决定。
4. **不让AICPU越过单算子边界提前启动。** 性能收益不能以破坏算子顺序为代价。
5. **不在普通launch中sync。** warmup/capture/close的quiescence全部由caller负责。
6. **不把workspace外置当作并发支持。** 其他可变状态未slot化前，外置workspace没有完整语义。

## 17.5 对 early orchestration 性能问题的最终回答

pto2 的优化动机是合理的：让AICPU尽早展开图，减少AICore等待。问题在于优化位置不对。单算子API只能观察自己的边界，不能提前消费stream后面的调度机会。

当前长期方案是 HBG：Host在本次算子调用/capture期间构建完整调度图，把graph作为该task的tiling-like参数交给CANN；device每次从pristine package恢复后执行。这样同样减少运行期动态编排成本，但所有工作仍属于当前L1节点，不跨越前后torch op。

因此两种方案的本质区别不是“要不要性能”，而是：

```text
pto2 early mode:
    借助capture/model知识，把orchestrator移到算子边界之前

current HBG L1:
    借助完整Host graph，把准备结果固化在算子自己的task参数内
```

后者更符合可组合的PyTorch自定义算子语义。

## 17.6 workspace差异为何不继续深究

pto2外部传workspace，本次内部持有workspace。这不是HBG/ACLGraph正确性的分水岭。当前L1不允许并发，context workspace地址在prepare后固定，并由graph/context生命周期保活，已经满足capture/replay。

只有当产品要开放并发、外部memory planning或torch allocator统一预算时，才值得把workspace变成显式API；届时必须与execution slot、capacity query、alignment和graph owner一起设计，不应只复制pto2参数列表。

---

## 18. 错误模型与可观测性

## 18.1 错误分类

| 类别 | 例子 | 状态影响 | 是否可继续launch |
| --- | --- | --- | --- |
| 纯Host参数错误 | shape/dtype/count/stride不匹配 | 不改变状态 | 修正参数后可以 |
| prepare输入错误 | duplicate callable、坏blob、unsupported HBG requirement | 尽量不改变；若已建资源则保留cleanup owner | 取决于phase，通常close |
| enqueue前native校验错误 | 错device pointer、错误stream切换 | 不提交metadata | 修正后可以 |
| 部分enqueue错误 | AICPU launch失败、event API失败 | Poisoned并执行异步cancel/join | 不可以，只能close |
| device orchestration错误 | scheduler/restore/dispatch/shutdown失败 | task返回错误，context视为不可继续 | 不可以，只能close |
| close错误 | unload/free/destroy失败 | 保持Closing与owner | 只允许retry close |
| 硬件失联 | core完全不report | 依赖外部timeout/recovery | context不再可用 |

## 18.2 HBG 16阶段故障注入

HBG package可携带仅测试使用、受plan hash保护的task-local fault marker。阶段包括：

1. restore copy；
2. restore publish；
3. after scheduler init；
4. before classify；
5. before dispatch；
6. shutdown；
7. runtime destroy；
8. scheduler init；
9. scheduler assign；
10. scheduler dispatch；
11. platform bridge；
12. affinity inputs；
13. KernelArgs Runtime；
14. physical core mapping；
15. physical core id；
16. slot fallback control。

marker属于单个 HostArgs/task/node，不放在resident global，避免一次注入泄漏到下一次正常调用。验证要求每个阶段：caller tail可达、hidden AICore退出、不做reset、紧邻的正常调用恢复、随后仍可capture/replay。

真实 teardown错误用独立unexpected-error汇合，优先于“受控fault可返回成功/预期错误”的测试逻辑，避免测试注入掩盖真正shutdown失败。

## 18.3 日志和状态的边界

设备返回 `507018` 等错误不能仅凭code推断“AICPU kernel主动返回非零”；CANN watchdog也可能把纯握手stall表成相同类别。定位必须结合：

- caller tail是否可达；
- AICore report收集进度；
- AICPU init/restore/dispatch阶段日志；
- event/API trace；
- 紧邻下一次调用是否恢复；
- 是否发生CANN禁止API。

---

## 19. 验证体系与当前证据

## 19.1 无硬件验证

当前测试分四层：

1. Python L1 wrapper：init owner、BaseException、scalar packing、layout绑定、adapter ABI、direct/queue路径；
2. Simpler Python：ChipWorker mode、queue capsule、close retry、host runtime ABI；
3. C++ contract：execution state、launch sequence、tensor validation、task ABI、HBG blob/slot/registry、handshake cancel；
4. runtime/codegen全量回归：TRB/HBG生成、artifact runtime cache key、L2/L3 wire兼容。

最终迁移后的 PyPTO runtime/codegen无硬件套件记录为 `1555 passed`，pre-commit hooks全通过。测试数字用于说明当时快照，不应替代后续提交的CI结果。

## 19.2 A2/A3 实机覆盖

已记录的 L1 能力包括：

- TRB eager、prepare/warmup、独立capture stream、ACLGraph多次replay；
- HBG eager与ACLGraph；
- torch predecessor/L1/torch successor顺序；
- 同context多callable；
- 同一graph add再mul；
- 同一callable多node；
- 两张graph串行使用；
- scalar、多输出、workspace；
- FP16、非uniform输入；
- matmul+bias多child；
- ReLU与SiLU；
- 8次replay与stream boundary stress；
- CANN禁止API trace，确认无sync/reset/capture query/model attach；
- HBG 16阶段no-reset fault injection；
- fault后同context正常调用与capture/replay；
- legacy L2测试后再次执行L1。

## 19.3 HBG HostArgs/graph owner专项

独立probe覆盖：

- 64 KiB、1 MiB、16 MiB、64 MiB HostArgs；
- full checksum与tail完整性；
- 64 MiB captured graph；
- 2048次task压力；
- 双graph各100次交替replay；
- graph destroy后64 MiB captured args对应内存回收。

这些结果证明本机当前CANN路径能持有和replay大参数，但：

- 64 MiB不是PyPTO/CANN公开上限；
- 2048不是launch规格；
- generic payload通过不等于任何未来production HBG最大图都通过；
- 新平台/新CANN版本仍需重新probe真实图大小。

## 19.4 L2/L3回归

- A2/A3 L3记录为7项通过；
- A2/A3 L2收集65个节点：63通过、1 skip、1个高性能`s8192` case可独立复现stall；
- 该case没有证据表明由L1改动引入，不能为了文档整齐写成65/65；
- L2 reset/运行后L1最终recheck通过，证明borrowed state未被legacy路径永久污染。

## 19.5 当前完成判据

就本次A2/A3范围，完成判据为：

- default taskQueue路径可作为普通torch op排队；
- TRB/HBG都能prepare、warmup、capture、连续replay并验数；
- launch不含禁止API；
- 多callable的local func id不冲突；
- HBG每次replay完整restore；
- graph package、working slot和registry ownership闭合；
- init/partial launch/close失败不丢owner、不UAF；
- no-reset错误路径能join已提交AICore；
- L2/L3 ABI未被静默破坏。

当前实现与上述实机/Host证据满足该判据。A5不属于本次验收结论。

---

## 20. 安全使用范式

## 20.1 单算子 eager

```python
ctx = pypto_init(programs=[compiled], device=device)
op = ctx.operator(compiled)

try:
    op.prepare()
    op.warmup(x, out=y)
    torch.npu.synchronize()

    op(x2, out=y2)
    torch.npu.synchronize()
finally:
    # 必须确认没有未完成的task或graph owner。
    ctx.close()
```

## 20.2 ACLGraph

```python
ctx = pypto_init(programs=[compiled], device=device)
op = ctx.operator(compiled)
graph = None

try:
    ctx.prepare()
    op.warmup(x, out=y)
    torch.npu.synchronize()

    # 使用torch_npu当前公开ACLGraph API在独立stream capture：
    # torch predecessor -> op(x, out=y) -> torch successor
    # 然后多次replay并外部sync验数。
finally:
    torch.npu.synchronize()
    if graph is not None:
        graph.reset()  # 或等价destroy
    ctx.close()
```

若 `pypto_init` 抛出 `L1InitializationError` 且携带 `cleanup_context`，caller仍需在安全条件下对该cleanup owner调用close/retry。不能只丢弃异常对象。

## 20.3 明确禁止的使用

- 两个线程共享一个context同时调用；
- 两条stream并发调用同一context；
- 两张ACLGraph并发replay共享一个context；
- graph未destroy就close；
- 依赖析构自动close；
- capture中首次lazy prepare；
- 修改capture节点所引用tensor的storage owner后继续replay；
- 把direct模式当taskQueue生产集成；
- 在HBG host builder里读取外部device tensor数值；
- 把2048次launch或64 MiB写成runtime保证。

---

## 21. 关键源码索引

## 21.1 PyPTO

| 主题 | 文件 |
| --- | --- |
| L1 Python API | `python/pypto/runtime/l1.py` |
| torch_npu taskQueue adapter | `python/bindings/torch_npu_l1_adapter.cpp` |
| compilation/runtime selection | `python/pypto/jit/decorator.py`, `python/pypto/ir/compile.py` |
| program/scalar metadata | `python/pypto/ir/compiled_program.py` |
| orchestration metadata/codegen | `src/codegen/orchestration/orchestration_codegen.cpp` |
| A2/A3 L1 ST | `tests/st/runtime/l1/` |
| 禁止API trace | `tests/st/runtime/l1/support/l1_cann_api_trace.cpp` |

## 21.2 Simpler

| 主题 | 文件 |
| --- | --- |
| L1 C ABI | `runtime/src/common/worker/pto_runtime_c_api.h` |
| ChipWorker/queue lease | `runtime/src/common/worker/chip_worker.cpp`, `l1_queue_call.h` |
| execution phase | `runtime/src/common/platform/onboard/host/l1_execution_state.{h,cpp}` |
| prepare/launch/close | `runtime/src/common/platform/onboard/host/device_runner_base.cpp` |
| 固定launch骨架 | `runtime/src/common/platform/onboard/host/l1_launch_sequence.h` |
| device state helper | `runtime/src/common/platform/onboard/host/device_runner_helpers.cpp` |
| TRB invocation ABI | `runtime/src/common/task_interface/l1_aicpu_args.h` |
| report/cancel协议 | `runtime/src/common/task_interface/aicore_handshake_protocol.h` |
| HBG launch blob | `runtime/src/common/task_interface/hbg_launch_blob.h` |
| HBG immutable plan | `runtime/src/common/worker/hbg_graph_plan.h` |
| HBG blob builder | `runtime/src/common/worker/hbg_launch_blob_builder.h` |
| HBG restore | `runtime/src/common/task_interface/hbg_restore.h` |
| HBG ContextRegistry | `runtime/src/common/task_interface/hbg_context_registry.h` |
| HBG execution slot | `runtime/src/common/task_interface/hbg_execution_slot*.h` |
| HBG callable registry | `runtime/src/common/task_interface/hbg_callable_registry.h` |
| HBG requirements | `runtime/src/common/task_interface/orchestration_requirements.h` |
| HBG fault protocol | `runtime/src/common/task_interface/hbg_l1_fault_injection.h` |
| A2/A3 AICPU/AICore实现 | `runtime/src/a2a3/` |

## 21.3 pto2 历史参考

| 主题 | 文件 |
| --- | --- |
| 私有stream拓扑 | `/mnt/workspace/inductor/pto2/pypto/framework/src/machine/runtime/context/stream_context.h` |
| capture/model attach与launch | `/mnt/workspace/inductor/pto2/pypto/framework/src/machine/runtime/launcher/device_launcher.cpp` |

---

## 22. 风险、限制与后续改进

## 22.1 最大剩余架构风险：无并发却有复杂共享状态

当前context只有一个workspace、working slot、Runtime、KernelArgs和hidden stream，但为了支持TRB/HBG、多callable、taskQueue和错误恢复，已经引入较多状态机。无并发假设一旦被调用方意外打破，Host mutex无法阻止不同graph replay在device上重叠。

改进方向不是再加一把Host锁，而是显式 `ExecutionSlot`：

- 每slot独立workspace/Runtime/KernelArgs/handshake/report/HBG working image；
- caller申请slot并获得完成fence；
- 只有fence证明完成后才回收到pool；
- captured node持有slot lease，graph destroy归还；
- pool大小是公开capacity，不依赖runtime内部launch计数。

## 22.2 HBG大HostArgs依赖CANN实现能力

当前WithHostArgs inline package已经通过大尺寸probe，但公开API边界、不同CANN版本和内存压力下的行为仍可能变化。改进选项：

1. 保留当前inline payload作为首选，启动时按真实production image做capability probe；
2. 若CANN未来提供正式task-owned external payload retain/release API，迁移到显式owner；
3. 若package超过可接受范围，拆成immutable GraphPlan allocation + 小HostArgs descriptor，但必须获得graph node级retain/release，不能用Host返回或event猜回收；
4. 对失败size返回明确capacity错误，不silent truncate。

## 22.3 HBG capture参数静态

当前argument snapshot hash把tensor地址/scalar绑定进package，符合静态ACLGraph。未来动态shape/地址更新需要：

- 定义哪些字段是graph-updateable；
- 重新计算identity/hash；
- 对受影响region重新host build/relocation；
- 保证node source与working slot generation原子切换；
- 与torch_npu graph update API建立owner协议。

在此之前，不应把“编译器支持动态shape”宣传成“同一captured HBG node可任意改shape”。

## 22.4 external/custom storage

当前default allocator路径完整，external storage依赖caller保活。后续可考虑：

- 检测storage是否由NPUCachingAllocator管理，非默认storage fail-fast；
- 接受显式lifetime token；
- 或接入PyTorch dispatcher schema，让框架拥有更强alias/lifetime信息。

在没有等价机制前，不要在无GIL callback里捕获Python owner。

## 22.5 跨Host进程resident DSO

v1只保证单Host进程、单live context。ContextRegistry消除了同进程第二context的resident-global冲突，但Host生成的generation不能证明跨进程顺序复用同一resident DSO时全局单调。

若未来支持跨进程接管，需要：

- device/driver分配的epoch或resident DSO原子nonce；
- 显式claim/release协议；
- 崩溃进程owner回收；
- 不依赖`CLOCK_MONOTONIC + process-local atomic`作为全局唯一性证明。

## 22.6 A5与新平台

当前代码保留A5对称实现和大量无硬件编译检查，但用户已明确本次只验收A2/A3。新平台转正必须重新跑：

- task args alignment；
- event capture行为；
- AICPU/AICore cache publish/invalidate；
- per-core report；
- HostArgs尺寸；
- HBG replay/fault matrix；
- L2/L3回归。

不能把A2/A3实机结论直接外推。

## 22.7 性能优化原则

允许的优化：

- HBG减少device动态构图；
- prepare预注册binary/handle；
- 缩小TRB invocation HostArgs；
- 缓存Host immutable plan；
- 在不改变owner语义下减少cache维护；
- 未来slot化并发。

不允许的优化：

- 让AICPU提前于caller stream算子位置执行；
- capture时跳过必要依赖；
- 私自attach stream到graph/model；
- 用内部sync掩盖lifetime问题；
- 按固定launch次数猜task完成；
- 复用仍可能被graph replay引用的package。

---

## 23. 维护者评审清单

修改 L1 launch、task ABI、HBG 或 torch adapter 时，至少逐项回答：

### API与边界

- caller stream是否仍显式传到native？
- taskQueue是否仍使用`.stream(false)`？
- callback是否完全不依赖Python/GIL？
- 是否新增了capture query、model handle或stream attach？
- eager与capture是否仍是同一launch路径？

### 分配与同步

- launch是否新增device alloc/free？
- 是否新增lazy binary register或H2D staging？
- 是否出现stream/device sync/reset？
- 新Host内存是否有task/callback生命周期owner？

### 地址与ABI

- 新字段是否版本化并做size/alignment/static_assert？
- 是否破坏ChipCallable/L3 legacy offset？
- device pointer是否只有一个trust source？
- CCE address-space类型是否在A2/A3真实target编译？
- 变长blob是否做overflow、overlap和精确长度检查？

### 时序与错误

- AICPU/AICore任一分支enqueue失败时，另一分支如何退出？
- caller尾部是否总能join hidden stream？
- 所有participant是否exactly-once arrive/depart？
- teardown错误是否可能被合成fault覆盖？
- close失败是否保留owner并拒绝新dispatch？

### HBG生命周期

- canonical plan是否仍不可变？
- 每个task/node是否有独立source snapshot？
- replay前是否完整restore full SM/arena？
- package是否绑定正确slot/callable/generation？
- registry是否仍由context拥有，而非resident global？
- graph destroy前是否可能释放package目标或context资源？

### 回归

- Python L1反例、C++ no-hardware contract是否通过？
- A2/A3 onboard TRB/HBG eager与ACLGraph是否验数？
- 多callable与同callable多node是否覆盖？
- 禁止API trace是否仍为零？
- L2/L3 ABI与代表性回归是否覆盖？

---

## 24. 原始计划到最终实现的追踪

| 原始问题/决策 | 最终实现 |
| --- | --- |
| A2/A3 与 A5 是否分叉 | ABI尽量共用；本次只以A2/A3上板验收，A5不作完成门槛 |
| 第一runtime范围 | TRB先完成，随后HBG进入正式支持 |
| simulator能否证明stream/capture | 不能；sim只做Host/device contract，ACLGraph必须onboard |
| program范围 | `@pl.program` |
| 动态shape | 编译内部不设额外障碍；v1 captured invocation仍是静态shape/layout/args快照 |
| tensor/scalar地址与值 | 每次task快照；capture后由ACLGraph owner决定是否更新，PyPTO不感知 |
| 并发 | v1明确禁止；同一context共享workspace/working state |
| L2/L3影响 | borrowed mode、ABI padding和按需资源隔离，保留原路径 |
| 底层stream参数 | C ABI强制传入；Python wrapper默认从taskQueue callback取得raw stream |
| taskQueue放置位置 | 独立torch_npu adapter，不污染Simpler core |
| prepare入口 | `pypto_init`声明全部program，`context.prepare()`整体准备 |
| AICPU stream | 直接使用caller stream |
| AICore stream | 内部hidden stream，不对外暴露 |
| warmup | ACLGraph前必须显式warmup并由caller外部sync |
| stream切换 | 同流FIFO；换流前旧tail必须完成，不向capture导入图外event wait |
| workspace | 当前context内部固定；不因pto2外置方案扩大v1 API |
| binary内存 | context lifetime累计，prepare注册；launch不lazy register |
| task args复用 | CANN WithHostArgs task/node snapshot，不按launch次数猜回收 |
| `aclrtLaunchKernelWithHostArgs` | TRB固定args与HBG变长inline package都采用 |
| reset API | 不提供；错误后Poisoned，外部quiescence后close |
| capture感知 | 不查询、不attach model、不区分eager/capture路径 |
| pto2 early AICPU | 明确拒绝跨单算子边界；性能路径由HBG承担 |
| HBG graph source | immutable GraphPlan -> fresh scratch -> CANN-owned snapshot |
| HBG执行状态 | context-owned mutable slot，每次eager/replay完整restore |
| HBG registry | 从resident global迁为Context-owned registry |
| graph生命周期 | caller持有graph/tensor/context；graph destroy后才close |
| kernel launch内部约2048规格 | 不依赖；只作为压力测试采样点 |
| device id | Python显式传入，底层校验current device，不替caller切换 |

这一追踪表用于说明：原计划中的开放问题已经落到具体代码契约；若未来修改表中任一结论，必须同步更新API、错误模型、测试和本文，而不能只改某个launch helper。

---

## 25. 最终结论

本次改造不是给原L2 runner加一个“传stream”的快捷入口，而是建立了一套新的borrowed-device执行契约：

- 对外是一个普通、异步、带caller stream的PyTorch算子；
- 对内用event包住AICPU与AICore两个分支；
- taskQueue、tensor allocator与native owner形成完整生命周期链；
- prepare固定所有graph可见资源，launch不分配、不同步、不感知capture；
- TRB用task-owned invocation snapshot解决连续异步调用；
- HBG把动态build graph建模为tiling-like immutable task参数，并在每次replay恢复context-owned working slot；
- ContextRegistry消除resident global跨context残留；
- Host/device cancel与completion gate让错误路径不依赖reset；
- L2/L3继续保有自己的owned-device语义和wire兼容。

与pto2相比，当前方案保留了外部stream、WithHostArgs和预注册handle这些正确基础，但明确放弃capture查询、model attach、私有AICPU stream提前执行和内部sync。性能优化由HBG在单算子内部完成，而不是跨越单算子边界。

在当前A2/A3、单Host进程、单live context、无并发、静态capture参数的v1范围内，这套设计已经形成从API、执行时序、参数快照、graph生命周期、错误闭包到实机验收的完整闭环。
