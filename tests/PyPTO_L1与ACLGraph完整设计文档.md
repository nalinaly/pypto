<!-- markdownlint-disable MD013 MD024 MD036 MD048 MD060 -->

# PyPTO L1 单算子与 ACLGraph 完整设计文档

## 0. 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档性质 | 当前实现的最终态设计说明、接口契约与维护指南 |
| 更新时间 | 2026-08-19 |
| PyPTO 基线 | `665a2c9667d9b349b6f52236f8716b5ee7acaa1e`，`main` |
| Simpler 基线 | `e58d54c0519dba4865ac5e026934aa906b4fee9b`，`main` |
| 当前验收平台 | A2/A3 onboard |
| 当前 L1 runtime | `tensormap_and_ringbuffer`（TRB）与 `host_build_graph`（HBG） |
| 当前公开入口 | `@pl.jit(execution="l1", runtime=...)` 像 Triton kernel 一样直接调用 |
| 底层调试入口 | `pypto_init/L1Context/L1Operator`，仅作 advanced control-plane API |
| 关联文档 | [实现过程记录](./PyPTO_L1与ACLGraph实现过程记录.md)、[原始实现计划](./pypto_l1_aclgraph_implementation_plan.md) |

本文是本次工作的唯一完整设计文档，已合并原《PyPTO Triton风格L1 JIT调用接口设计》的全部内容。前半部以native执行协议为主，后半部完整保留公开JIT API、产品层对象模型、实现顺序和验收证据。它不是阶段性计划，也不是测试流水账。出现信息冲突时，采用以下优先级：

1. 当前 PyPTO 与 Simpler 源码；
2. 已记录的 A2/A3 实机证据；
3. 本文明确写出的最终契约；
4. 实现过程记录中的历史判断；
5. 原始实现计划中的预案。

原始实现计划保留了大量设计推演和风险分析，仍然值得阅读；实现过程记录保留了故障、反例和上板证据。本文将二者收敛成一套可直接用于维护、评审和后续演进的最终模型。

### 0.1 相对早期计划的关键修正

当前实现已经超出早期 Phase 1 的最小范围，以下结论必须以当前实现为准：

1. HBG 已具有正式的 L1 调用、ACLGraph capture/replay 和错误恢复路径，不再是“暂不支持”的占位能力。
2. 正常调用最终采用 AICore-first 的 Host enqueue 顺序，同时以共同 Start event 保持单算子边界，并为 AICPU enqueue 失败补充 Host cancel 与 join 闭包；不再采用早期文档中的简单 AICPU-first 描述。
3. HBG 的 mutable execution-slot registry 已由 resident AICPU DSO global 状态迁移为 Context-owned device registry。resident DSO 仅临时 latch 当前 context registry 地址，不再拥有跨 context 的 registry 内容或 generation 状态。
4. 普通用户不再手工组装 `pypto_init -> operator -> prepare -> warmup -> close`；公开入口是 `@pl.jit(execution="l1", runtime=...)`，首次普通 eager 调用自动完成 specialization、init、prepare 和 warmup。
5. HBG callable identity 与 function binding 完整自包含在每个 CANN-owned launch package 中；ContextRegistry 只保留可变 execution-slot trust root，不再拥有固定 HBG callable table。
6. TRB L1 code registry 动态 append 并按进程 pin，不再暴露 64-callable 上限；旧 64 槽只保留给 L2/L3。
7. 新增 L1 路径永不调用 `aclrtBinaryUnLoad` 或 `rtsBinaryUnload`。可选 `pypto.l1.shutdown(device=...)` 只退役 context-owned 可安全释放的资源，binary 和 graph-visible code handle 保持 process pinned。

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
4. 支持 `@pl.jit(execution="l1")` 生成的 TRB 与 HBG 编译产物；底层仍以 compiled program/callable 为运行单元。
5. 公开 API 以首次 ordinary eager 隐式完成 prepare/warmup，随后由 caller external sync 再 capture/replay；advanced API 仍可显式驱动同一流程。
6. 支持同一 hidden device owner 按 specialization 动态 append 多个 callable，且每个 callable 有独立 `func_id` 命名空间。
7. 保持 L2/L3 的 API、资源模型和 wire ABI 兼容。
8. init/prepare/launch/close 的失败都必须保持可诊断、可拒绝后续调用，并在可行时支持显式重试清理。

## 2.2 当前非目标

1. 不允许同一设备上多个 live L1 context。
2. 不允许 L1 调用并发执行；PyPTO 仍占用全部 AICore。
3. 不允许同一 context 内跨 stream 未 quiesce 切换。
4. 不提供运行期动态扩容 workspace、HBG working slot 或 package capacity。TRB L1 code registry 例外：它只允许在 capture 外 append 新的 immutable callable、不回收旧 entry。
5. 不提供 capture 后修改 tensor 地址、scalar 或 HBG 拓扑的 graph update API。
6. 不提供 L1 内部 stream/device synchronize。
7. 不承诺外部 `from_blob`/自定义 allocator storage 在调用方提前销毁时仍安全。
8. 不把完全未 report 的硬件 core 失联恢复纳入算子内协议；该类故障交给 CANN op timeout、driver fault containment 或外部 device/context recovery。
9. 当前验收只以 A2/A3 为准；A5 与 A5 simulator 不作为本次完成条件。

## 2.3 核心硬不变量

1. **外部 stream 是唯一入口顺序源。** L1 不切换 current device，也不替 caller 选择 stream。
2. **launch 路径不分配或释放 device memory。** 所有 device 资源在 prepare 阶段固定。
3. **launch 路径不做 stream/device sync，不做 reset。**
4. **不查询 capture 状态，不取得 graph/model handle，不调用 `rtStreamAddToModel`。**
5. **AICPU task 位于 caller stream。** hidden stream 只承载 AICore 分支。
6. **两个分支必须在单算子边界内 fork/join。** caller 尾部必须等待 hidden AICore 完成。
7. **每次 Host 调用都有不可变参数快照。** 下一次 Host 调用不能覆盖尚未消费的 task args。
8. **图可见地址在 capture 前固定。** capture 期间禁止当前 specialization 的 lazy register、H2D staging、arena 增长和 registry 变更；新 specialization 只能在图外的后续 ordinary eager 调用中 append。
9. **shutdown/advanced close 不猜测设备是否空闲。** caller 必须先保证所有 eager/captured work quiescent 并销毁/reset相关 graph；不调用 shutdown 时资源安静地 pin 到进程结束。
10. **L1 与 L2/L3 模式互斥。** 借用设备的 L1 context 不能触发 L2 的 device reset/aclFinalize 路径。
11. **L1 binary 与 graph-visible code handle 按进程 pin。** 新路径在 init rollback、prepare rollback、shutdown 与析构中都不调用 BinaryUnLoad。

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
        | @pl.jit(execution="l1", runtime=...) ordinary call
        v
pypto.runtime.l1_jit
  hidden per-device owner ---- specialization registry
        |
        | advanced/debug shares pypto.runtime.l1 control plane
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
| `pypto.runtime.l1_jit` | `@pl.jit(execution="l1")` 公开调用、hidden owner、specialization late append、eager输出分配与capture前置检查 | 暴露context/operator ceremony、capture探测、内部sync |
| `pypto.runtime.l1` | advanced/debug控制面，以及公共的编译产物、shape/dtype/layout/scalar校验 | 作为普通用户的首选入口、替caller管理graph生命周期 |
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
import pypto
import pypto.language as pl


@pl.jit(execution="l1", runtime="tensormap_and_ringbuffer")
def add(
    x: pl.Tensor[[64, 128], pl.FP32],
    y: pl.Tensor[[64, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
):
    ...


# ordinary eager：首次调用隐式 specialize/init/prepare/warmup，
# pure Out 可由 wrapper 通过 torch allocator 创建。
z = add(x, y)
torch.npu.synchronize()

# ACLGraph capture 前预分配输出；capture 内仍是相同的普通调用。
add(x, y, out=z)

# 完全可选：调用前由用户证明所有 task/graph 已终止引用。
pypto.l1.shutdown(device=0)
```

API 约束：

1. decorator 固定 `execution="l1"` 与 TRB/HBG runtime；同一 device owner 不允许混用 runtime。
2. device 从 NPU tensor 推导，并必须等于 torch_npu 当前设备；初始化不会替用户切设备。
3. 每个新 specialization 必须先在 capture 外做一次 ordinary eager 调用并由 caller 外部同步；capture 中首次 lazy init/prepare 明确报错。
4. ordinary eager 省略全部 pure Out 时，wrapper 使用 torch allocator 分配并返回 tensor/tuple；capture 要求事前分配并显式传入全部输出。
5. hidden owner 线程亲和；能检测到的 Host 并发直接 fail-fast，不宣称因此已防住 graph replay 并发。
6. 后续出现的 specialization 可以在前一 callable 已 warm 后动态 append，ID 单调增长且不复用。
7. GC/atexit 不调用 runtime close；不显式 shutdown 时安静 pin 到进程结束。
8. `pypto_init/L1Context/L1Operator` 保留给 bring-up、故障注入和底层审计，不是普通用户样板。

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
- prepare 上传 callable-local 静态状态并固定本 specialization 的图可见地址；TRB L1 允许后续在 capture 外 append 新 callable；
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
New -> Initializing -> Collecting <-> ReadyEnqueued
          |               |              |
          +---------------+--------------+
                                  failure -> Poisoned

Collecting/ReadyEnqueued/Poisoned
                  | begin_close
                  v
               Closing --retry--> Closing --success--> Closed
```

关键语义：

- `Collecting`：初始 callable 尚未全部排入 prepare；
- `ReadyEnqueued`：已知 callable 的 prepare task 已排入 caller stream；成功 launch 不会关闭 append admission。
- `Sealed`仅作为旧 ABI/历史状态的兼容输入；当前 `seal()` 归一化回 `ReadyEnqueued`，不表示固定 callable capacity。
- `Poisoned`：执行期错误后拒绝 prepare/launch，但仍允许 close；
- `Closing`：第一项 destructive teardown 前即粘性进入，任何 prepare/launch 都 fail-closed；close 失败可重试；
- `Closed`：资源释放完成。

析构函数、Python GC与`atexit`都不调用 ACL/runtime close API。用户不调用可选`shutdown()`时，资源安静地保留到进程结束；实现至多输出低级别诊断日志，不能制造`Core`类报错，也不能在未知graph/stream状态下隐式销毁。

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
| AICPU binary/function handle | process-pinned L1 code owner | init | 是 | 进程结束；L1 shutdown 不调用 BinaryUnLoad |
| AICore binary/function handle | process-pinned L1 code owner | prepare | 是 | 进程结束；不 unregister/复用 |
| Runtime/KernelArgs | L1 context | prepare | 是 | close |
| HBG callable-local function table | CANN-owned self-contained package | 每次 launch | 是 | task/graph owner 管理 package，code 仍 process pin |
| TRB append-only code entry | resident AICPU registry | 新 callable prepare | 是 | 不回收；进程结束 |
| workspace/arena/register windows | L1 context | prepare | 是 | close |
| `L1AicoreReport[]` | L1 context | prepare | 是 | close |
| queue call snapshot | taskQueue entry | 每次 Host 调用 | Host 队列可见 | callback 完成后释放 |
| CANN HostArgs copy | CANN task/graph node | enqueue/capture | 是 | task 或 graph owner 管理 |
| HBG host GraphPlan | L1 callable-local单条cache | 首次调用或参数语义变化 | 否 | 新identity事务替换旧entry；context close时释放 |
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

## 7.3 prepare 与 late append

早期实现会在第一次 launch 后 seal context，因此要求 `pypto_init(programs=[...])` 一次声明所有 program。这不符合 JIT specialization 按需出现的模型，也人为暴露了固定 callable capacity。

当前协议改为：

1. 首个 specialization 的 ordinary eager 调用建立 hidden owner 并只 prepare 该 callable。
2. 后续新 specialization 可在 capture 外调用中获得单调不复用的 context-local ID，完成 Host 验证后 append。
3. HBG 新 callable 的 identity/function table 只出现在它自己的 per-task package 中，不扩容 resident table。
4. TRB 新 callable 在 AICPU 端发布一个 append-only node；旧 node、code mapping 与 graph-visible handle 不覆盖、不回收。
5. 任一 specialization 的首次 prepare 都不能发生在 capture 内；这不意味进程后续不能在图外发现新 specialization。

advanced `L1Context.prepare()` 仍会按确定顺序准备当前已声明的 callable，但不再把 context 或 capacity 冻结。

## 7.4 prepare 的职责

prepare 可以 enqueue 异步 H2D/registration task，但必须在 caller stream 上建立 PrepareTail 边界。主要工作包括：

- 精确解析 callable blob 长度，拒绝截断、越界与重复 identity；
- 加载/注册 AICPU 与 AICore binary，缓存可直接 launch 的 handle；
- 为每个 callable 建立独立 `func_id -> device function address` 快照；
- 创建 Runtime、KernelArgs、arena、register windows、workspace；
- 分配每核独占 cache line 的 `L1AicoreReport`；
- 初始化 HBG working slot 与 ContextRegistry；
- 上传并回读关键 device pointer/metadata 做完整性检查；
- 冻结 worker count、workspace/HBG working-slot capacity、已准备 callable 的地址和 ABI metadata；TRB registry 仍可在图外 append 新 immutable entry；
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

又因为当前没有公开契约保证“`BinaryUnLoad` 后 CANN 会替用户继续保活已被 captured graph 引用的 funcHandle/code”，新 L1 路径采用更强的 process-pin 规则：

- init rollback、prepare rollback、shutdown 和析构都不调用 `aclrtBinaryUnLoad`/`rtsBinaryUnload`；
- `FinalizeL1Pinned()` 只释放可证明不再被 task/graph 引用的 bootstrap buffer，然后丢弃 Host loader 的 unload ownership record；
- AICore registered handle、AICPU binary 和 TRB resident code 不 unregister、不复用，直到进程结束；
- legacy L2/L3 仍走自己的 `Finalize()` 与 unload/reset 语义，不被 L1 改写。

## 11.2 callable identity 与函数地址

`func_id` 是单个 compiled program 内的局部编号，不是 context-global id。每个 callable 持有自己的函数地址快照和 identity hash。HBG invocation 还携带 callable、argument snapshot 和 function binding hash，防止 package 与错误 slot/callable 配对。

prepare 后 identity 不可变，但 context 不因首次 launch 而 seal 新 callable admission。HBG package 自包含 callable identity/function binding；TRB 使用动态 append-only node。任何 duplicate identity conflict、同 id 不同 content hash/地址、缺失 handle 或 ABI 不一致都在发布或 enqueue 前拒绝。

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

普通 JIT 用户可以完全不调用 close/shutdown，hidden owner 会安静 pin 到进程结束。若用户选择调用 `pypto.l1.shutdown(device=...)`，或 advanced API 调用 `L1Context.close()`，caller 必须：

1. 停止新的 L1 enqueue；
2. 等待所有 eager task 完成；
3. 等待所有 graph replay 完成；
4. reset/destroy 持有 L1 node 的 ACLGraph；
5. 保持 context、binary、tensor storage 和 HBG source owner 到上述步骤完成。

shutdown/close 自身不做 synchronize，也不查询 graph owner。它重复调用幂等；失败时保留 owner 供显式重试。GC/atexit 不调用 runtime close。销毁某一张 graph 不会自动触发 device-wide shutdown，因为其他 graph/task 可能仍持有同一 owner。

## 12.2 fail-closed teardown

native close 在第一项 destructive teardown 前进入 `Closing`。随后：

- prepare/launch 全部拒绝；
- L1-pinned loader 释放可安全退役的 bootstrap buffer，然后对 device free、event/stream destroy 逐项记录错误；不表达 binary unload。
- 某项失败时保留其 handle/ownership table，不清空后伪装成功；
- hidden execution state 与 per-device claim 只在所有外层资源成功释放后关闭；
- `ChipWorker` 只有 native finalize 成功后才退役可安全关闭的 Host DSO owner；graph-visible CANN binary/function handle 继续 process pin；
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

L1在每个callable内保留一个context-owned cache entry。只有tensor完整descriptor/address和scalar bit pattern都与entry精确一致时才复用plan；hash只用于快速筛选，随后必须逐字段比较，不能把hash碰撞当作命中。参数变化时先完整构建候选plan，成功后才事务替换旧entry，因此cache有界且失败不破坏上一份可用plan。cache只延长canonical Host plan的寿命，不延长任何mutable execution state，也不替CANN持有某个captured node的参数。

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

只有所有相关 graph replay结束、外部sync且相关graph均已reset/destroy后，用户才可以调用可选的设备级`shutdown()`（advanced API中等价为close hidden context）。销毁一张graph本身绝不意味着该device上已经不存在其他graph，也不会自动触发shutdown。

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
```

DeviceRunner在prepare创建、初始化和注册它，并持有到可选shutdown/advanced close成功；若用户不调用shutdown，则随hidden owner pin到进程结束。它只保存context-owned mutable execution-slot trust root。callable identity、tensor/scalar snapshot、callable-local function binding与pristine graph全部在每个CANN-owned launch package中自包含，所以不再需要HBG resident callable table。AICPU resident DSO只保存当前registry device address，不能保存：

- slot内容；
- callable内容或固定大小的 callable slot；
- 上一个 context generation；
- 跨 context conflict状态。

这一设计取代了早期 resident-global registry。后者在第一个 context close后仍随 resident scheduler DSO存活，第二个 context可能遇到 stale/conflict；用 Host generation反复 reset resident global既难证明所有线程不再访问，也把资源 owner放错了层级。

ContextRegistry让生命周期重新对齐：谁创建context，谁拥有registry。native advanced close会在外部quiescence契约满足后释放它；公开产品层的成功shutdown则把owner退役并继续pin graph-visible binary，不承诺同一进程再次初始化。新进程中的首个context会得到全新的地址与内容。v1仍限定单Host进程、单live context；跨Host进程顺序复用resident DSO不在当前generation唯一性保证内。

## 15.6 HBG 每次 invocation/replay 的 device 协议

```text
1. AICPU entry取得当前 ContextRegistry地址
2. 校验registry header/context generation
3. 取得唯一ExecutionSlotRegistration
4. 校验HostArgs header/binding/identity/hash/region和package内自包含的callable-local function table
5. 所有participant汇合；唯一leader restore完整pristine regions
6. leader发布restore verdict（release）
7. peers acquire读取；失败共同进入error epilogue
8. scheduler init/assign最终裁决
9. orchestration classify/dispatch
10. shutdown/runtime destroy
11. arrive -> unique finalize -> snapshot result -> depart
12. last-depart cleanup本轮可变executor状态
```

replay时步骤完全相同。Host不会再次 build图，但 CANN重放 AICPU task，leader从 graph-node-owned source重新 restore工作区，所以第二次及后续 replay不会继承已消费 scheduler状态。

## 15.7 多 callable 与多 node

- context-global callable id只用于Host owner管理与package identity，AICPU不用它索引resident HBG callable table；
- 每个 node/package拥有自己的 callable identity、function table/hash和argument snapshot；
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

CANN task args容量通过真实 probe验证，不把约 2048 launch或源码内某个 256 MiB常量写入产品规格。每个 HBG package 由 `aclrtLaunchKernelWithHostArgs` 交给 CANN，task/captured node 各自持有不可变 snapshot；PyPTO 不依赖固定 HBG callable capacity，也不按 launch 计数回收 package。context working slot 仍固定容量并由调用时序保证不重叠。

---

## 16. ACLGraph 生命周期

## 16.1 标准流程

```text
first ordinary eager @pl.jit(execution="l1") call
        |
specialize/compile + hidden owner init + this callable prepare/launch
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
optional pypto.l1.shutdown(device=...)
```

capture 内的输出 tensor 必须在 capture 前分配并显式传入。后续出现的新 specialization 重复上述“图外 ordinary eager + external synchronize”，不需要在进程开始时批量列出全部 program。

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
- 任一相关graph仍存活时不得调用设备级shutdown/advanced close；
- 不支持两条 stream同时 replay两个 graph。

## 16.5 为什么没有 `reset()` 用户 API

正常 warmup到capture只需要 caller同步与 stream-switch gate，不需要 PyPTO reset内部状态。显式 reset API会诱导用户在 graph仍持有地址时清理slot/event，或在错误后尝试复用半拆状态。当前正确动作只有：正常继续调用，或停止调用、外部quiesce、销毁全部相关graph，再可选shutdown。Poisoned owner不能 reset回Ready；binary无论如何都不在L1路径unload。

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
| shutdown/close错误 | pinned-loader bootstrap资源释放、device free或stream/event destroy失败 | 保持Closing与owner | 只允许retry shutdown/advanced close；绝不BinaryUnLoad |
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

最终迁移阶段的 PyPTO runtime/codegen无硬件套件曾记录为 `1555 passed`。当前 JIT 产品层收口快照另外通过：JIT decorator/compile extraction/L1 facade `169 passed`，L1 Python/taskQueue/lifecycle/source guard `62 passed`，Simpler C++ non-hardware `120/120 passed`，两仓 pre-commit 全通过。测试数字用于说明对应提交快照，不应替代后续CI结果。

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
- HBG self-contained graph package、CANN task/node snapshot与context-owned working-slot ownership闭合；
- init/partial launch/close失败不丢owner、不UAF；
- no-reset错误路径能join已提交AICore；
- L2/L3 ABI未被静默破坏。

当前实现与上述实机/Host证据满足该判据。A5不属于本次验收结论。

---

## 20. 安全使用范式

## 20.1 单算子 eager

```python
@pl.jit(execution="l1", runtime="tensormap_and_ringbuffer")
def kernel(
    x: pl.Tensor[[64, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
):
    ...


# 首次 ordinary eager 自动初始化/准备，并由 torch allocator 创建 pure Out。
y = kernel(x)

# 后续 eager 调用可继续省略输出，也可显式复用已分配 tensor。
kernel(x2, out=y2)
```

eager 本身不需要为 PyPTO 加 `try/finally`。只有调用方需要观察结果或退役全局 owner 时才做外部 synchronize。

## 20.2 ACLGraph

```python
graph = None

# 图外普通调用：完成 specialization/init/prepare/真实 warmup。
warmup_out = kernel(x)
torch.npu.synchronize()

# capture 前分配图内输出，并持有到 graph 销毁。
captured_out = torch.empty_like(x)

try:
    graph = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(graph, stream=capture_stream):
        kernel(x, out=captured_out)

    graph.replay()
finally:
    torch.npu.synchronize()
    if graph is not None:
        graph.reset()

# 完全可选；不调用则安静pin到进程结束。
pypto.l1.shutdown(device=device)
```

底层 advanced API 若直接使用 `pypto_init`，仍要处理 `L1InitializationError.cleanup_context` 和显式 close/retry；这是故障注入/维护者契约，不应回流为普通用户样板。

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
| HBG callable registry历史兼容类型 | `runtime/src/common/task_interface/hbg_callable_registry.h`；当前L1 active path不读取resident callable table |
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
| program范围 | 公开入口为 `@pl.jit(execution="l1")`，底层沿用 compiled program/callable |
| 动态shape | 编译内部不设额外障碍；v1 captured invocation仍是静态shape/layout/args快照 |
| tensor/scalar地址与值 | 每次task快照；capture后由ACLGraph owner决定是否更新，PyPTO不感知 |
| 并发 | v1明确禁止；同一context共享workspace/working state |
| L2/L3影响 | borrowed mode、ABI padding和按需资源隔离，保留原路径 |
| 底层stream参数 | C ABI强制传入；Python wrapper默认从taskQueue callback取得raw stream |
| taskQueue放置位置 | 独立torch_npu adapter，不污染Simpler core |
| prepare入口 | 首次ordinary eager隐式prepare当前specialization；后续specialization可在图外late append |
| AICPU stream | 直接使用caller stream |
| AICore stream | 内部hidden stream，不对外暴露 |
| warmup | ACLGraph前必须显式warmup并由caller外部sync |
| stream切换 | 同流FIFO；换流前旧tail必须完成，不向capture导入图外event wait |
| workspace | 当前context内部固定；不因pto2外置方案扩大v1 API |
| binary内存 | process pin；prepare注册，launch不lazy register，L1任何路径不BinaryUnLoad |
| task args复用 | CANN WithHostArgs task/node snapshot，不按launch次数猜回收 |
| `aclrtLaunchKernelWithHostArgs` | TRB固定args与HBG变长inline package都采用 |
| reset API | 不提供；错误后Poisoned，外部quiescence后可选shutdown/advanced close |
| capture感知 | 不查询、不attach model、不区分eager/capture路径 |
| pto2 early AICPU | 明确拒绝跨单算子边界；性能路径由HBG承担 |
| HBG graph source | immutable GraphPlan -> fresh scratch -> CANN-owned snapshot |
| HBG执行状态 | context-owned mutable slot，每次eager/replay完整restore |
| HBG registry | mutable execution slot从resident global迁为Context-owned registry；callable信息进入self-contained package |
| TRB registry | L1动态append-only，无公开64-callable上限，旧entry/code不回收不复用 |
| graph生命周期 | caller持有graph/tensor；hidden owner强引用存活，graph destroy/quiesce后才可选shutdown |
| kernel launch内部约2048规格 | 不依赖；只作为压力测试采样点 |
| device id | Python显式传入，底层校验current device，不替caller切换 |

这一追踪表用于说明：原计划中的开放问题已经落到具体代码契约；若未来修改表中任一结论，必须同步更新API、错误模型、测试和本文，而不能只改某个launch helper。

---

## 25. 最终结论

本次改造不是给原L2 runner加一个“传stream”的快捷入口，而是建立了一套新的borrowed-device执行契约：

- 对外是 `@pl.jit(execution="l1", runtime=...)` 装饰后像Triton kernel一样直接调用的普通、异步PyTorch算子；
- 对内用event包住AICPU与AICore两个分支；
- taskQueue、tensor allocator与native owner形成完整生命周期链；
- 首次ordinary eager隐式完成init/prepare/warmup；capture前外部sync，capture内显式传入预分配输出；
- prepare固定已知specialization的graph可见资源，launch不做device alloc/free、不同步、不感知capture；
- TRB用task-owned invocation snapshot解决连续异步调用，code registry动态append并按进程pin；
- HBG把动态build graph建模为tiling-like immutable task参数，并在每次replay恢复context-owned working slot；
- HBG package自包含callable identity/function binding，ContextRegistry只持有mutable slot，消除resident callable global跨context残留；
- L1 binary/function handle按进程pin，可选shutdown不调用BinaryUnLoad；
- Host/device cancel与completion gate让错误路径不依赖reset；
- L2/L3继续保有自己的owned-device语义和wire兼容。

与pto2相比，当前方案保留了外部stream、WithHostArgs和预注册handle这些正确基础，但明确放弃capture查询、model attach、私有AICPU stream提前执行和内部sync。性能优化由HBG在单算子内部完成，而不是跨越单算子边界。

在当前A2/A3、单Host进程、单live context、无并发、静态capture参数的v1范围内，这套设计已经形成从API、执行时序、参数快照、graph生命周期、错误闭包到实机验收的完整闭环。

---

## 第二篇：Triton 风格公开 API 与产品层设计

> 本篇完整保留原独立接口设计中的公共 API、隐藏 owner、HBG/TRB 生命周期、实现顺序、测试矩阵、拒绝方案与最终落地证据。与前篇重复之处用于从产品层重新建立推导链，而不是另一套相互独立的规范；如措辞存在差异，以源码事实和本篇较新的产品层收口为准。
>
> 状态：已实现，A2/A3 TRB/HBG eager + ACLGraph 已上板验证
> 适用范围：A2/A3 onboard、PyTorch/torch_npu、TRB 与 HBG、eager 与 ACLGraph
> 合并说明：本篇原为独立的 Triton 风格接口设计，现已完整并入本文；历史推演与实测细节继续参见实现过程记录和原始实现计划。
> 核心目标：把已经跑通的 L1 底层能力包装成像 Triton JIT kernel 一样自然的 Python API，同时保留现有 taskQueue、失败所有权和 ACLGraph 生命周期边界。

---

### 1. 结论先行

普通用户最终只需要定义并调用 kernel：

~~~python
import torch
import pypto.language as pl


@pl.jit(execution="l1", runtime="tensormap_and_ringbuffer")
def add(
    x: pl.Tensor[[64, 128], pl.FP32],
    y: pl.Tensor[[64, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        # kernel body
        ...


# eager：允许省略 Out，由 PyTorch wrapper 使用 torch allocator 分配。
z = add(x, y)

# 显式输出：eager 和 ACLGraph capture 都支持。
add(x, y, out=z)
~~~

用户不再需要在正常路径里接触：

~~~python
ctx = pypto_init(...)
op = ctx.operator(...)
op.prepare()
op.warmup(...)
try:
    ...
finally:
    ctx.close()
~~~

L1 JIT 的公共语义是：

1. 首次普通 eager 调用自动 specialize、compile、初始化隐藏的 device runtime、注册 callable，并 enqueue 第一次执行。
2. 首次调用不能发生在 ACLGraph capture 内；用户必须先在图外以同一 specialization 做一次 eager warmup，并由 caller 外部同步。
3. 已 warm 的 specialization 在 capture 内只表现为一个普通 torch_npu 算子：使用当前 stream、进入 taskQueue、没有内部 stream/device sync，不查询 capture，不取得 graph handle。
4. eager 省略输出时，由 PyTorch wrapper 使用 torch allocator 分配；capture 时要求输出在 capture 前分配并显式传入。
5. HBG 每次调用生成的 graph package 等价于 AscendC 的 tiling data：通过 `aclrtLaunchKernelWithHostArgs` 交给 CANN，captured node 持有自己的不可变 invocation package。
6. HBG package 自包含 callable identity、参数快照、callable-local function binding 与 pristine graph，不依赖固定大小的 resident callable table。
7. TRB 使用动态、append-only 的 code registry；不复用旧 token，不覆盖旧 entry。现阶段接受其增长风险并明确记录。
8. 删除“最多 64 个 L1 callable”的公共限制。仍保留“单个 callable 最多多少个 child kernel、单次最多多少 tensor/scalar”等独立 ABI 上限。
9. 新增 L1 JIT 路径不得调用 `aclrtBinaryUnLoad` 或 `rtsBinaryUnload`。CANN binary 和图可见的 code handle 按进程生命周期 pin。
10. `pypto.l1.shutdown(device=...)` 完全可选；不调用时安静 pin 到进程结束。销毁某一张 ACLGraph 绝不自动触发 device 级 shutdown。

---

### 2. 为什么当前 API 不适合作为最终用户接口

当前 `pypto_init -> context.operator -> prepare -> warmup -> close` 是验证底层所有权和错误路径所需的控制面 API，但它把实现细节泄漏给每一个 kernel 调用者：

| 当前暴露概念 | 实际属于谁 | 为什么不应成为普通用户负担 |
| --- | --- | --- |
| `L1Context` | PyPTO runtime owner | 用户只想调用 kernel，不应手工组装 device runtime |
| `L1Operator` | compiled callable 与 native handle 的桥 | 与 Python 函数本身重复形成第二个“算子对象” |
| `prepare()` | binary/stream/event/workspace/callable 注册 | 只要求发生在 capture 外，不要求用户知道每一步 |
| `warmup()` | 首次真实调用 | 普通 eager 调用本身就可以承担 warmup |
| `close()` | 进程级资源策略 | captured graph 生命周期通常长于一个 Python 词法作用域 |
| `try/finally` | 失败所有权保护 | 正确但不适合作为每一个算子的样板代码 |

Triton 风格的关键不是少写几行，而是将对象模型对齐：

~~~text
Python function
    |
    +-- @pl.jit specialization cache
    |
    +-- hidden L1 runtime/cache
    |
    '-- direct __call__ on current torch_npu stream
~~~

用户可见对象只有一个：被 `@pl.jit` 装饰后的 kernel。编译产物、L1 runtime 和 callable registry 都是它背后的缓存层。

---

### 3. 已确认的设计决定

| 主题 | 决定 |
| --- | --- |
| 主入口 | 使用现有 `@pl.jit`，通过 `execution="l1"` 选择 L1 |
| runtime 选择 | decorator 可固定 `runtime="tensormap_and_ringbuffer"` 或 `"host_build_graph"` |
| 默认行为 | 不带 `execution="l1"` 的现有 `@pl.jit` 行为完全不变 |
| scalar 语法 | 只使用现有 `pl.Scalar[pl.FP32]` 等表达，不新增 constexpr 或新类型 |
| eager 输出 | wrapper 可用 torch allocator 自动创建 |
| capture 输出 | 必须在 capture 前创建并显式传给 kernel |
| prepare | 首次 eager 调用隐式执行；不提供难看的 batch prepare 公共接口 |
| 未 warm 即 capture | 明确失败，错误信息要求先在图外调用一次 |
| capture 感知 | PyPTO 不查询 capture 状态，不拿 graph handle |
| taskQueue | 默认走 C++ adapter、`stream(false)`、`RunOpApiV2` 和 allocator `recordStream` |
| workspace | 继续由 PyPTO 内部持有；不成为公共调用参数 |
| 并发 | v1 不支持；能检测时 fail-fast，无法观察的 graph replay 并发写入限制文档 |
| 显式 session/context manager | 不提供 |
| 全局 init | 首版不提供 `pypto.l1.init(...)` |
| 清理 | 仅提供完全可选的 device 级 `pypto.l1.shutdown(device=...)` |
| 自动 close | GC/atexit 不调用 runtime close；不制造进程退出时的 Core 或异常 |
| graph 销毁 | 单张 graph 销毁与 device shutdown 没有绑定关系 |
| callable 上限 | 移除公开的固定 64 限制 |
| HBG registry | invocation package 自包含，不按 resident 数字槽查 callable |
| TRB registry | 动态 append、内容去重、token 永不复用 |
| CANN 所有权 | CANN 持有每个 launch/captured node 的 HostArgs bytes，不推导为持有其引用的 binary |
| Binary unload | 新路径禁止调用任何 `BinaryUnLoad` |
| 平台 | 本设计和首轮验收只要求 A2/A3；A5/A5sim 不作为门槛 |

---

### 4. 公共 Python API

#### 4.1 decorator

现有签名：

~~~python
@pl.jit
def kernel(...):
    ...

@pl.jit(auto_scope=False)
def kernel(...):
    ...
~~~

建议扩展为：

~~~python
@pl.jit(
    execution="l1",
    runtime="tensormap_and_ringbuffer",
    auto_scope=True,
)
def kernel(...):
    ...
~~~

概念签名：

~~~python
def jit(
    func=None,
    *,
    auto_scope: bool = True,
    execution: Literal["default", "l1"] = "default",
    runtime: Literal[
        "tensormap_and_ringbuffer",
        "host_build_graph",
    ] | None = None,
):
    ...
~~~

规则：

- `execution="default"` 保持当前 dispatch 语义。
- `execution="l1"` 打开本文设计的隐藏 L1 registry。
- L1 未指定 `runtime` 时使用项目当前默认 runtime，即 TRB。
- decorator 已固定 runtime 后，调用时 `RunConfig.runtime` 只能省略或与其相同；冲突时在编译/设备初始化前报错。
- `@pl.jit.incore`、`inline`、`opaque`、`extern` 仍是依赖函数声明，不单独接受 `execution="l1"`。execution 只属于 orchestration entry。
- L1 仅接受单机单卡 `CompiledProgram`；host/distributed JIT entry 不得选择 L1。

#### 4.2 eager：显式输出

~~~python
@pl.jit(execution="l1")
def saxpy(
    x: pl.Tensor[[1024], pl.FP32],
    y: pl.Tensor[[1024], pl.FP32],
    alpha: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[1024], pl.FP32]],
):
    ...


out = torch.empty_like(x)
saxpy(x, y, 0.5, out=out)
~~~

调用返回 `out`，而不是 `None`。多输出返回 tuple。这样显式输出与 return-style 调用在 Python 组合上保持一致：

~~~python
out = saxpy(x, y, 0.5, out=out)
~~~

是否最终保持现有 in-place API 的 `None` 返回值，可以在实现前由兼容测试确认；本设计推荐返回输出 tensor，因为它更接近 Triton 和普通 torch op 的可组合性。

#### 4.3 eager：隐式输出分配

当所有 `pl.Out[...]` 都有完整静态 shape/dtype annotation 时，允许省略：

~~~python
out = saxpy(x, y, 0.5)
~~~

wrapper 执行：

~~~python
out = torch.empty(
    declared_shape,
    dtype=declared_torch_dtype,
    device=current_npu_device,
)
saxpy(x, y, 0.5, out=out)
return out
~~~

约束：

- 这是 PyTorch convenience wrapper 行为，不进入 simpler/native ABI。
- native PyPTO 不替输入/输出 tensor 分配内存。
- annotation 不足以推导 shape/dtype 时，必须显式传出参。
- 多输出时，全部可推导才允许全部省略；首版不支持只省略其中一部分。
- 隐式分配只承诺 ordinary eager。ACLGraph capture 的 supported 用法必须显式传入 capture 前已经分配的输出。
- wrapper 不查询 capture；若用户在 capture 内误用隐式分配，行为由 torch_npu allocator/capture 规则决定，PyPTO 不承诺可用。

#### 4.4 scalar

沿用现有语言类型：

~~~python
@pl.jit(execution="l1")
def scale(
    x: pl.Tensor[[1024], pl.FP16],
    alpha: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[1024], pl.FP16]],
):
    ...


out = scale(x, 0.125)
~~~

wrapper 必须按声明 dtype 做 bit-exact packing：

| 声明 | Python 输入示例 | task args bit pattern |
| --- | --- | --- |
| `pl.Scalar[pl.FP16]` | `1.0` | FP16 `0x3c00` |
| `pl.Scalar[pl.BF16]` | `1.0` | BF16 `0x3f80` |
| `pl.Scalar[pl.FP32]` | `1.0` | FP32 `0x3f800000` |
| integer/index | Python `int` | 对应宽度和符号位扩展 |
| `pl.Scalar[pl.BOOL]` | `True` | `1` |

不支持的 scalar dtype 在 specialization 初始化阶段 fail-fast，不允许首个 device launch 才发现。

ACLGraph capture 记录的是该次 kernel launch 的 scalar snapshot。replay 是否希望改变 scalar，是 ACLGraph 使用方的参数更新问题，不由 PyPTO隐式推断。

#### 4.5 ACLGraph

~~~python
@pl.jit(execution="l1", runtime="host_build_graph")
def fused_step(
    x: pl.Tensor[[64, 128], pl.FP32],
    y: pl.Tensor[[64, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
):
    ...


# 1. 图外普通调用：完成 specialization、runtime init、callable prepare 和真实 warmup。
warmup_out = torch.empty_like(x)
fused_step(x, y, out=warmup_out)
torch.npu.synchronize()  # caller 显式完成 warmup epoch

# 2. capture 使用预分配 tensor。
graph_in = torch.empty_like(x)
graph_mid = torch.empty_like(x)
graph_out = torch.empty_like(x)

graph = torch.npu.NPUGraph()
capture_stream = torch.npu.Stream()
with torch.npu.graph(graph, stream=capture_stream):
    torch.add(graph_in, y, out=graph_mid)
    fused_step(graph_mid, y, out=graph_out)
    torch.mul(graph_out, 2.0, out=graph_out)

# 3. replay 不回到 Python，不重新 build package。
graph.replay()
torch.npu.synchronize()
~~~

规范：

- warmup 与 capture 可以使用不同 stream。
- 从 warmup stream 切到 capture stream 前由 caller 外部同步。
- capture 内不调用 `prepare`、不分配 PyPTO device state、不注册 binary、不做 H2D、不 sync。
- capture 内的 HBG host build 只生成本次 HostArgs package；package 由 CANN launch/capture 接管。
- graph replay 不进入 Python，不进入 `JITFunction.__call__`，也无法被 Python lock 观察。
- graph、输入、输出 storage 必须由用户保持到最后一次 replay 完成并销毁 graph。
- 销毁该 graph 后，不自动调用 `shutdown`，因为同 device 上可能还有其他 graph。

#### 4.6 可选 shutdown

~~~python
# 完全可选。
pypto.l1.shutdown(device=0)
~~~

契约：

- 不调用：无警告、无异常、无 Core；资源安静地 pin 到进程退出。
- 调用前：用户保证该 device 上所有 PyPTO L1 task 已完成，所有相关 ACLGraph 已销毁且不会再次 replay。
- 调用中：关闭该 device 的新 L1 admission，执行能够安全完成的 context/resource retirement。
- 重复调用：幂等。
- 失败：保留 owner；后续可再次调用。
- GC、`weakref.finalize` 和 `atexit` 不调用 native close，只允许低级别诊断日志。
- shutdown 不由某个 kernel、某个 JITFunction 或某一张 graph 的析构触发。
- 新路径不调用 `aclrtBinaryUnLoad` 或 `rtsBinaryUnload`；进程级 CANN binary owner 仍保持 pinned。
- shutdown 不是“恢复可重新选择 runtime”的保证。首版可将 device 标记为终止使用，进程内重新 init 留到后续协议。

---

### 5. 对用户隐藏的对象模型

#### 5.1 总体结构

~~~text
JITFunction
  |
  +-- CacheKey -> CompiledProgram
  |
  '-- L1DispatchFacade
        |
        +-- L1SpecializationKey -> L1CallableRecord
        |
        '-- process L1JITRegistry
              |
              '-- device_id -> DeviceL1Owner
                    |
                    +-- runtime kind (TRB or HBG)
                    +-- taskQueue adapter
                    +-- native borrowed-device runtime
                    +-- workspace / Runtime / KernelArgs
                    +-- hidden AICore stream + events
                    +-- pinned binary owner
                    +-- callable records
                    '-- retirement state
~~~

`L1JITRegistry` 必须是进程强引用对象，而不是 weak cache。原因是 ACLGraph replay 可以在 Python kernel 对象暂时不可达之后继续使用已捕获的 handle 和 package；Python GC 不能证明 device 不再引用这些资源。

#### 5.2 registry key

Device owner 建议按以下 key 管理：

~~~python
DeviceOwnerKey(
    process_id,
    device_id,
    platform,
    runtime,
    l1_runtime_config_fingerprint,
)
~~~

首版同一 device 同时只允许一个 active runtime kind。若已有 TRB owner，再首次调用 HBG kernel，明确报错：

~~~text
device 0 already owns an active PyPTO L1 runtime
runtime='tensormap_and_ringbuffer'; requested='host_build_graph'.
Destroy all graphs, externally quiesce the device, then start a new process
or use the future runtime-switch protocol.
~~~

不自动 shutdown，不自动切 runtime，不 reset device。

#### 5.3 callable identity

不能继续以 Python 对象地址、输出目录或可复用的小整数作为持久 identity。

建议：

~~~text
CallableContentKey =
    hash(
        canonical ChipCallable bytes,
        orchestration SO bytes/hash,
        AICore image hash,
        runtime name,
        platform,
        callable ABI major,
        tensor/scalar signature
    )
~~~

实现要求：

- hash 用于索引和去重，不单独作为信任证明。
- 命中 hash 后仍比较完整 size、secondary hash 和 signature；冲突 fail closed。
- 同一 artifact 多个 Python wrapper 共享一条 callable record。
- 不同 specialization 若生成完全相同 artifact，可按 content key 去重。
- 两个 callable 的 child `func_id` 都从 0 开始完全合法；function table 永远是 callable-local。

#### 5.4 specialization identity

JIT specialization key继续由现有 `CacheKey` 决定，包括 source、shape、dtype、layout、dynamic dim、scalar specialization、platform、runtime、pass strategy 和 memory planner。

L1 在它之上再绑定：

~~~text
L1SpecializationKey =
    (JIT CacheKey, CallableContentKey, device_id, runtime_config_fingerprint)
~~~

tensor stride 不得只做“>0”检查。首个成功 enqueue 后绑定 shape/dtype/stride/layout metadata；后续调用必须一致。失败的首次 enqueue 不得提前提交 layout binding。

---

### 6. 生命周期与状态机

#### 6.1 device owner

~~~text
Absent
  |
  | first eager call
  v
Initializing --failure before ownership--> Absent
  |
  +--failure with retained native owner--> CleanupOnly
  |
  v
Ready
  |
  | optional shutdown
  v
Retiring --retryable failure--> Retiring
  |
  v
Retired
~~~

状态语义：

- `Initializing`：只允许初始化线程继续；其他线程 fail-fast，不等待 device。
- `CleanupOnly`：禁止 compile/prepare/launch，只允许 `shutdown` retry。
- `Ready`：允许新 specialization 进行 append-only prepare。
- `Retiring`：拒绝全部新调用，保留所有未确认安全释放的 owner。
- `Retired`：公共 shutdown 幂等返回；CANN binary 仍可按进程 lifetime pin。

#### 6.2 callable

~~~text
Absent
  |
  | first eager call outside capture
  v
Preparing
  |
  +--validation/compile failure--> Absent
  |
  +--enqueue failure-------------> FailedRetained or Absent
  |
  v
ReadyEnqueued
  |
  | first successful invocation enqueue
  v
Warm
~~~

`ReadyEnqueued` 和 `Warm` 都表示 host enqueue 成功，不表示 device 已完成。ACLGraph 前的 `torch.npu.synchronize()` 仍由 caller 执行。

首个调用若在 capture 内，prepare 所需的资源操作会被 runtime 拒绝。wrapper 仅根据“该 specialization 尚未准备 + prepare 失败”翻译为清晰错误；不得调用 capture query：

~~~text
PyPTO L1 specialization is not prepared for ACLGraph capture.
Call this kernel once with the same shape/dtype/layout outside capture,
synchronize externally, then capture it.
~~~

#### 6.3 append-after-warm

Triton 风格 JIT 必须允许：

~~~python
kernel_a(x, out=a)  # owner 已 Ready
torch.npu.synchronize()

# 稍后第一次看见另一个 kernel/specialization。
kernel_b(x, out=b)  # append prepare，不重建 owner
~~~

因此 native L1 phase不能在第一次 launch 后永久 `Sealed`。应改为：

- context-wide runtime state初始化后冻结；
- callable registry保持 appendable；
- 已发布 callable entry永久不变；
- 新 prepare不得改变旧 captured node引用的任何地址、token、function table或package；
- append与 device execution并发在 v1 不受支持，检测到时直接报错。

---

### 7. 第一次 eager 调用的完整流程

~~~text
Python kernel(...)
  |
  +-- bind args / infer omitted Out
  +-- build current JIT CacheKey
  +-- compile or hit CompiledProgram cache
  +-- validate current device/thread/runtime
  +-- get-or-create DeviceL1Owner
  +-- compute CallableContentKey
  +-- lookup/append callable record
  |     |
  |     +-- load/pin binary and code resources
  |     +-- pre-register AICore handle
  |     +-- TRB: append dynamic code registry entry
  |     '-- HBG: prepare host builder and context working slot
  |
  +-- build tensor/scalar snapshot
  +-- taskQueue enqueue on current stream
  |     |
  |     +-- caller stream: clear launch state / record start / AICPU
  |     +-- hidden stream: wait start / AICore / record done
  |     '-- caller stream: wait done / record serial tail
  |
  '-- commit Warm + layout binding only after successful enqueue
~~~

不允许出现：

- `aclrtSynchronizeStream`、`aclrtSynchronizeDevice`；
- device reset；
- capture query；
- `rtStreamAddToModel`；
- private AICPU execution stream；
- launch 时 lazy binary registration；
- input/output device allocation；
- `BinaryUnLoad`。

---

### 8. taskQueue 与 tensor lifetime

正式路径继续使用独立 torch_npu adapter：

~~~cpp
auto npu_stream = c10_npu::getCurrentNPUStream();
auto raw_stream = npu_stream.stream(false);

at_npu::native::OpCommand::RunOpApiV2(
    op_name,
    [lease, tensors, raw_stream]() {
        return lease->invoke(raw_stream);
    },
    false
);
~~~

必须保持的两层 lifetime：

1. queue callback 运行前：C++ lambda 持有 `std::vector<at::Tensor>`，防止 Python 引用提前释放。
2. task 已入 caller stream、device 尚未消费时：对普通 NPU caching allocator storage调用 `recordStream`。

queue descriptor必须是纯 C++、copy-safe 的 retained lease，不捕获 Python object，不在无 GIL callback里访问 nanobind/pybind object。

external/from_blob/custom allocator storage无法保证 `recordStream` 生效。v1 契约：

- 能检测为非默认 allocator 时优先 fail-fast；
- 无法可靠检测时，用户必须让 external owner存活到 stream完成；capture 时存活到 graph销毁和最后一次 replay完成；
- 文档不得声称“删除所有 Python tensor引用仍一定安全”覆盖这类 storage。

---

### 9. HBG：把 graph 当作 CANN 管理的 per-task tiling package

#### 9.1 生命周期分层

HBG 需要明确区分五种对象，但这些层不应暴露给 Python 用户：

| 层 | 内容 | owner | 生命周期 |
| --- | --- | --- | --- |
| `GraphPlan` | host build 的 canonical pristine graph | L1 callable-local单条cache | 参数语义稳定时跨invocation复用；变化后事务替换 |
| serialized launch blob | header、regions、identity、pristine payload | 调用栈临时对象 | 到 `WithHostArgs` 接管 |
| runtime-owned HostArgs | CANN 复制后的 task args 和 inline payload | CANN task/captured node | task 完成或 graph 销毁 |
| working execution slot | mutable SM、runtime arena、heap、Runtime/KernelArgs | PyPTO device owner | hidden context lifetime |
| code resources | AICPU entry、AICore binary/func handle、host orch SO | PyPTO pinned code owner | 至少覆盖所有 task/graph；binary到进程退出 |

关键边界：

> `aclrtLaunchKernelWithHostArgs` 管理 launch argument bytes，不等价于它自动管理这些 bytes 内所有 device address 所引用的 binary、workspace 或 mutable slot。

所以 HBG 可以把 graph/tiling package 交给 CANN，却仍必须由 PyPTO pin code 和 working slot。

#### 9.2 为什么 package 必须 pristine

当前 HBG scheduler执行时会原地改变：

- ready queue；
- task state；
- completion flags；
- wake list；
- host done mask；
- completed task/subtask计数；
- runtime/scheduler内部指针。

因此 captured node不能把上次执行后的 working image当作下次 replay输入。每次 invocation/replay必须：

~~~text
CANN-owned immutable pristine package
               |
               | AICPU leader restore
               v
PyPTO-owned mutable working slot
               |
               v
      scheduler execution mutates it
~~~

同一 graph连续 replay至少两次而 host不重新 build，是该协议的基本验收条件。

#### 9.3 self-contained package

建议将 HBG launch blob ABI升级，使一个 package自身携带或完整覆盖：

~~~cpp
struct HbgLaunchBlobHeaderV2 {
    uint32_t magic;
    uint16_t abi_major;
    uint16_t abi_minor;
    uint32_t header_size;
    uint32_t total_size;
    uint32_t region_count;
    uint32_t flags;

    uint64_t plan_generation;
    uint64_t plan_hash;
    uint64_t inline_payload_addr;  // CANN placeholder patches this field
    uint64_t inline_payload_size;

    HbgExecutionBinding binding;
    HbgCallableIdentity callable;
    HbgArgumentIdentity arguments;
    HbgFunctionBindingIdentity functions;
};
~~~

inline payload至少包含：

1. pristine shared-memory image；
2. pristine runtime-arena image；
3. 必要的 GM heap initializer；
4. `HbgPrebuiltInvocationState`；
5. callable-local `func_id -> device address` table；
6. host task count、argument snapshot和所有 replay reset所需字段。

当前 `HbgPrebuiltInvocationState` 已经有 1024-entry function table和 `function_binding_hash`。新设计应把它正式定义为 package trust chain的一部分，并让 `plan_hash` 覆盖该区域。

#### 9.4 去掉固定 HBG callable registry

当前路径是：

~~~text
callable_id [0, 64)
    |
    v
HbgContextRegistry.callables[callable_id]
    |
    +-- callable_hash
    '-- function_binding_hash
~~~

新路径：

~~~text
CANN-owned Hbg package
    |
    +-- content-based callable identity
    +-- callable-local function table
    +-- function-binding hash
    +-- complete plan hash
    '-- argument snapshot hash

Context-owned state
    |
    '-- only trusted working-slot binding/generation
~~~

具体改变：

- `HbgInvocationIdentity.callable_id` 不再是 registry数组下标。
- 可删除它，或保留为只用于日志的单调 trace token；validator不得检查 `< 64`。
- `HbgAicpuInvocationView` 不再要求额外传入 `HbgCallableRegistration`。
- AICPU直接验证 package内部 identity、hash、function table和 context-owned execution-slot binding。
- `HbgContextRegistry` 不再内嵌 `HbgCallableRegistry entries[64]`；只保留 context generation、working slot registration和必要控制状态。
- 旧 `simpler_aicpu_l1_hbg_register_callable` 从新 ABI执行链移除。
- 旧 ABI若要兼容，可保留只读 legacy entry，但新 JIT从不调用。

这消除了 HBG 的 callable数量上限，也避免新 callable注册改变旧 graph依赖的 resident table。

#### 9.5 CANN placeholder

canonical plan和单次 launch buffer必须分开：

~~~text
immutable canonical GraphPlan
    |
    | serialize
    v
writable one-launch byte buffer
    |
    | aclrtPlaceHolderInfo(addrOffset, dataOffset)
    v
CANN-owned device args image
~~~

原因是 placeholder API允许 runtime原地 patch传入的 writable HostArgs。不能把 canonical plan本身交给该 API后继续当作不可变 cache。

`inline_payload_addr` 应被 patch为：

~~~text
runtime_device_args_base + header_size
~~~

所有 region source都使用相对 offset，禁止把 host临时 buffer地址固化进 package。

#### 9.6 working slot

首版每个 device owner只保留一个 HBG mutable working slot：

- 符合 PyPTO 当前独占所有 AICore、不能并发的事实；
- workspace仍由 PyPTO内部管理；
- 每个 captured node有独立 pristine source package；
- 多个 node可共享同一 destination slot，但必须由外部 stream/graph时序保证不重叠。

这不是 graph package ownership的简化：source按 node独立，destination按 context共享。

未来支持并发时再引入 slot pool和 replay-aware lease；首版不提前实现。

#### 9.7 HBG 调用时序

~~~text
Python/taskQueue callback
  |
  +-- host_build_graph(current tensor addresses/scalars)
  +-- relocate to zero-based/slot-relative image
  +-- serialize pristine regions + callable-local binding
  +-- compute identity/hash
  '-- aclrtLaunchKernelWithHostArgs(
          header + regions + inline pristine payload,
          placeholder
      )

ACLGraph capture
  |
  '-- CANN captures the runtime-owned args image

Each replay
  |
  +-- AICPU validates package and destination binding
  +-- restores pristine source into working slot
  +-- rewires trusted runtime pointers
  +-- releases scheduler/AICore
  '-- completes operator fork/join
~~~

launch不做显式 H2D graph upload；HostArgs copy是 CANN kernel launch参数机制的一部分。

---

### 10. TRB：动态 append-only code registry

#### 10.1 为什么 CANN HostArgs 不能替代 TRB registry

TRB invocation snapshot可以由 CANN持有：

- tensor descriptors；
- tensor device pointers；
- scalar bit patterns；
- callable token；
- context-lifetime Runtime/KernelArgs地址。

但 TRB AICPU仍需跨 invocation保留：

- orchestration SO的 `dlopen` handle；
- orchestration entry/config function pointer；
- callable-local AICore function table；
- code identity和ABI metadata。

这些不是简单 task args bytes，不能因为 HostArgs被捕获就卸载。

#### 10.2 callable id 与 content key

实现保留已有 wire ABI 的 `int32_t callable_id`，但完全取消了 `[0, 64)` 槽位语义。
这是有意的最小 ABI 改造：本次问题的根因是可覆盖的定长表，不是 32-bit 宽度本身。

当前规则：

- id从 0 单调增长，在同一 process-owned L1 owner内永不循环、不复用、不覆盖。
- Python使用canonical `ChipCallable` bytes的SHA-256做content dedupe；同内容的不同wrapper共享已有id。
- TRB register ABI v2同时携带 `callable_id` 和非零 `callable_hash`。
- 相同id、hash和callable-local kernel table完全一致时注册幂等；任意一项不同则fail closed。
- `INT32_MAX`用尽时报identity-space exhausted，不wrap。这是wire类型的理论边界，不是公开capacity。
- 已capture的旧node持有的id始终解析到同一entry。

未来若扩展到64-bit token，必须作为显式ABI升级同时更新host、AICPU、queue-call快照和测试；
不在这次已验证路径中引入两种id宽度。

#### 10.3 AICPU 数据结构

实现采用每个callable一个稳定地址node的单链append-only registry：

~~~cpp
struct L1OrchSoNode {
    int32_t callable_id;
    uint64_t callable_hash;
    OrchSoEntry entry;
    L1OrchSoNode *next;
};

L1OrchSoNode *l1_orch_so_head;
~~~

没有使用可能搬移元素的 `std::vector`，也没有把新L1 entry放回legacy
`orch_so_table_[64]`。node在完整 `dlopen`/`dlsym`/地址表快照成功后才publish到head；失败时释放
未发布node，既有链不变。

entry发布：

~~~text
allocate/fill private entry
  |
  +-- validate SO and function symbols
  +-- copy callable-local kernel table
  +-- seal complete identity
  '-- release-publish Ready
~~~

只在capture外的首次eager prepare路径增长。launch路径只读ready entry，不做 `dlopen`、malloc或替换。

首版lookup是O(N)链表遍历。这是已记录、已接受的TRB长时风险：先保证地址稳定和旧graph
不失效，再用实际long-running specialization数据决定是否换成chunked registry或附加稳定索引。

#### 10.4 注册 ABI

`L1RegisterCallableArgs` 已升级为 ABI version 2，实际形态：

~~~cpp
struct L1RegisterCallableArgs {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t callable_id;
    uint32_t kernel_count;
    uint64_t callable_hash;
    uint64_t dev_orch_so_addr;
    uint64_t dev_orch_so_size;
    char device_orch_func_name[...];
    char device_orch_config_name[...];
    L1CallableKernelAddr kernel_addrs[L1_MAX_KERNELS_PER_CALLABLE];
};
~~~

仍然保留单 callable child kernel数上限。它限制单份 function-binding ABI大小，不是“总共能注册多少 kernel函数”。

invocation ABI保持已上板的int32 id形态：

~~~cpp
struct L1AicpuInvocationArgs {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t callable_id;
    KernelArgs kernel_args;
    ChipStorageTaskArgs orch_args;
};
~~~

hash在prepare/register时封印immutable identity；launch按id只读已发布entry，随后将该entry的
callable-local function table写入本轮Runtime。

#### 10.5 不复用、不覆盖

明确禁止：

- registry满后从 slot 0重新覆盖；
- LRU驱逐；
- 仅凭 Python kernel对象析构卸载；
- 仅凭一张 graph销毁释放；
- 用 event query推断“所有 graph未来都不会 replay”；
- 新注册同 token不同内容。

CANN没有向 PyPTO提供“所有持有该 callable token的 graph/task都已销毁”的通用回调。因此循环复用会把旧 graph静默指向新 code，是比增长更危险的错误。

#### 10.6 动态增长风险

这项设计按用户决定先接受，但必须记录：

1. 长进程不断生成不同 JIT specialization时，AICPU `dlopen` handle、临时 SO文件、kernel table和host/device artifact会持续增长。
2. 没有 graph-aware eviction协议，所以不能安全做通用 LRU。
3. AICPU heap碎片和文件描述符/映射数量可能成为资源瓶颈。
4. 注册失败发生在首次 eager调用，不得污染旧 entry。
5. growth必须用checked arithmetic，单次 SO大小和累计 pinned bytes都需观测。
6. 不能将 OOM解释为可覆盖旧 token；只能报 `ResourceExhausted`。

首版不提供公开 callable count上限。可以提供诊断和可选软阈值：

~~~text
registered_callable_count
pinned_orch_so_bytes
pinned_aicore_image_bytes
registry_chunk_count
largest_callable_bytes
~~~

软阈值只告警；若配置硬 byte budget，则在任何 device/DLOpen状态改变前fail-fast。它按 bytes限制资源，而不是恢复“64个”这种无语义数量上限。

---

### 11. Binary 与 code resource 所有权

#### 11.1 四类资源必须分开

| 资源 | 典型建立方式 | 新 L1 JIT 生命周期 |
| --- | --- | --- |
| simpler AICPU runtime binary | `aclrtBinaryLoadFromData` | process pinned，绝不 BinaryUnLoad |
| AICore binary/function handle | register binary/kernel | process pinned，不做 unregister/reuse |
| TRB callable orchestration SO | AICPU内 `dlopen` | append registry entry，至少到 shutdown；无安全回收则到进程退出 |
| HBG host orchestration SO | host `dlopen`，用于 build graph | callable owner pin；shutdown后仅在外部quiescence成立时释放 |

#### 11.2 强制规则

新代码中禁止出现：

~~~cpp
aclrtBinaryUnLoad(...);
rtsBinaryUnload(...);
~~~

不只禁止 happy path调用，也禁止：

- init rollback；
- prepare rollback；
- shutdown；
- Python析构；
- atexit；
- error recovery。

Binary load成功后若后续步骤失败，将 handle转移到 process-lifetime pinned owner，而不是尝试 unload。

#### 11.3 与现有 `LoadAicpuOp::Finalize` 的关系

现有 `LoadAicpuOp::Finalize()` 会根据 load mode调用 BinaryUnLoad，因此新 JIT owner不能直接复用它作为最终 teardown。

建议拆分：

~~~cpp
class LoadAicpuOp {
 public:
  // Legacy L2/L3 behavior, untouched.
  int FinalizeLegacy();

  // Transfer loaded binary + function handles to a process-lifetime owner.
  // No unload, including destructor.
  PinnedAicpuBinary DetachForL1ProcessLifetime();
};
~~~

或者给 L1单独实现 `PinnedAicpuBinaryOwner`，从初始化开始就不走可 unload owner。

要求：

- legacy L2/L3 teardown语义不因本设计改变；
- pinned owner destructor为 no-op；
- 进程退出交由 OS/driver进程资源回收；
- 不伪装成“已释放”；诊断中标明 intentionally pinned；
- 单元测试用 fake HostApi确认整个 L1 JIT路径的 unload call count始终为 0。

#### 11.4 shutdown 的精确定义

`shutdown` 是逻辑 device-owner retirement，不是 binary unload：

~~~text
Ready
  |
  +-- close admission
  +-- require caller-declared external quiescence
  +-- retire mutable workspace/stream/event when safely provable
  +-- retire callable-side non-binary resources when protocol permits
  '-- transfer all binary handles to process-pinned owner
~~~

若 TRB AICPU registry清理需要额外 device task：

- cleanup task只能在 capture外enqueue；
- shutdown不做内部 stream/device sync；
- owner在 cleanup完成可被非阻塞证明前保持在 `Retiring`；
- cleanup enqueue失败时owner保留，可重试；
- 无法证明完成时宁可继续 pin，不提前 `dlclose`/free；
- 首版可以把物理资源保留到进程退出，不能以不安全回收换取“shutdown看起来释放成功”。

---

### 12. 并发边界

#### 12.1 能检测的并发

以下情况立即报错：

- 两个 host线程同时首次初始化同一 device；
- compile/prepare和另一次 host launch重叠；
- 同一 hidden owner同时准备 TRB和HBG；
- eager从不同 caller stream进入，而上次 serial tail仍未完成；
- shutdown与新调用重叠；
- 新 callable append发生在已知 active host enqueue窗口。

实现可以使用：

- Python registry mutex；
- native operation mutex；
- owner thread/pid检查；
- stream identity；
- 非阻塞 event status；
- state generation。

禁止用内部 sync“解决”并发。

#### 12.2 无法检测的并发

ACLGraph replay绕过 Python和PyPTO host。以下情况首版无法完整检测：

- 两张 graph在不同 stream并发 replay；
- graph replay与 eager L1 launch重叠；
- graph replay期间首次注册新 callable；
- graph销毁前调用 shutdown但调用方错误地声称已 quiescent。

这些路径明确 unsupported。因为 PyPTO当前占用全部 AICore并共享 workspace/working slot，未检测到的重叠可能导致数据破坏或 runtime错误。

文档不能声称 host lock提供了 device并发安全。

#### 12.3 后续方向

完整并发支持需要至少一项：

- ACLGraph/runtime外部资源 retain/release hook；
- graph-aware execution-slot pool；
- graph replay admission token；
- host_build_graph统一编排多个算子；
- device-side generation/lease协议。

不在本次 Triton风格API改造中提前实现。

---

### 13. 错误模型

#### 13.1 错误类型

建议公共异常层次：

~~~python
class L1Error(RuntimeError): ...
class L1WarmupRequiredError(L1Error): ...
class L1RuntimeConflictError(L1Error): ...
class L1ConcurrencyError(L1Error): ...
class L1ResourceExhaustedError(L1Error): ...
class L1ShutdownError(L1Error): ...
~~~

普通用户不接触 cleanup context，但 registry必须内部持有失败owner：

~~~text
exception returned to user
    |
    '-- process registry keeps CleanupOnly/Retiring owner strongly reachable
~~~

#### 13.2 pre-enqueue failure

以下失败不得改变 persistent state：

- 参数个数/方向错误；
- device/dtype/shape/stride不匹配；
- unsupported scalar；
- runtime conflict；
- callable artifact/hash校验失败；
- HBG package size/region/hash错误；
- registry byte budget不足。

使用临时候选对象完成全部验证，最后一步才 publish/commit。

#### 13.3 enqueue后失败

taskQueue callback或native launch部分成功后失败时：

- 不回收可能被device引用的args/code/workspace；
- owner进入 failed-retained/poisoned状态；
- AICore-first launch若AICPU enqueue失败，继续使用已经设计的 host failure CANCEL + hidden done join错误闭包；
- 后续调用拒绝，允许可选 shutdown重试；
- 不调用 BinaryUnLoad。

---

### 14. 与当前低层 API 的关系

#### 14.1 保留但降级为 advanced

以下类型暂不删除：

- `L1Config`；
- `L1Context`；
- `L1Operator`；
- `pypto_init`。

用途：

- native bring-up；
- fault injection；
- 精细测试；
- 调试特定 callable prepare/launch；
- 兼容现有内部ST。

它们不再出现在主用户文档“安全使用范式”中。

#### 14.2 不允许两套 owner独立存在

如果 advanced API和JIT API同时作用于同一 device，必须共享同一 `DeviceL1Owner` 或明确互斥。不能各自建立 hidden stream/workspace/runtime：

~~~text
JIT API owner
      X  forbidden parallel ownership
manual L1Context owner
~~~

首版最简单规则：若 device已有任一 owner，另一种入口报 conflict。

---

### 15. 与 pto2 历史实现的对比

| 维度 | pto2 历史 L1 | 本设计 |
| --- | --- | --- |
| runtime | 仅接近当前 TRB的动态/ring-buffer路径 | TRB + HBG |
| 用户入口 | kernel-like launch，但需要旧式资源约定 | `@pl.jit(execution="l1")` 直接调用 |
| workspace | 外部传入 | PyPTO内部管理 |
| AICPU stream | PyPTO私有stream，并通过model attach跨边界 | 直接使用caller stream |
| AICore stream | hidden/private | hidden，但严格在单算子fork/join内 |
| ACLGraph | 查询capture、`rtStreamAddToModel` 等 | 不查询capture、不拿graph handle、不attach model |
| AICPU提前启动 | 为性能跨越单kernel边界 | 明确禁止；未来由host_build_graph获取跨算子优化 |
| taskQueue | 无当前正式adapter契约 | `RunOpApiV2 + stream(false) + Tensor lease` |
| HBG graph package | 不支持当前HBG | CANN-owned per-node pristine tiling package |
| callable identity | 历史实现语义 | content key + callable-local function table |
| binary lifetime | 历史清理逻辑 | 新路径明确永不 BinaryUnLoad |
| graph生命周期 | 依赖model/capture耦合 | graph-transparent；CANN持args，PyPTO pin引用资源 |

pto2可借鉴的是“普通kernel调用形态”；不能照搬的是提前启动AICPU、把hidden stream挂进model以及跨越单算子边界的性能优化。

---

### 16. 具体代码改造范围

以下是建议的实现文件和职责，行号以当前主分支附近符号为准，实施时按symbol定位。

#### 16.1 PyPTO Python

1. `python/pypto/jit/decorator.py`
   - 扩展 `_JITDecorator.__call__`：接受 `execution`、`runtime`。
   - `JITFunction` 保存不可变execution/runtime默认值。
   - `__call__` 在 `execution=="l1"` 时转交隐藏L1 facade。
   - cache key继续包含runtime；decorator与RunConfig冲突时fail-fast。
   - 支持L1 omitted-Out binding。

2. 新增 `python/pypto/runtime/l1_jit.py`
   - `L1JITRegistry`；
   - `DeviceL1Owner`；
   - `L1CallableRecord`；
   - state machine、strong owner、shutdown；
   - content identity、dedupe、layout commit；
   - 对当前低层worker的封装。

3. `python/pypto/runtime/l1.py`
   - 将参数构造、scalar pack、tensor descriptor校验抽成JIT/manual共用helper。
   - 移除JIT路径对 `MAX_REGISTERED_CALLABLE_IDS` 的检查。
   - advanced context是否保留legacy 64限制取决于native ABI迁移；最终应同步去除。

4. `python/pypto/l1.py`
   - 暴露 `shutdown(device: int) -> None`。
   - 低层类型可继续导出，但文档标advanced。

5. `python/pypto/__init__.py` 或已有public export位置
   - 确保 `pypto.l1.shutdown` 稳定可导入。

6. `python/bindings/torch_npu_l1_adapter.cpp`
   - 复用现有 `stream(false)`、`RunOpApiV2`、Tensor lease和 `recordStream`。
   - 不增加capture query。
   - 新增shutdown/control queue-call只在必要时；仍不得捕获Python object。

#### 16.2 simpler host/native

1. `runtime/src/common/task_interface/callable_protocol.h`
   - 不再把 `MAX_REGISTERED_CALLABLE_IDS=64` 作为新L1协议。
   - 可保留legacy常量供旧ABI，不得从新Python API导出为公共capacity。

2. `runtime/src/common/task_interface/l1_aicpu_args.h`
   - 增加TRB registration/invocation ABI v2；
   - 保留non-negative `int32_t callable_id`，取消64-slot含义；
   - registration新增不可变 `callable_hash`；
   - 保持单callable kernel count、tensor/scalar args等独立ABI上限。

3. `runtime/src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`
   - L1使用稳定node的linked append-only registry，legacy L2/L3继续使用 `orch_so_table_[64]`；
   - 注册时分配/加载，launch只读；
   - duplicate content幂等；
   - 禁止slot overwrite和token reuse；
   - 收集pinned bytes/count诊断；
   - shutdown cleanup若无法安全确认则继续pin。

4. `runtime/src/common/platform/onboard/host/device_runner_base.{h,cpp}`
   - native context在首次launch后仍允许append prepare；
   - host `callables_` 按token/content key组织；
   - prepare先完整validate，再publish；
   - HBG不再enqueue resident callable registration；
   - launch路径使用prepared handle，不允许lazy registration；
   - JIT owner teardown走pinned-binary路径。

5. `runtime/src/common/aicpu_loader/host/load_aicpu_op.{h,cpp}`
   - 增加process-lifetime detach/pin能力；
   - 新L1路径不调用现有 `Finalize()` 中的 BinaryUnLoad分支；
   - legacy L2/L3行为保持。

6. `runtime/src/common/worker/chip_worker.{h,cpp}`
   - 提供append callable和logical shutdown C++ facade；
   - dispatch state继续保护callback与finalize竞态；
   - retained owner失败语义不变。

7. `runtime/src/common/worker/pto_runtime_c_api.h` 与 `c_api_shared.cpp`
   - 新增版本化token API；
   - 去除新ABI文档中的64上限；
   - C ABI底层仍强制device pointer、shape/stride/bounds等校验。

#### 16.3 HBG

1. `runtime/src/common/task_interface/hbg_launch_blob.h`
   - ABI v2；
   - identity不再检查callable id小于64；
   - package hash覆盖function table和全部pristine region。

2. `runtime/src/common/task_interface/hbg_aicpu_invocation.h`
   - view构造不再依赖 `HbgCallableRegistration`；
   - 对package内部trust chain做完整校验。

3. `runtime/src/common/task_interface/hbg_context_registry.h`
   - ABI升级；
   - 删除fixed callable registry；
   - 保留context-owned execution slot/generation。

4. `runtime/src/common/task_interface/hbg_callable_registry.h`
   - 从新路径删除；
   - 若保留仅用于legacy ABI兼容和旧UT。

5. `runtime/src/a2a3/runtime/host_build_graph/aicpu/aicpu_executor.cpp`
   - 移除run前resident callable slot acquire；
   - 从本次package读取/校验prebuilt invocation和function binding；
   - 每次replay先restore再dispatch。

6. `runtime/src/common/worker/hbg_callable_function_binding.h`
   - 保持每个callable独立1024-entry table；
   - 提供canonical serialization/hash helper。

#### 16.4 bindings、stubs与文档

- `runtime/python/bindings/task_interface.cpp`：新native entry。
- `runtime/python/simpler/task_interface.py`：新worker方法、异常和mode保护。
- PyPTO对应 `.pyi`：decorator参数和 `shutdown`。
- 正式用户文档最终应在 `docs/en` 与 `docs/zh` 同步；本文件是实施设计参考。

---

### 17. 实现顺序

#### Phase A：公共 API 骨架，不改变native协议

1. 给 `@pl.jit` 增加 `execution/runtime` metadata。
2. 新建隐藏 `L1JITRegistry`，暂时适配现有manual context。
3. 实现direct call、eager output allocation、scalar pack、taskQueue。
4. 加first-capture warmup错误翻译和shutdown空壳/state admission。
5. Python无硬件UT通过后再改native。

目的：先冻结用户调用形态，避免native重构期间API来回变化。

#### Phase B：HBG package self-contained

1. 将function table正式纳入package hash。
2. 修改HBG view/validator，不读取resident callable registry。
3. 移除HBG callable register task和fixed array。
4. 保持execution slot context-owned。
5. 验证旧captured node在新增其他callable后仍replay同一code。

HBG先做，因为它可以直接彻底消除64上限，而不是引入动态device registry。

#### Phase C：TRB dynamic registry

1. 引入v2 registration identity ABI。
2. AICPU linked append registry。
3. host content dedupe和单调id。
4. 移除Python/native 64检查。
5. 故障注入growth allocation/dlopen/symbol失败。
6. 压测连续specialization并记录资源增长。

#### Phase D：pinned binary owner与shutdown

1. 将新L1 binary owner从legacy Finalize路径分离。
2. 所有init/prepare失败都转移到strong retained owner。
3. 实现device级admission close和retryable retirement。
4. 增加“新路径unload调用次数必须为0”的测试。
5. GC/atexit静默策略。

#### Phase E：A2/A3 ACLGraph验收

1. TRB golden path；
2. HBG golden path；
3. torch predecessor -> L1 -> torch successor；
4. 多次replay、多个callable、同一graph两个L1 node；
5. 新callable append后旧graph replay；
6. taskQueue和tensor lifetime压力；
7. shutdown前置条件和不自动close。

---

### 18. 测试设计

#### 18.1 Python无硬件UT

| 用例 | 断言 |
| --- | --- |
| bare现有 `@pl.jit` | 行为不变，不创建L1 owner |
| `execution="l1"` | 首调compile/prepare/launch，后续cache hit |
| runtime冲突 | native init前失败 |
| omitted single Out | 使用torch allocator并返回tensor |
| omitted multiple Out | 返回tuple |
| annotation不足 | 要求显式out |
| explicit Out | 不额外分配 |
| scalar pack | FP16/BF16/FP32/int/bool bit-exact |
| unsupported scalar | 初始化前拒绝 |
| first prepare失败 | 不提交warm/layout |
| output stride变化 | warm后拒绝 |
| device/thread变化 | fail-fast |
| KeyboardInterrupt during init | registry持有cleanup owner |
| GC JITFunction | 不触发native close |
| repeated shutdown | 幂等 |
| shutdown failure | owner retained并可retry |
| >64 logical records | Python不再报固定capacity |

#### 18.2 simpler C++无硬件UT

TRB registry：

- chunk边界前后append；
- 以测试chunk size强制多次grow；
- duplicate identity返回existing token；
- hash碰撞但完整identity不同报conflict；
- token永不复用；
- malloc失败不污染已有entry；
- `dlopen`/symbol/kernel-table失败事务回滚；
- launch查旧token在多次grow后仍正确；
- pinned byte checked arithmetic；
- concurrent registration被拒绝或串行publish；
- cleanup失败保留entry。

HBG：

- package内function table参与hash；
- 两个callable都使用 `func_id=0`，各自解析到自己的地址；
- 不提供resident registration也能建立invocation view；
- callable identity/table/hash任一篡改都拒绝；
- old package在build/register新callable后仍验证通过；
- pristine restore后第二次执行状态与第一次相同；
- placeholder patch只修改one-launch buffer；
- working slot binding不匹配时fail closed。

binary：

- fake `aclrtBinaryLoadFromData` 成功后，后续每个故障点都不调用unload；
- shutdown不调用unload；
- destructor/atexit path不调用unload；
- legacy L2/L3 Finalize测试保持原预期。

#### 18.3 A2/A3 onboard ST

最小必过矩阵：

| runtime | 场景 |
| --- | --- |
| TRB | first eager、second eager、换输入验数 |
| TRB | warmup -> 独立stream capture -> 8次replay |
| TRB | graph中torch add -> L1 -> torch mul |
| TRB | 同graph两个不同callable，二者 `func_id=0` |
| TRB | 新callable append后replay旧graph |
| HBG | first eager、second eager、pristine restore |
| HBG | warmup -> capture -> 8次replay |
| HBG | 同graph两个不同callable |
| HBG | 新package/callable出现后旧graph继续replay |
| 两者各自 | enqueue后立即删Python tensor引用并施加allocator压力 |
| 两者各自 | graph销毁前不shutdown；销毁后可选shutdown |

不要求A5/A5sim作为本次合入门槛。

#### 18.4 并发负面测试

- 两个Python线程同时首次调用：一个成功，另一个明确 `L1ConcurrencyError`，不得hang。
- 已知不同stream且old tail未完成：拒绝。
- shutdown与调用重叠：调用拒绝。
- graph并发replay不能可靠host检测：测试标为unsupported contract，不伪造“已保护”结论。

---

### 19. 兼容与迁移

#### 19.1 用户代码迁移

Before：

~~~python
compiled = kernel.compile(x, y, out)
ctx = pypto_init(programs=[compiled], device=torch.npu.current_device())
op = ctx.operator(compiled)

try:
    ctx.prepare()
    op.warmup(x, y, out=out)
    torch.npu.synchronize()
    op(x2, y2, out=out2)
finally:
    ctx.close()
~~~

After：

~~~python
@pl.jit(execution="l1")
def kernel(...):
    ...


kernel(x, y, out=out)  # compile + prepare + warmup
torch.npu.synchronize()
kernel(x2, y2, out=out2)
~~~

ACLGraph before/after的根本执行协议不变，变化只是prepare/context ownership被隐藏。

#### 19.2 ABI兼容

- ChipCallable既有L2/L3 wire layout不得因L1 token改变。
- L1 registration/invocation使用独立ABI major。
- HBG launch blob/context registry显式bump ABI。
- 新AICPU runtime与host必须成对部署。
- legacy 64-slot entry可在过渡期保留，但新JIT不得生成它。
- L2/L3 runtime不引用新L1 dynamic registry，资源和行为保持。

---

### 20. 可观测性

建议提供只读debug信息，但不成为正常使用必需：

~~~python
pypto.l1.debug_state(device=0)
~~~

返回概念字段：

~~~python
{
    "state": "ready",
    "runtime": "tensormap_and_ringbuffer",
    "registered_callables": 17,
    "specializations": 23,
    "pinned_orch_so_bytes": 12_345_678,
    "pinned_binary_count": 1,
    "workspace_bytes": ...,
    "last_error": None,
}
~~~

是否公开此函数可后置；内部日志/测试至少应能读取等价计数。

日志原则：

- 首次device owner init：INFO一次；
- 新callable append：DEBUG/INFO，包含token/hash/bytes；
- content dedupe：DEBUG；
- 超过软阈值：WARN一次/分级；
- GC/atexit保留资源：默认静默或DEBUG；
- 不调用shutdown：不是warning；
- shutdown因未quiescent造成错误：ERROR并保留owner。

---

### 21. 明确拒绝的替代方案

#### 21.1 自动创建并销毁 context

拒绝：

~~~python
with pypto.l1(...):
    kernel(...)
~~~

graph可超出词法scope继续replay，RAII/context manager无法证明device引用结束，容易过早close。

#### 21.2 每个 JITFunction一个native context

拒绝。多个kernel会重复workspace/stream/event/runtime，并违反单device owner。

#### 21.3 graph销毁自动shutdown

拒绝。一张graph销毁不代表同device其他graph不存在。

#### 21.4 registry循环复用

拒绝。没有CANN graph引用计数时，旧captured token可能在未来replay；覆盖会静默执行错误code。

#### 21.5 依赖 BinaryUnLoad后的runtime保活

拒绝。当前没有公开契约保证captured graph仍引用funcHandle时，BinaryUnLoad后runtime替用户继续保活binary。

#### 21.6 query capture并走双分支

拒绝。L1算子应capture-transparent；首次capture失败通过prepared状态翻译，不主动查询。

#### 21.7 rtStreamAddToModel

拒绝。它让AICPU orchestration越过单算子边界提前执行。未来跨算子性能优化由host_build_graph完成。

#### 21.8 内部stream sync

拒绝。正常launch、prepare和capture路径都不做stream/device sync。shutdown也以保守pin替代不安全的同步清理。

---

### 22. 风险清单

#### R1：TRB registry无界增长

级别：当前最大已知工程风险。
接受理由：没有graph-aware安全驱逐协议；稳定token优先于内存回收。
当前缓解：content dedupe、稳定linked node、失败前完整构造、id不复用。
待补缓解：byte accounting、软阈值、ResourceExhausted分类和长时压测。
最终解决：CANN提供外部资源retain/release，或PyPTO获得graph生命周期通知。

#### R2：CANN HostArgs snapshot语义

风险：HBG依赖captured node持有完整变长args和placeholder内联payload。
缓解：A2/A3上板验证同graph多次replay、host buffer销毁/复用后仍正确；严格验证args size和placeholder offset。
失败策略：capability fail closed，不退回显式launch时H2D。

#### R3：shared HBG working slot

风险：两张graph并发replay会踩同一mutable state。
缓解：v1明确不支持并发；能检测的host并发报错。
最终解决：execution-slot pool + graph-aware lease。

#### R4：进程级binary pin

风险：长进程code资源不释放。
接受理由：正确性优先，且用户明确禁止BinaryUnLoad。
缓解：dedupe、统计、文档提示；资源极限时fail-fast。
不可使用的缓解：在graph可能存活时unload。

#### R5：首次capture错误翻译依赖runtime失败

风险：不同CANN版本返回码可能变化。
缓解：翻译条件同时要求 `state is not prepared`；保留原始error code/message；上板版本矩阵。
不采用：capture query。

#### R6：隐式output allocation被误用于capture

风险：wrapper不查capture，用户省略out。
缓解：文档和示例只展示capture显式out；若torch_npu明确提供普通op一致的无副作用capture-safe allocator契约，可后续放宽。

#### R7：host锁无法保护replay并发

风险：用户误以为“有mutex就安全”。
缓解：文档、错误信息和测试明确host lock边界；不宣传完整并发支持。

---

### 23. 完成标准

只有同时满足以下条件，才可以将Triton风格L1 API标为supported：

#### 公共体验

- 用户只写 `@pl.jit(execution="l1", runtime=...)` 和 `kernel(...)`。
- ordinary eager不要求 `init/prepare/warmup/context/close`。
- scalar只使用现有 `pl.Scalar[...]`。
- eager支持torch allocator返回式输出。
- capture示例只使用预分配输出。
- 可选 `shutdown` 不调用也不会产生错误。

#### runtime正确性

- 默认taskQueue路径使用 `stream(false)`、`RunOpApiV2`、Tensor lease和allocator `recordStream`。
- AICPU使用caller stream；内部AICore stream不外露。
- 单算子内部完整fork/join。
- launch不sync、不reset、不query capture、不attach model。
- HBG package self-contained且每次replay恢复pristine state。
- TRB registry动态append，旧token永不失效。
- 公共64-callable限制被移除。
- 新路径任何成功/失败/shutdown/析构路径的 BinaryUnLoad调用次数都为0。

#### 兼容性

- 非L1 `@pl.jit` 行为不变。
- L2/L3 ChipCallable wire和runtime回归通过。
- advanced manual L1 tests继续通过或有明确迁移说明。
- 仅A2/A3作为本阶段硬件门槛。

#### ACLGraph实证

- TRB和HBG都完成图外warmup、换stream capture、torch前后继、多次replay验数。
- 同一graph包含两个不同L1 callable。
- 两个callable可各自使用 `func_id=0`。
- 新callable append后旧graph继续正确replay。
- host临时HBG serialization buffer释放/复用后，captured graph仍正确。
- 默认allocator下删除Python临时引用并施加分配压力仍正确。

---

### 24. 最终原则

这次API改造不改变L1的本质边界：

> 对用户，它是一个普通的 `@pl.jit` kernel；对torch_npu，它是当前stream上的一个普通异步算子；对CANN，HBG graph是每个task/captured node拥有的变长tiling package；对PyPTO，workspace、working slot和code resources是必须独立pin住的被引用资源。

由此得到三条不可打破的所有权规则：

1. CANN owns invocation bytes，不自动推导为owns referenced binary。
2. PyPTO owns code/workspace/working state，不感知某一张ACLGraph的capture与销毁。
3. 没有可靠release信号时宁可append/pin，也不复用token、不卸载binary、不猜测graph已结束。

该方案把当前“正确但难用”的manual L1控制面保留下来，同时在其上建立一个真正可供PyTorch用户直接调用的Triton风格产品接口。

---

### 25. 实际落地结果

本节记录实现后的事实；若前文某个“建议”与本节冲突，以本节和当前源码为准。

#### 25.1 用户实际看到的API

~~~python
@pl.jit(execution="l1")
def add(x: ..., y: ..., out: pl.Out[...]):
    ...

# 首次ordinary eager：specialize + compile + owner init + prepare + launch。
z = add(x, y)
torch.npu.synchronize()

# capture前分配输出；capture内仍是普通Python call形态。
z = torch.empty_like(x)
with torch.npu.graph(graph, stream=capture_stream):
    add(x, y, out=z)

# 完全可选，且调用前由用户保证所有task/graph已终止引用。
pypto.l1.shutdown(device=0)
~~~

`@pl.jit` 保存 `execution/runtime` 元数据；非L1 decorator的compile/dispatch/cache路径不变。L1的
`compile()` 和 `lower()` 自动生成A2/A3 onboard `RunConfig`，仍允许CPU sample tensor只做specialization；
真正dispatch强制所有tensor在同一NPU device。

输出分配只发生在PyTorch wrapper：

- 所有pure `pl.Out[...]` 都省略时，根据静态annotation用 `torch.empty` 分配并返回。
- 所有输出都显式给出时，不额外分配，返回同一tensor或tuple。
- 部分输出省略直接报错，避免一半alias、一半allocator所有的模糊语义。
- capture不查询图状态；因此契约而不是隐式探测来要求capture内显式传out。

#### 25.2 隐藏device owner与append流程

`python/pypto/runtime/l1_jit.py` 使用模块级strong registry保持每个device唯一owner：

~~~text
@pl.jit specialization
        |
        v
canonical CompiledProgram
        |
        v
device owner lookup/create -- one platform/runtime/frozen L1Config
        |
        +-- content hit --> existing L1Operator
        |
        '-- content miss --> L1Context.add_program()
                              |
                              '-- monotonic int32 id + lazy prepare
        |
        v
taskQueue adapter enqueue --> caller stream ordinary operator
~~~

该registry没有atexit finalizer，不依赖JITFunction或Tensor wrapper的GC时机。首次native init失败但保留了
cleanup owner时，`L1InitializationError.cleanup_context` 会被strong registry接管，使显式shutdown仍可重试。

当前host并发检测是best-effort：owner thread亲和，且一次Python dispatch/shutdown必须取得non-blocking
invoke lock。这不被宣传为graph replay并发保护；CANN在host不可见的并发replay仍属于v1不支持范围。

#### 25.3 manual L1与JIT共用的参数契约

`python/pypto/runtime/l1.py` 仍是唯一参数打包与底层强校验入口：

- tensor强制NPU/current device、静态shape、dtype、正的uint32 stride、非autograd。
- 首个成功enqueue绑定shape/dtype/stride；后续布局改变在enqueue前拒绝。
- scalar使用旧有 `pl.Scalar[...]` annotation，按声明dtype做bit-exact low-byte pack，包括FP16、BF16、FP32、
  FP64、整数和bool。
- `L1Context.add_program()` 允许首次launch之后append新callable；`prepared` 改为per-callable状态。
- canonical `ChipCallable` 内容相同时复用已有state，不重复注册。

manual `pypto_init/context/operator` 保留为advanced/debug控制面，但不再是普通用户文档的首选入口。

#### 25.4 HBG落地的生命周期

HBG的callable身份和callable-local function table由每个launch package自包含。native路径已移除：

- Host每callable的 `HbgCallableRegistration` owner。
- prepare时 `simpler_aicpu_l1_hbg_register_callable` task。
- AICPU execute时对resident `callables[callable_id]` 的acquire。
- `HbgContextRegistry` 中的固定64项callable表。

`HbgContextRegistry` 现在只保留context-owned execution-slot/generation状态。每次launch构造可写serialization
buffer，然后通过 `aclrtLaunchKernelWithHostArgs` 把完整bytes和placeholder交给CANN。CANN为eager task或
captured node保留该次args snapshot；AICPU leader每次执行/replay都从它的pristine regions恢复到
context-owned mutable working slot，才放行scheduler。

这保证两个captured HBG node不会共享或覆盖一份host graph image，也不会因为各自的child
`func_id` 都从0开始而冲突。单一working slot/workspace的安全前提仍是这些node不并发执行。

#### 25.5 TRB落地的code registry

TRB的AICPU进程保留legacy L2/L3 `orch_so_table_[64]`，但borrowed L1完全不读写该表。L1使用
append-only `L1OrchSoNode` 链：每个node持有id、hash、`dlopen` handle、entry/config/bind function、
callable-local kernel地址快照和发布状态。

prepare是唯一增长点，launch仅遍历已发布node并读它。duplicate id只有在hash、kernel count和全部
kernel binding相同时幂等，否则直接conflict。注册失败不会把半成品node连入全局链。

当前没有驱逐、循环复用或公开callable count限制。链表lookup与无界资源增长是明确接受的风险，
不是已经解决的问题。实现优先保证旧graph的id、code handle和地址永不被新specialization覆盖。

#### 25.6 shutdown与binary所有权

`LoadAicpuOp::FinalizeL1Pinned()` 是L1专用收口：

1. 调用者已在外部销毁graph并证明device quiescent。
2. 它可以释放异步bootstrap期的辅助buffer；失败时保留owner并返回，供显式重试。
3. 它不调用 `aclrtBinaryUnLoad` 或 `rtsBinaryUnload`。
4. 它清除本loader的host-side handle记录，使随后destructor也不会间接unload。
5. AICore register handle和TRB resident code同样按process pin处理。

因此 `shutdown()` 的“成功”表示context-owned stream/event/workspace/working state已按契约退役，不表示
CANN code binary被卸载。当前高层registry保留retired owner并不承诺同一host进程内重新初始化该device；
这是“binary/process pin”和“不猜测graph生命周期”的直接结果。

#### 25.7 实现文件对照

| 文件 | 已落地职责 |
| --- | --- |
| `python/pypto/jit/decorator.py` | `execution/runtime`、L1 binding、eager Out allocator、RunConfig默认与冲突校验 |
| `python/pypto/runtime/l1_jit.py` | strong device owner、late append、host admission、retryable shutdown |
| `python/pypto/runtime/l1.py` | manual/JIT共用的callable dedupe、参数校验/打包、per-callable prepare |
| `python/pypto/l1.py` | 公开 `pypto.l1.shutdown` |
| `runtime/.../l1_execution_state.*` | launch后仍保持append admission，close失败fail-closed |
| `runtime/.../device_runner_base.*` | 去掉L1 64-cap与HBG callable registration，prepare/launch/close事务 |
| `runtime/.../l1_aicpu_args.h` | TRB registration ABI v2 + callable hash |
| `runtime/src/a2a3/.../tensormap_and_ringbuffer/.../aicpu_executor.cpp` | L1 linked append-only code registry |
| `runtime/.../hbg_{launch_blob,aicpu_invocation,context_registry}.h` | self-contained package与context-only slot trust root |
| `runtime/.../load_aicpu_op.*` | `FinalizeL1Pinned` 与零BinaryUnLoad路径 |
| `tests/st/runtime/l1/test_l1_jit_aclgraph.py` | 公开API的TRB/HBG真机capture/replay验收 |

#### 25.8 已执行验证

无硬件结果：

- JIT decorator、compile extraction与新L1 facade：169 passed。
- L1 Python、taskQueue、init/close owner、source guard定向回归：62 passed。
- simpler C++ non-hardware：120/120 passed。
- A2/A3 onboard TRB/HBG的host、AICPU、AICore六类目标：全部编译通过。
- 受影响Python/C++文件的ruff、clang-format与 `git diff --check`：通过。

device0 A2/A3真机使用两个独立进程验证TRB和HBG，两者都覆盖：

1. 第一个JIT callable首次eager自动分配输出、隐式init/prepare/launch并验数。
2. 第一个callable执行完成后才发现第二个callable，完成late append和eager验数。
3. 外部sync后切到独立capture stream。
4. 图内顺序为 `torch.add -> L1 add -> L1 mul -> torch.add`。
5. 所有capture输出均在图外预分配。
6. 三组输入连续replay，每次同步后验数成功。
7. 先device quiescence，再 `graph.reset()`，最后可选 `shutdown()`。

本阶段不将A5/A5sim编译或上板作为完成门槛，不对它们做支持性声明。

#### 25.9 仍然明确存在的边界

- TRB registry是O(N)且无界增长；这是用户已明确接受的当前最大风险。
- 没有graph-aware release callback，所以不做LRU、token循环或binary unload。
- 同一owner的graph replay并发不支持；host mutex不能观测或阻止CANN内部replay。
- 首次调用必须是ordinary eager；PyPTO不query capture，原始native/CANN错误会被增补warmup指引。
- capture内省略out不属于v1契约，即使torch allocator在某个版本上偶然允许也不宣传为supported。
- `shutdown()` 不会sync，不能从一张graph的销毁推导为device级可关闭。

#### 25.10 A2/A3 HBG同构无依赖direct-AIV fast path

普通HBG必须恢复pristine scheduler image、启动AICPU participant、完成AICore握手与调度，再在结束时汇合并
销毁本轮runtime。该路径是通用DAG语义的正确实现，但对“一个AIV child + 静态数量同构work + 无任何依赖”
的pointwise子集，调度控制可能比child计算高三个数量级。

当前A2/A3实现因此增加一个严格、可回退的fast path：Host orchestration先生成真实graph；只有确认所有task
使用同一个AIV0 kernel、相同tensor/scalar arity、无fanin/predicate/task attrs/DFX，且静态block数一致时，
才把graph压平为immutable direct package。普通graph不满足条件时继续走既有AICPU scheduler，语义不变。

direct AIV entry启动当前平台的全部AIV block，并用确定性grid-stride消费逻辑work：

```text
for work_id = block_idx; work_id < work_count; work_id += block_num:
    invoke_same_child(task_args[work_id / logical_block_num],
                      work_id % logical_block_num)
```

所以物理48个AIV block既能执行47/48个work，也能执行50、96、97或更多work；不要求task数等于核数。
该严格子集无需跨核atomic cursor，避免引入不必要的claim/contention协议。task成本不均匀时可能出现tail
imbalance，未来若需要动态均衡，应设计和验证A2/A3自己的atomic协议。

package生命周期仍遵守HBG的tiling原则：`HbgGraphPlan`拥有canonical tensor/scalar/task snapshot；相同完整参数
语义可命中callable-local单条cache，但每次launch仍复制一份可写HostArgs；`aclrtLaunchKernelWithHostArgs`的placeholder把单pointer参数patch到CANN为eager task
或captured node保有的inline package。context只拥有per-lane mutable scratch和child code；package不引用Host
临时vector。binary/function handle按process pin，禁止`BinaryUnLoad`。

这个设计只借鉴A5 `fdwic-swimlane-deps`“AICore Scalar控制流可以承担task选择”的思想，没有复制其实现。
参考分支固定96 worker（32 AIC + 64 AIV），并依赖A5 Scalar/SIMT、cross-core atomic、claim tournament、
shared TensorMap和完成协议；这些ABI与拓扑不能在A3直接使用。本次实现和完成门槛仅覆盖A2/A3，不对A5或
A5sim作支持声明。

当前A3 device0已验证一个`@pl.jit` HBG task产生50个逻辑work、由48个AIV block完成，eager与
`direct L1 -> torch.mul` ACLGraph三次replay均验数通过；最终重建复跑的热replay（含successor和
stream sync）约61.4us。
完整调试证据、拒绝矩阵和所有权单测见实现过程记录10.69。

#### 25.11 HBG GraphPlan热调用cache与taskQueue dequeue

torch_npu profiler把taskQueue callback的全部Host工作都计入`Dequeue@pypto_*`。原实现每次callback都调用
Host orchestration、构建HBG graph、复制pristine region并生成direct package，即使warmup、后续eager和
capture使用完全相同的tensor地址、layout和scalar，也会重复约260ms的GraphPlan构建；这不是taskQueue
出队调度本身的固定开销。

当前实现给每个L1 HBG callable增加一个`HbgGraphPlanCache`，契约如下：

1. cache key覆盖tensor count、每个tensor的device address、buffer size、owner、offset、version、dtype、
   address space、shape、stride、extent、contiguous/manual-dependency标志，以及所有scalar原始bit pattern；
2. 先比较稳定hash，再逐字段比较完整语义快照，unused dimension与padding不参与identity；
3. cache只有一个entry。新地址、layout或scalar产生miss，重新执行Host build，并在成功后原子替换；不会按
   capture次数或地址数量无界积累Host graph；
4. callable function table、execution-slot binding和runtime配置在当前context内prepare后冻结；cache本身
   归callable/context所有，close时随`CallableState`释放；
5. cache命中只复用immutable canonical plan/direct package。普通HBG每次仍生成fresh writable launch blob，
   direct-AIV每次仍生成fresh `[prefix | inline package]` HostArgs；CANN继续为每个eager task或captured node
   持有自己的runtime-owned snapshot，working slot仍在每次执行/replay完整restore；
6. `plan_generation`标识package/restore identity，而非每次Host提交都必须递增的执行计数。同一个captured
   node replay本就重复同一代；只要每次restore完整执行，普通eager重复提交同一plan也遵循相同协议。

device0、A2/A3、Torch 2.12对目标trace的最终复跑结果：

| 指标 | 优化前 | 优化后 |
| --- | ---: | ---: |
| 4次launch dequeue均值 | 262716.89us | 65398.89us（包含一次miss） |
| 首次warmup/cache miss | 264644.73us | 261455.55us |
| 后续两次warmup/cache hit | 261144.46/261701.17us | 42.24/31.40us |
| capture launch/cache hit | 263377.18us | 66.36us |
| 三次热hit均值 | 262074.27us量级 | 46.67us |

热均值约缩短5629倍，capture dequeue约缩短3969倍；四次总和因仍保留一次必要build而缩短约4.02倍。
ACLGraph内`torch.add -> PyPTO HBG direct-AIV -> torch.relu`数值通过，8次replay通过。首次miss仍是约261ms，
这是当前动态HBG build成本；若要优化“每次都换tensor地址”的eager场景，下一阶段应把不含地址的结构模板
与per-call参数patch进一步拆开，不能直接忽略地址后复用当前package。
