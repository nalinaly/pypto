# PyPTO Triton 风格 L1 JIT 调用接口设计

<!-- markdownlint-disable MD048 MD060 -->

> 状态：已实现，A2/A3 TRB/HBG eager + ACLGraph 已上板验证
> 适用范围：A2/A3 onboard、PyTorch/torch_npu、TRB 与 HBG、eager 与 ACLGraph
> 关联文档：`PyPTO_L1与ACLGraph完整设计文档.md`、`PyPTO_L1与ACLGraph实现过程记录.md`、`pypto_l1_aclgraph_implementation_plan.md`
> 核心目标：把已经跑通的 L1 底层能力包装成像 Triton JIT kernel 一样自然的 Python API，同时保留现有 taskQueue、失败所有权和 ACLGraph 生命周期边界。

---

## 1. 结论先行

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

## 2. 为什么当前 API 不适合作为最终用户接口

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

## 3. 已确认的设计决定

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

## 4. 公共 Python API

### 4.1 decorator

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

### 4.2 eager：显式输出

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

### 4.3 eager：隐式输出分配

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

### 4.4 scalar

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

### 4.5 ACLGraph

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

### 4.6 可选 shutdown

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

## 5. 对用户隐藏的对象模型

### 5.1 总体结构

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

### 5.2 registry key

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

### 5.3 callable identity

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

### 5.4 specialization identity

JIT specialization key继续由现有 `CacheKey` 决定，包括 source、shape、dtype、layout、dynamic dim、scalar specialization、platform、runtime、pass strategy 和 memory planner。

L1 在它之上再绑定：

~~~text
L1SpecializationKey =
    (JIT CacheKey, CallableContentKey, device_id, runtime_config_fingerprint)
~~~

tensor stride 不得只做“>0”检查。首个成功 enqueue 后绑定 shape/dtype/stride/layout metadata；后续调用必须一致。失败的首次 enqueue 不得提前提交 layout binding。

---

## 6. 生命周期与状态机

### 6.1 device owner

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

### 6.2 callable

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

### 6.3 append-after-warm

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

## 7. 第一次 eager 调用的完整流程

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

## 8. taskQueue 与 tensor lifetime

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

## 9. HBG：把 graph 当作 CANN 管理的 per-task tiling package

### 9.1 生命周期分层

HBG 需要明确区分五种对象，但这些层不应暴露给 Python 用户：

| 层 | 内容 | owner | 生命周期 |
| --- | --- | --- | --- |
| `GraphPlan` | host build 的 canonical pristine graph | PyPTO host builder | 构建本次 invocation |
| serialized launch blob | header、regions、identity、pristine payload | 调用栈临时对象 | 到 `WithHostArgs` 接管 |
| runtime-owned HostArgs | CANN 复制后的 task args 和 inline payload | CANN task/captured node | task 完成或 graph 销毁 |
| working execution slot | mutable SM、runtime arena、heap、Runtime/KernelArgs | PyPTO device owner | hidden context lifetime |
| code resources | AICPU entry、AICore binary/func handle、host orch SO | PyPTO pinned code owner | 至少覆盖所有 task/graph；binary到进程退出 |

关键边界：

> `aclrtLaunchKernelWithHostArgs` 管理 launch argument bytes，不等价于它自动管理这些 bytes 内所有 device address 所引用的 binary、workspace 或 mutable slot。

所以 HBG 可以把 graph/tiling package 交给 CANN，却仍必须由 PyPTO pin code 和 working slot。

### 9.2 为什么 package 必须 pristine

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

### 9.3 self-contained package

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

### 9.4 去掉固定 HBG callable registry

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

### 9.5 CANN placeholder

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

### 9.6 working slot

首版每个 device owner只保留一个 HBG mutable working slot：

- 符合 PyPTO 当前独占所有 AICore、不能并发的事实；
- workspace仍由 PyPTO内部管理；
- 每个 captured node有独立 pristine source package；
- 多个 node可共享同一 destination slot，但必须由外部 stream/graph时序保证不重叠。

这不是 graph package ownership的简化：source按 node独立，destination按 context共享。

未来支持并发时再引入 slot pool和 replay-aware lease；首版不提前实现。

### 9.7 HBG 调用时序

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

## 10. TRB：动态 append-only code registry

### 10.1 为什么 CANN HostArgs 不能替代 TRB registry

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

### 10.2 callable id 与 content key

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

### 10.3 AICPU 数据结构

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

### 10.4 注册 ABI

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

### 10.5 不复用、不覆盖

明确禁止：

- registry满后从 slot 0重新覆盖；
- LRU驱逐；
- 仅凭 Python kernel对象析构卸载；
- 仅凭一张 graph销毁释放；
- 用 event query推断“所有 graph未来都不会 replay”；
- 新注册同 token不同内容。

CANN没有向 PyPTO提供“所有持有该 callable token的 graph/task都已销毁”的通用回调。因此循环复用会把旧 graph静默指向新 code，是比增长更危险的错误。

### 10.6 动态增长风险

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

## 11. Binary 与 code resource 所有权

### 11.1 四类资源必须分开

| 资源 | 典型建立方式 | 新 L1 JIT 生命周期 |
| --- | --- | --- |
| simpler AICPU runtime binary | `aclrtBinaryLoadFromData` | process pinned，绝不 BinaryUnLoad |
| AICore binary/function handle | register binary/kernel | process pinned，不做 unregister/reuse |
| TRB callable orchestration SO | AICPU内 `dlopen` | append registry entry，至少到 shutdown；无安全回收则到进程退出 |
| HBG host orchestration SO | host `dlopen`，用于 build graph | callable owner pin；shutdown后仅在外部quiescence成立时释放 |

### 11.2 强制规则

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

### 11.3 与现有 `LoadAicpuOp::Finalize` 的关系

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

### 11.4 shutdown 的精确定义

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

## 12. 并发边界

### 12.1 能检测的并发

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

### 12.2 无法检测的并发

ACLGraph replay绕过 Python和PyPTO host。以下情况首版无法完整检测：

- 两张 graph在不同 stream并发 replay；
- graph replay与 eager L1 launch重叠；
- graph replay期间首次注册新 callable；
- graph销毁前调用 shutdown但调用方错误地声称已 quiescent。

这些路径明确 unsupported。因为 PyPTO当前占用全部 AICore并共享 workspace/working slot，未检测到的重叠可能导致数据破坏或 runtime错误。

文档不能声称 host lock提供了 device并发安全。

### 12.3 后续方向

完整并发支持需要至少一项：

- ACLGraph/runtime外部资源 retain/release hook；
- graph-aware execution-slot pool；
- graph replay admission token；
- host_build_graph统一编排多个算子；
- device-side generation/lease协议。

不在本次 Triton风格API改造中提前实现。

---

## 13. 错误模型

### 13.1 错误类型

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

### 13.2 pre-enqueue failure

以下失败不得改变 persistent state：

- 参数个数/方向错误；
- device/dtype/shape/stride不匹配；
- unsupported scalar；
- runtime conflict；
- callable artifact/hash校验失败；
- HBG package size/region/hash错误；
- registry byte budget不足。

使用临时候选对象完成全部验证，最后一步才 publish/commit。

### 13.3 enqueue后失败

taskQueue callback或native launch部分成功后失败时：

- 不回收可能被device引用的args/code/workspace；
- owner进入 failed-retained/poisoned状态；
- AICore-first launch若AICPU enqueue失败，继续使用已经设计的 host failure CANCEL + hidden done join错误闭包；
- 后续调用拒绝，允许可选 shutdown重试；
- 不调用 BinaryUnLoad。

---

## 14. 与当前低层 API 的关系

### 14.1 保留但降级为 advanced

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

### 14.2 不允许两套 owner独立存在

如果 advanced API和JIT API同时作用于同一 device，必须共享同一 `DeviceL1Owner` 或明确互斥。不能各自建立 hidden stream/workspace/runtime：

~~~text
JIT API owner
      X  forbidden parallel ownership
manual L1Context owner
~~~

首版最简单规则：若 device已有任一 owner，另一种入口报 conflict。

---

## 15. 与 pto2 历史实现的对比

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

## 16. 具体代码改造范围

以下是建议的实现文件和职责，行号以当前主分支附近符号为准，实施时按symbol定位。

### 16.1 PyPTO Python

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

### 16.2 simpler host/native

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

### 16.3 HBG

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

### 16.4 bindings、stubs与文档

- `runtime/python/bindings/task_interface.cpp`：新native entry。
- `runtime/python/simpler/task_interface.py`：新worker方法、异常和mode保护。
- PyPTO对应 `.pyi`：decorator参数和 `shutdown`。
- 正式用户文档最终应在 `docs/en` 与 `docs/zh` 同步；本文件是实施设计参考。

---

## 17. 实现顺序

### Phase A：公共 API 骨架，不改变native协议

1. 给 `@pl.jit` 增加 `execution/runtime` metadata。
2. 新建隐藏 `L1JITRegistry`，暂时适配现有manual context。
3. 实现direct call、eager output allocation、scalar pack、taskQueue。
4. 加first-capture warmup错误翻译和shutdown空壳/state admission。
5. Python无硬件UT通过后再改native。

目的：先冻结用户调用形态，避免native重构期间API来回变化。

### Phase B：HBG package self-contained

1. 将function table正式纳入package hash。
2. 修改HBG view/validator，不读取resident callable registry。
3. 移除HBG callable register task和fixed array。
4. 保持execution slot context-owned。
5. 验证旧captured node在新增其他callable后仍replay同一code。

HBG先做，因为它可以直接彻底消除64上限，而不是引入动态device registry。

### Phase C：TRB dynamic registry

1. 引入v2 registration identity ABI。
2. AICPU linked append registry。
3. host content dedupe和单调id。
4. 移除Python/native 64检查。
5. 故障注入growth allocation/dlopen/symbol失败。
6. 压测连续specialization并记录资源增长。

### Phase D：pinned binary owner与shutdown

1. 将新L1 binary owner从legacy Finalize路径分离。
2. 所有init/prepare失败都转移到strong retained owner。
3. 实现device级admission close和retryable retirement。
4. 增加“新路径unload调用次数必须为0”的测试。
5. GC/atexit静默策略。

### Phase E：A2/A3 ACLGraph验收

1. TRB golden path；
2. HBG golden path；
3. torch predecessor -> L1 -> torch successor；
4. 多次replay、多个callable、同一graph两个L1 node；
5. 新callable append后旧graph replay；
6. taskQueue和tensor lifetime压力；
7. shutdown前置条件和不自动close。

---

## 18. 测试设计

### 18.1 Python无硬件UT

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

### 18.2 simpler C++无硬件UT

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

### 18.3 A2/A3 onboard ST

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

### 18.4 并发负面测试

- 两个Python线程同时首次调用：一个成功，另一个明确 `L1ConcurrencyError`，不得hang。
- 已知不同stream且old tail未完成：拒绝。
- shutdown与调用重叠：调用拒绝。
- graph并发replay不能可靠host检测：测试标为unsupported contract，不伪造“已保护”结论。

---

## 19. 兼容与迁移

### 19.1 用户代码迁移

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

### 19.2 ABI兼容

- ChipCallable既有L2/L3 wire layout不得因L1 token改变。
- L1 registration/invocation使用独立ABI major。
- HBG launch blob/context registry显式bump ABI。
- 新AICPU runtime与host必须成对部署。
- legacy 64-slot entry可在过渡期保留，但新JIT不得生成它。
- L2/L3 runtime不引用新L1 dynamic registry，资源和行为保持。

---

## 20. 可观测性

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

## 21. 明确拒绝的替代方案

### 21.1 自动创建并销毁 context

拒绝：

~~~python
with pypto.l1(...):
    kernel(...)
~~~

graph可超出词法scope继续replay，RAII/context manager无法证明device引用结束，容易过早close。

### 21.2 每个 JITFunction一个native context

拒绝。多个kernel会重复workspace/stream/event/runtime，并违反单device owner。

### 21.3 graph销毁自动shutdown

拒绝。一张graph销毁不代表同device其他graph不存在。

### 21.4 registry循环复用

拒绝。没有CANN graph引用计数时，旧captured token可能在未来replay；覆盖会静默执行错误code。

### 21.5 依赖 BinaryUnLoad后的runtime保活

拒绝。当前没有公开契约保证captured graph仍引用funcHandle时，BinaryUnLoad后runtime替用户继续保活binary。

### 21.6 query capture并走双分支

拒绝。L1算子应capture-transparent；首次capture失败通过prepared状态翻译，不主动查询。

### 21.7 rtStreamAddToModel

拒绝。它让AICPU orchestration越过单算子边界提前执行。未来跨算子性能优化由host_build_graph完成。

### 21.8 内部stream sync

拒绝。正常launch、prepare和capture路径都不做stream/device sync。shutdown也以保守pin替代不安全的同步清理。

---

## 22. 风险清单

### R1：TRB registry无界增长

级别：当前最大已知工程风险。
接受理由：没有graph-aware安全驱逐协议；稳定token优先于内存回收。
当前缓解：content dedupe、稳定linked node、失败前完整构造、id不复用。
待补缓解：byte accounting、软阈值、ResourceExhausted分类和长时压测。
最终解决：CANN提供外部资源retain/release，或PyPTO获得graph生命周期通知。

### R2：CANN HostArgs snapshot语义

风险：HBG依赖captured node持有完整变长args和placeholder内联payload。
缓解：A2/A3上板验证同graph多次replay、host buffer销毁/复用后仍正确；严格验证args size和placeholder offset。
失败策略：capability fail closed，不退回显式launch时H2D。

### R3：shared HBG working slot

风险：两张graph并发replay会踩同一mutable state。
缓解：v1明确不支持并发；能检测的host并发报错。
最终解决：execution-slot pool + graph-aware lease。

### R4：进程级binary pin

风险：长进程code资源不释放。
接受理由：正确性优先，且用户明确禁止BinaryUnLoad。
缓解：dedupe、统计、文档提示；资源极限时fail-fast。
不可使用的缓解：在graph可能存活时unload。

### R5：首次capture错误翻译依赖runtime失败

风险：不同CANN版本返回码可能变化。
缓解：翻译条件同时要求 `state is not prepared`；保留原始error code/message；上板版本矩阵。
不采用：capture query。

### R6：隐式output allocation被误用于capture

风险：wrapper不查capture，用户省略out。
缓解：文档和示例只展示capture显式out；若torch_npu明确提供普通op一致的无副作用capture-safe allocator契约，可后续放宽。

### R7：host锁无法保护replay并发

风险：用户误以为“有mutex就安全”。
缓解：文档、错误信息和测试明确host lock边界；不宣传完整并发支持。

---

## 23. 完成标准

只有同时满足以下条件，才可以将Triton风格L1 API标为supported：

### 公共体验

- 用户只写 `@pl.jit(execution="l1", runtime=...)` 和 `kernel(...)`。
- ordinary eager不要求 `init/prepare/warmup/context/close`。
- scalar只使用现有 `pl.Scalar[...]`。
- eager支持torch allocator返回式输出。
- capture示例只使用预分配输出。
- 可选 `shutdown` 不调用也不会产生错误。

### runtime正确性

- 默认taskQueue路径使用 `stream(false)`、`RunOpApiV2`、Tensor lease和allocator `recordStream`。
- AICPU使用caller stream；内部AICore stream不外露。
- 单算子内部完整fork/join。
- launch不sync、不reset、不query capture、不attach model。
- HBG package self-contained且每次replay恢复pristine state。
- TRB registry动态append，旧token永不失效。
- 公共64-callable限制被移除。
- 新路径任何成功/失败/shutdown/析构路径的 BinaryUnLoad调用次数都为0。

### 兼容性

- 非L1 `@pl.jit` 行为不变。
- L2/L3 ChipCallable wire和runtime回归通过。
- advanced manual L1 tests继续通过或有明确迁移说明。
- 仅A2/A3作为本阶段硬件门槛。

### ACLGraph实证

- TRB和HBG都完成图外warmup、换stream capture、torch前后继、多次replay验数。
- 同一graph包含两个不同L1 callable。
- 两个callable可各自使用 `func_id=0`。
- 新callable append后旧graph继续正确replay。
- host临时HBG serialization buffer释放/复用后，captured graph仍正确。
- 默认allocator下删除Python临时引用并施加分配压力仍正确。

---

## 24. 最终原则

这次API改造不改变L1的本质边界：

> 对用户，它是一个普通的 `@pl.jit` kernel；对torch_npu，它是当前stream上的一个普通异步算子；对CANN，HBG graph是每个task/captured node拥有的变长tiling package；对PyPTO，workspace、working slot和code resources是必须独立pin住的被引用资源。

由此得到三条不可打破的所有权规则：

1. CANN owns invocation bytes，不自动推导为owns referenced binary。
2. PyPTO owns code/workspace/working state，不感知某一张ACLGraph的capture与销毁。
3. 没有可靠release信号时宁可append/pin，也不复用token、不卸载binary、不猜测graph已结束。

该方案把当前“正确但难用”的manual L1控制面保留下来，同时在其上建立一个真正可供PyTorch用户直接调用的Triton风格产品接口。

---

## 25. 实际落地结果

本节记录实现后的事实；若前文某个“建议”与本节冲突，以本节和当前源码为准。

### 25.1 用户实际看到的API

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

### 25.2 隐藏device owner与append流程

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

### 25.3 manual L1与JIT共用的参数契约

`python/pypto/runtime/l1.py` 仍是唯一参数打包与底层强校验入口：

- tensor强制NPU/current device、静态shape、dtype、正的uint32 stride、非autograd。
- 首个成功enqueue绑定shape/dtype/stride；后续布局改变在enqueue前拒绝。
- scalar使用旧有 `pl.Scalar[...]` annotation，按声明dtype做bit-exact low-byte pack，包括FP16、BF16、FP32、
  FP64、整数和bool。
- `L1Context.add_program()` 允许首次launch之后append新callable；`prepared` 改为per-callable状态。
- canonical `ChipCallable` 内容相同时复用已有state，不重复注册。

manual `pypto_init/context/operator` 保留为advanced/debug控制面，但不再是普通用户文档的首选入口。

### 25.4 HBG落地的生命周期

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

### 25.5 TRB落地的code registry

TRB的AICPU进程保留legacy L2/L3 `orch_so_table_[64]`，但borrowed L1完全不读写该表。L1使用
append-only `L1OrchSoNode` 链：每个node持有id、hash、`dlopen` handle、entry/config/bind function、
callable-local kernel地址快照和发布状态。

prepare是唯一增长点，launch仅遍历已发布node并读它。duplicate id只有在hash、kernel count和全部
kernel binding相同时幂等，否则直接conflict。注册失败不会把半成品node连入全局链。

当前没有驱逐、循环复用或公开callable count限制。链表lookup与无界资源增长是明确接受的风险，
不是已经解决的问题。实现优先保证旧graph的id、code handle和地址永不被新specialization覆盖。

### 25.6 shutdown与binary所有权

`LoadAicpuOp::FinalizeL1Pinned()` 是L1专用收口：

1. 调用者已在外部销毁graph并证明device quiescent。
2. 它可以释放异步bootstrap期的辅助buffer；失败时保留owner并返回，供显式重试。
3. 它不调用 `aclrtBinaryUnLoad` 或 `rtsBinaryUnload`。
4. 它清除本loader的host-side handle记录，使随后destructor也不会间接unload。
5. AICore register handle和TRB resident code同样按process pin处理。

因此 `shutdown()` 的“成功”表示context-owned stream/event/workspace/working state已按契约退役，不表示
CANN code binary被卸载。当前高层registry保留retired owner并不承诺同一host进程内重新初始化该device；
这是“binary/process pin”和“不猜测graph生命周期”的直接结果。

### 25.7 实现文件对照

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

### 25.8 已执行验证

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

### 25.9 仍然明确存在的边界

- TRB registry是O(N)且无界增长；这是用户已明确接受的当前最大风险。
- 没有graph-aware release callback，所以不做LRU、token循环或binary unload。
- 同一owner的graph replay并发不支持；host mutex不能观测或阻止CANN内部replay。
- 首次调用必须是ordinary eager；PyPTO不query capture，原始native/CANN错误会被增补warmup指引。
- capture内省略out不属于v1契约，即使torch allocator在某个版本上偶然允许也不宣传为supported。
- `shutdown()` 不会sync，不能从一张graph的销毁推导为device级可关闭。
