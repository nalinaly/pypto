# L1 算子与 ACLGraph

PyPTO L1 把一个编译后的 `@pl.jit` 或 `@pl.program` 表现成一次异步、类似
AscendC 的单算子调用。PyTorch 拥有 current device、输入输出 storage 和 caller
stream；PyPTO 拥有内部 workspace、常驻 runtime 状态和一条隐藏 AICore stream，
但不会同步 stream、查询 capture 状态、reset device，也不会向用户暴露隐藏分支。

L1 在 onboard 平台支持两种 Simpler runtime：

| Runtime | 编译取值 | 执行模型 |
| ------- | -------- | -------- |
| TensorMap 与 ring buffer（TRB） | `"tensormap_and_ringbuffer"` | AICPU 在执行时构建并派发 task |
| Host-built graph（HBG） | `"host_build_graph"` | Host 构建 pristine task graph package；每次调用先恢复再派发 |

## 编译与初始化

编译时选择 runtime。它属于生成产物和 JIT cache key，不是 launch 时切换项：

```python
import torch
import torch_npu

from pypto.l1 import pypto_init
from pypto.runtime import RunConfig

device = 1
torch_npu.npu.set_device(device)

compiled = my_kernel.compile(
    config=RunConfig(
        platform="a2a3",
        device_id=device,
        runtime="host_build_graph",  # 或 "tensormap_and_ringbuffer"
    )
)

ctx = pypto_init(programs=[compiled], device=device)
op = ctx.operator(compiled)
```

若使用 program 对象，则调用
`ir.compile(program, platform="a2a3", runtime="host_build_graph")`。没有显式
runtime 时会继承 `DistributedConfig.runtime`；两个显式来源必须一致。同一个
`L1Context` 声明的所有 program 必须使用相同 onboard platform 与 runtime。

默认JIT输出目录通过原子方式保证唯一，所以同一时刻分别编译TRB与HBG也不会共用
产物目录。若显式设置`RunConfig.save_kernels_dir`，不得让不同runtime复用同一路径，
也不得让同一个`JITFunction`的不同cache key复用；PyPTO会在覆盖第一份延迟加载产物
之前直接报错。不同编译器/JIT owner之间的同runtime历史重编译仍把显式目录视为
caller-owned。

`device` 是必填参数，并且必须等于当前 torch_npu device；PyPTO 不替调用方切换
device。

## Warmup、capture 与 replay

在 capture 之外 prepare、warmup，再由调用方显式同步后开始 capture：

```python
graph = None
try:
    ctx.prepare()
    op.warmup(x, weight, out=y)
    torch_npu.npu.synchronize(device)  # 调用方拥有的 warmup 边界

    capture_stream = torch_npu.npu.Stream(device=device)
    graph = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(graph, stream=capture_stream):
        torch.add(prefix, 1, out=x)
        op(x, weight, out=y)
        torch.mul(y, 2, out=result)

    graph.replay()
    capture_stream.synchronize()
finally:
    # 此后不得再有 graph replay。
    torch_npu.npu.synchronize(device)
    if graph is not None:
        graph.reset()
    ctx.close()
```

`prepare()` 与 `warmup()` 表示成功入队，不表示 device 已完成，因此 capture 前的
外部同步是必需的。普通 eager 模式下，未 prepare 的第一次调用会便利地自动
prepare；但第一次调用绝不能发生在 capture 内：PyPTO 有意不查询当前 stream 是否
正在 capture。

默认 taskQueue adapter 获取 current raw stream 时不会排空队列，并在该 stream 上
记录普通 torch_npu caching-allocator storage。`L1Config(use_task_queue=False)` 只用于
bring-up/debug；通过 Python 获取 raw stream 会在 taskQueue 已开启时排空 host queue。

## HBG graph package 所有权

HBG image 是可变执行状态，不是 immutable executable：scheduler queue、completion
flag、task state 与 runtime pointer 都会在执行中被消费或改写。因此 PyPTO 使用两类
对象：

- 一份 pristine、immutable graph package，内嵌在本次 launch 的 mutable HostArgs
  blob 中；CANN 像管理 AscendC inline tiling data 一样，把它随本次 launch task/
  captured node 一起复制；
- 一份 context-owned mutable execution slot，其地址保持稳定；AICPU leader 在每次
  eager execution 和每次 graph replay 前恢复其中的 shared-memory 与 runtime-arena
  image。

所以两个 captured HBG node 会分别保留自己的 graph package 和 callable-local
function table，即使两个 CompiledProgram 的第一个 kernel 都从 `func_id=0` 开始编号。
PyPTO 不会根据 host event 推测 captured package 已经失效或可以复用。

HBG L1 借用 external device tensor，不经 host staging tensor contents。Host graph
builder 可以使用 tensor address、shape、stride、dtype、scalar value 与静态 topology；
如果生成的 orchestration 需要在 host 读取或写入 tensor contents，则在 graph 可用前
fail-closed 拒绝。

## 生命周期与支持边界

在 graph 不再可能 replay 之前，必须保持 context 与 graph 引用的全部 tensor/storage
存活。对于 `from_blob` 或 custom/external allocator storage，`recordStream` 未必能够
延长生命周期；其外部 owner 必须存活到 graph 销毁且实际 stream 完成。`close()` 不做
隐式同步，因此有意不提供 context manager。Native teardown 失败时，context 继续持有
所有权；外部 quiescence 后可以重试 `close()`。

首版 L1 的边界如下：

- 只支持 onboard `a2a3` 与 `a5`，不支持 simulator 执行；
- 静态 shape 与 dtype；第一次成功入队后固定 shape、dtype 与 stride，但 tensor address
  与 scalar value 可以变化；
- 输出由调用方通过 `out=` 显式传入；
- 仅 inference：不支持 autograd、distributed/`CommCtx`、SDMA 或 DFX；
- 每个 device 只有一个非并发 L1 context，不支持 eager call 或 graph replay 并发；
  PyPTO 当前会占用配置的 AICore 集合；
- workspace 继续由 PyPTO 内部管理，只在上述串行执行契约下共享。

维护中的 TRB/HBG capture/replay 端到端例子见
`tests/st/runtime/l1/test_l1_aclgraph.py`。
