# L1 算子与 ACLGraph

PyPTO L1 把编译程序表现为一次异步、类似 AscendC 的单算子调用。常规接口有意采用
Triton 风格：给普通 `@pl.jit` 函数增加 `execution="l1"`，随后直接用 torch NPU
tensor 调用它。

PyTorch 拥有 current device、caller stream 和外部 tensor storage；PyPTO 拥有内部
workspace、常驻 runtime 状态和一条隐藏 AICore stream。一次 launch 不同步 stream、
不查询 capture 状态、不 reset device，也不向用户暴露内部 AICPU/AICore fork/join。

当前 L1 只支持 A2/A3 onboard，并提供两种 runtime：

| Runtime | 装饰器取值 | 执行模型 |
| ------- | ---------- | -------- |
| TensorMap 与 ring buffer（TRB） | `"tensormap_and_ringbuffer"` | AICPU 在执行时构建并派发 task |
| Host-built graph（HBG） | `"host_build_graph"` | Host 构建自包含 graph package；每次调用先恢复再派发 |

## 定义并调用 L1 算子

```python
import pypto.language as pl


@pl.jit(execution="l1", runtime="host_build_graph")
def add(
    lhs: pl.Tensor[[64, 128], pl.FP32],
    rhs: pl.Tensor[[64, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs_tile = pl.load(lhs, [0, 0], [64, 128])
        rhs_tile = pl.load(rhs, [0, 0], [64, 128])
        pl.store(pl.add(lhs_tile, rhs_tile), [0, 0], out)
    return out


# 普通 eager 调用。PyPTO 会延迟完成编译、初始化和 prepare。若调用方省略纯
# 输出，PyTorch wrapper 会通过 torch.empty 在输入所在设备分配，因此调用形态与
# 普通 torch 算子一致。
result = add(lhs, rhs)
```

省略 `runtime` 时默认选择 `"tensormap_and_ringbuffer"`。runtime 属于 JIT cache key
和生成产物，不是 launch 时的动态开关。第一次 tensor 调用推断 device；该 device
必须已经是 torch_npu current device，PyPTO 不替调用方切换。

scalar 沿用现有 PyPTO 表达，例如 `pl.Scalar[pl.FP32]`，L1 不引入第二套语法。
不同调用可以改变 tensor address 和 scalar value；第一次成功入队后，shape、dtype、
stride 与参数布局固定。

每个 device 会延迟创建一个进程级 L1 owner。后续使用相同 platform、runtime 和
runtime configuration 的 JIT specialization 或函数动态追加到该 owner；公共接口没有
批量 prepare，也不暴露固定 callable 容量。runtime/configuration 冲突会在 launch 前
直接报错。

## Warmup、capture 与 replay

第一次调用必须是在 capture 外的普通 eager 调用，它会完成该算子所需的延迟编译、
初始化和 prepare。PyPTO 不查询 stream 是否正在 capture，因此未初始化算子的第一次
调用如果发生在 capture 内，会给出要求 warmup 的错误，而不会静默修改全局状态。

ACLGraph 不负责为算子分配输出。graph 引用的输出需要在 capture 前分配，并通过
`out=` 显式传入：

```python
import pypto
import torch
import torch_npu


# 在 capture 外 warmup 图中会出现的每个 specialization。
add(lhs, rhs, out=warmup_out)
torch_npu.npu.synchronize(device)

capture_stream = torch_npu.npu.Stream(device=device)
graph = torch_npu.npu.NPUGraph()
with torch_npu.npu.graph(graph, stream=capture_stream):
    torch.add(source, bias, out=pre_l1)
    add(pre_l1, rhs, out=add_out)
    torch.mul(add_out, 2, out=result)

for new_input in replay_inputs:
    source.copy_(new_input)
    graph.replay()
    capture_stream.synchronize()

# 完全可选。它不与某一张 graph 的销毁绑定；调用方必须先证明该 device 上所有
# L1 task 和仍可 replay 的 graph 都已经 quiescent。
torch_npu.npu.synchronize(device)
graph.reset()
pypto.l1.shutdown(device=device)
```

多纯输出的 eager 省略规则是“全部省略或全部提供”。capture 期间必须显式提供全部
输出，因为 allocator activity 应在 capture 的算子调用之外完成。graph 引用的输入、
输出以及 external storage 必须一直存活到 graph 销毁且相关 stream 真正完成。

默认 torch_npu adapter 通过 `RunOpApiV2` 进入 taskQueue，使用 `.stream(false)` 获取
raw stream，在 queued callback 执行前保留 C++ tensor handle，并把普通 caching-
allocator storage record 到 launch stream。这使 L1 真正成为队列中的一个有序 torch
算子，而不是 Python 侧绕过队列的 raw-stream 调用。external、`from_blob` 或 custom
allocator storage 不由 caching allocator 保活，必须由其 owner 保证更长生命周期。

## HBG graph package 生命周期

HBG image 是可变执行状态：scheduler queue、completion flag、task state 与 runtime
pointer 会在执行过程中被消费或改写。因此 L1 将它拆成：

- 一份自包含、pristine graph package，放在本次 launch 的 HostArgs blob 中；CANN
  像管理 AscendC inline tiling data 一样，把它随 launch task/captured node 一起快照
  并持有；
- 一份 context-owned、device 地址稳定的可写 execution slot；每次 eager execution 和
  graph replay 前，AICPU leader 都把 pristine shared-memory/runtime-arena image 恢复到
 该 slot。

所以每个 captured HBG node 都持有独立 graph package 与 callable-local function table；
即使两个 program 的第一个 kernel 都从 `func_id=0` 编号也不会冲突。这里不存在固定
resident callable table，PyPTO 也不会根据某个 host event 猜测 CANN 已经释放 captured
package。

生成的 HBG orchestration 可以使用 tensor address、shape、stride、dtype、scalar value
和静态 topology。若 orchestration 需要 Host 读取或写入借用 tensor 的内容，则会在构图
前 fail-closed 拒绝。

## TRB registry 与 binary 生命周期

TRB 需要 device-resident code-address registry。L1 遇到新 callable 时动态 append
immutable entry；entry 不 eviction、不循环复用、不覆盖，公共 API 也没有 64-callable
上限。这比复用 captured task 仍可能引用的地址更安全，但长期进程若生成无界
specialization，device/AICPU metadata 也可能无界增长；应用需要控制 specialization
数量并监控内存。

当前没有公开 runtime 契约保证：captured graph 仍引用 function handle 时，binary
unload 后 runtime 会替用户保活。因此 L1 路径在任何阶段（包括 `shutdown()`）都不调用
`aclrtBinaryUnLoad` 或 `rtsBinaryUnload`。binary code 与 function handle 会 pin 到进程
结束；这是明确的生命周期策略，不应被“修复”为 eager unload。

## 可选 shutdown 与 owner

`pypto.l1.shutdown(device=...)` 完全可选：

- 不调用时，L1 owner 安静地 pin 到进程结束；
- 它从不做同步，只能在该 device 上所有 L1 task 和所有 graph owner 都 quiescent 后调用；
- 重复调用幂等；
- teardown 失败时保留 owner，可以重试；
- Python GC 与 `atexit` 不做 runtime teardown，也不卸载 binary。

销毁一张 ACLGraph 不代表其他 graph 不存在，因此不会自动触发 device owner shutdown。

## 支持边界

- 只支持 A2/A3 onboard；A5 与 simulator 不属于当前已验证范围。
- shape、dtype、stride 与参数布局静态；tensor address 与 scalar value 可以变化。
- eager 支持省略纯输出并由 PyTorch 分配；capture 要求显式传入预分配输出。
- 仅 inference：不支持 autograd、distributed/`CommCtx`、SDMA 或 DFX。
- 不支持 eager 并发、eager 与 replay 重叠或 graph replay 并发。Host 同时调用会尽力
  fail-fast；外部发起的 graph replay 不一定能被 PyPTO 观察到，调用方仍必须串行化。
- workspace 保持由 PyPTO 内部管理，并仅在上述串行执行契约下共享。

底层 `pypto_init`/`L1Context` 接口保留给实现测试和高级 bring-up，但不是普通用户入口。
维护中的公共 API TRB/HBG capture/replay 用例见
`tests/st/runtime/l1/test_l1_jit_aclgraph.py`。
