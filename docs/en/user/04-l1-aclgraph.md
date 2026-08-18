# L1 Operators and ACLGraph

PyPTO L1 presents a compiled `@pl.jit` or `@pl.program` as one asynchronous,
AscendC-like operator call. PyTorch owns the current device, input/output
storage, and caller stream. PyPTO owns its internal workspace, persistent
runtime state, and one hidden AICore stream, but does not synchronize streams,
inspect capture state, reset the device, or expose that hidden branch.

L1 supports both Simpler runtimes on onboard targets:

| Runtime | Compile value | Execution model |
| ------- | ------------- | --------------- |
| TensorMap and ring buffer (TRB) | `"tensormap_and_ringbuffer"` | AICPU builds and dispatches tasks at execution time |
| Host-built graph (HBG) | `"host_build_graph"` | Host builds a pristine task graph package; each invocation restores it before dispatch |

## Compile and initialize

Select the runtime when compiling. It is part of the generated artifact and
the JIT cache key, not a launch-time switch:

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
        runtime="host_build_graph",  # or "tensormap_and_ringbuffer"
    )
)

ctx = pypto_init(programs=[compiled], device=device)
op = ctx.operator(compiled)
```

For a program object, use
`ir.compile(program, platform="a2a3", runtime="host_build_graph")`. A
`DistributedConfig.runtime` value is inherited when no explicit runtime is
given; two explicit sources must agree. Every program declared in one
`L1Context` must use the same onboard platform and runtime.

Default JIT output directories are atomically unique, so compiling the same
specialization for TRB and HBG cannot alias files even within one clock tick.
If `RunConfig.save_kernels_dir` is explicit, do not reuse that directory for a
different runtime, or for different cache keys of the same `JITFunction`;
PyPTO rejects those conflicts before they can overwrite the first lazy
artifact. Historical same-runtime rebuilding by a distinct compiler/JIT owner
still treats an explicit directory as caller-owned.

`device` is mandatory. It must equal the current torch_npu device; PyPTO never
changes the caller's device.

## Warm up, capture, and replay

Prepare and warm up outside capture, then synchronize explicitly before
capturing:

```python
graph = None
try:
    ctx.prepare()
    op.warmup(x, weight, out=y)
    torch_npu.npu.synchronize(device)  # caller-owned warmup boundary

    capture_stream = torch_npu.npu.Stream(device=device)
    graph = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(graph, stream=capture_stream):
        torch.add(prefix, 1, out=x)
        op(x, weight, out=y)
        torch.mul(y, 2, out=result)

    graph.replay()
    capture_stream.synchronize()
finally:
    # No graph may replay after this point.
    torch_npu.npu.synchronize(device)
    if graph is not None:
        graph.reset()
    ctx.close()
```

`prepare()` and `warmup()` report successful enqueue, not device completion.
The external synchronize before capture is therefore required. Calling an
unprepared operator in ordinary eager mode performs a convenience prepare, but
the first call must never occur inside capture: PyPTO deliberately does not ask
whether the current stream is being captured.

The default taskQueue adapter obtains the current raw stream without draining
the queue and records ordinary torch_npu caching-allocator storage on that
stream. `L1Config(use_task_queue=False)` is only a bring-up/debug path; obtaining
the Python raw stream can drain an enabled taskQueue.

## HBG graph-package ownership

An HBG image is mutable execution state, not an immutable executable: scheduler
queues, completion flags, task state, and runtime pointers are consumed or
rewritten while the operator runs. PyPTO therefore uses two objects:

- a pristine, immutable graph package embedded in the launch's mutable HostArgs
  blob; CANN copies it with that launch task/captured node, like AscendC inline
  tiling data;
- one context-owned mutable execution slot whose address stays fixed and whose
  shared-memory and runtime-arena images are restored by the AICPU leader before
  every eager execution and every graph replay.

Two captured HBG nodes consequently retain independent graph packages and
callable-local function tables even when both compiled programs number their
first kernel as `func_id=0`. PyPTO does not reuse or infer the lifetime of a
captured package from a host-side event.

HBG L1 borrows external device tensors and never stages their contents through
host memory. The host graph builder may use tensor addresses, shapes, strides,
dtypes, scalar values, and static topology. A generated orchestration that
would read or write tensor contents on the host is rejected fail-closed before
its graph can be used.

## Lifetime and supported boundary

Keep the context and all graph-referenced tensors/storage alive until the graph
can no longer replay. For `from_blob` or custom/external allocator storage,
`recordStream` may not extend its lifetime; its external owner must remain live
through graph destruction and actual stream completion. `close()` performs no
implicit synchronization and is intentionally not a context manager. If native
teardown fails, ownership remains with the context so `close()` can be retried
after external quiescence.

The first L1 version has these boundaries:

- onboard `a2a3` and `a5` only; simulator execution is unsupported;
- static shape and dtype; shape, dtype, and stride are fixed by the first
  successful enqueue, while tensor addresses and scalar values may change;
- explicit caller-provided outputs via `out=`;
- inference only: no autograd, distributed/`CommCtx`, SDMA, or DFX;
- one non-concurrent L1 context per device, with no concurrent eager calls or
  graph replays; PyPTO currently occupies the configured AICore set;
- workspace remains internal to PyPTO and is shared only under that serial
  execution contract.

See `tests/st/runtime/l1/test_l1_aclgraph.py` for the maintained end-to-end
TRB/HBG capture and replay examples.
