# L1 Operators and ACLGraph

PyPTO L1 exposes a compiled program as one asynchronous, AscendC-like operator
call. The normal interface intentionally looks like Triton: annotate a regular
`@pl.jit` function with `execution="l1"`, then call the function with torch NPU
tensors.

PyTorch owns the current device, caller stream, and external tensor storage.
PyPTO owns its internal workspace, persistent runtime state, and one hidden
AICore stream. A launch never synchronizes streams, queries capture state,
resets the device, or exposes the hidden AICPU/AICore fork-and-join.

The supported L1 target is A2/A3 onboard. Two runtimes are available:

| Runtime | Decorator value | Execution model |
| ------- | --------------- | --------------- |
| TensorMap and ring buffer (TRB) | `"tensormap_and_ringbuffer"` | AICPU builds and dispatches tasks at execution time |
| Host-built graph (HBG) | `"host_build_graph"` | Host builds a self-contained graph package; every invocation restores it before dispatch |

## Define and call an L1 operator

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


# Ordinary eager use. PyPTO compiles, initializes, and prepares lazily. A pure
# output omitted by the caller is allocated through torch.empty on the input
# device, so this has the same call shape as a normal torch operator.
result = add(lhs, rhs)
```

Omitting `runtime` selects `"tensormap_and_ringbuffer"`. Runtime selection is
part of the JIT cache key and generated artifact; it is not a launch-time
switch. The first tensor call infers the device, which must already be the
current torch_npu device. PyPTO never changes it.

Use existing PyPTO scalar annotations such as `pl.Scalar[pl.FP32]`; L1 does not
introduce a second scalar syntax. Tensor addresses and scalar values may change
between calls, while shape, dtype, stride, and argument layout are fixed by the
first successful enqueue.

One process-owned L1 owner is created lazily per device. Later JIT
specializations and functions using the same platform, runtime, and runtime
configuration are appended to it; no public batch-prepare or fixed callable
capacity is exposed. A conflicting runtime/configuration fails before launch.

## Warm up, capture, and replay

The first call must be an ordinary eager call outside capture. It performs the
lazy compilation/initialization/prepare needed by the operator. PyPTO does not
query whether a stream is being captured, so an uninitialized first call made
inside capture fails with a warm-up diagnostic rather than silently mutating
global state.

ACLGraph does not allocate operator outputs. Allocate graph-bound outputs before
capture and pass them with `out=`:

```python
import pypto
import torch
import torch_npu


# Warm up every specialization that will appear in the graph.
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

# Optional retirement. It is not tied to destruction of one graph: the caller
# must first prove that every L1 task and every graph that can replay on this
# device is quiescent.
torch_npu.npu.synchronize(device)
graph.reset()
pypto.l1.shutdown(device=device)
```

Eager output omission is all-or-none for multiple pure outputs. During capture,
all outputs must be supplied explicitly because allocator activity belongs
outside the captured operator call. Input/output tensors and any external
storage referenced by a graph must remain alive until that graph is destroyed
and its streams are complete.

The default torch_npu adapter enters taskQueue through `RunOpApiV2`, obtains the
raw stream with `.stream(false)`, retains C++ tensor handles until the queued
callback runs, and records ordinary caching-allocator storage on the launch
stream. This makes L1 one ordered torch operator rather than a Python-side raw
stream escape. External, `from_blob`, or custom-allocator storage is not owned
by the caching allocator; its owner must enforce the longer lifetime.

## HBG graph-package lifetime

An HBG image is mutable execution state: scheduler queues, completion flags,
task state, and runtime pointers are consumed or rewritten while it runs. L1
therefore separates:

- a self-contained, pristine graph package in that launch's HostArgs blob;
  CANN snapshots and owns it with the launch task/captured node, analogously to
  inline AscendC tiling data; and
- a context-owned mutable execution slot with a stable device address. The
  AICPU leader restores the pristine shared-memory and runtime-arena images into
  this slot before every eager execution and every graph replay.

Each captured HBG node consequently has its own graph package and
callable-local function table, even when two programs both number their first
kernel as `func_id=0`. There is no fixed resident callable table and PyPTO never
uses an event to guess that CANN has released a captured package.

Generated HBG orchestration may use tensor addresses, shapes, strides, dtypes,
scalar values, and static topology. An orchestration that requires the host to
read or write borrowed tensor contents is rejected before graph construction.

## TRB registry and binary lifetime

TRB needs a device-resident code-address registry. L1 appends immutable entries
dynamically as new callables are encountered. Entries are never evicted,
recycled, or overwritten, and the public API has no 64-callable limit. This is
deliberately safer than reusing an address that a captured task may still
reference, but long-lived processes compiling unbounded specializations can
grow device/AICPU metadata without bound; applications must control
specialization cardinality and monitor memory.

There is no public runtime guarantee that a captured graph keeps a registered
function handle valid after binary unload. Consequently the L1 path never calls
`aclrtBinaryUnLoad` or `rtsBinaryUnload`, including during `shutdown()`. Binary
code and function handles remain process-pinned. This is an explicit lifetime
policy, not an accidental leak to be repaired with eager unloading.

## Optional shutdown and ownership

`pypto.l1.shutdown(device=...)` is optional:

- omitting it quietly pins the L1 owner until process exit;
- it never synchronizes and must be called only after all L1 tasks and all
  graph owners on that device are quiescent;
- repeated calls are idempotent;
- a teardown failure retains the owner and can be retried;
- Python GC and `atexit` never perform runtime teardown or binary unload.

Destroying one ACLGraph is not a reason to shut down the device owner because
other graphs may still reference it.

## Supported boundary

- A2/A3 onboard only; A5 and simulator execution are outside the current
  verified scope.
- Static shape, dtype, stride, and argument layout; tensor addresses and scalar
  values may vary.
- Eager pure-output allocation is supported; capture requires preallocated
  explicit outputs.
- Inference only: no autograd, distributed/`CommCtx`, SDMA, or DFX.
- Concurrent eager calls, eager-versus-replay overlap, and concurrent graph
  replays are unsupported. Host-side overlap is rejected best-effort, but
  externally initiated graph replay cannot always be observed by PyPTO; callers
  must serialize it.
- Workspace remains internal to PyPTO and is shared under that serial execution
  contract.

The low-level `pypto_init`/`L1Context` API remains available for implementation
tests and advanced bring-up, but it is not the normal user interface. See
`tests/st/runtime/l1/test_l1_jit_aclgraph.py` for the maintained public-API TRB
and HBG capture/replay test.
