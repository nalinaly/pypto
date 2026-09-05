# Local A5 environment for nalinaly/pypto

This records the 2026-09-05 setup under `/home/q00473782/inductor`, intended for
basic A5 execution and subsequent `inductor_pto` development. Validation covers
representative numerical checks, not a full regression or performance claim.

## Sources and environment

The execution checkout is `pto_qcy/pypto`, based on
`https://github.com/nalinaly/pypto` commit
`9cece0b730a96fe1a52c2637537132f524ffe1ea`. The upstream `pto/` directory is
reference material only and is excluded from imports, libraries and build caches.

| Component | Workspace-relative path | Revision |
| --- | --- | --- |
| Simpler | `pto_qcy/pypto/runtime` | Fork baseline `06c8b29698c547e339ed78b0acaefa6ca52e0d49`; A5 L1 HBG with header dependencies `9d53975cf65ab67975230a6c70d5c48d6220e25c` |
| PTOAS sources | `pto_qcy/PTOAS` | v0.57, `307d0484a9e7d5e36f01b253d2bebe4d2f45fe81` |
| PTOAS executable | `pto_qcy/.venv/bin/ptoas` | Official v0.57 CPython 3.12 x86_64 wheel |
| PTO-ISA sources | `pto_qcy/pto-isa` | `f51c92f610827daad0ddfb383072e03d514b4ae9` |
| Build-managed PTO-ISA | `pto_qcy/pypto/runtime/build/pto-isa` | Same revision, resolved from `runtime/pto_isa.pin` |
| Python environment | `pto_qcy/.venv` | Python 3.12.3, isolated from system-site-packages |
| PyTorch / torch-npu | Same venv | `2.12.0+cpu` / `2.12.0` |

CANN is the existing installation at
`/home/q00473782/Ascend/cann-9.2.0-weekly.20260902.01/cann-9.2.0`.
No standalone LLVM was installed; neither the cannbot_dsl toolchain nor
`/home/q00473782/llvm` was changed.

The wheel in `pto_qcy/downloads/` is
`ptoas-0.57-cp312-cp312-manylinux_2_34_x86_64.whl`, SHA256
`50973353d15ec78e81745ddef64029b918a9963e1c5735bfa7e65c00a94a7f92`.
The x86_64 checksum in `toolchain/versions.env` covers the CPython 3.10 wheel,
so it must not be used for this CPython 3.12 wheel.

The Simpler fix is a local, unpushed commit retained by the parent gitlink.
Carry that submodule commit when migrating this environment to another machine.

## Activation and rebuilding

```bash
cd /home/q00473782/inductor
source pto_qcy/env.sh
bash pto_qcy/rebuild.sh
```

Activation loads CANN and the independent venv, removes reference-checkout
paths and CANN's unrelated `pypto` package path, and points all PyPTO dependency
roots and compiler caches into `pto_qcy`. Repeated activation deduplicates paths.
`PTO_PLATFORM` defaults to `a5` only when unset. The venv contains CMake, Ninja,
nanobind, scikit-build-core, pybind11, numpy, pytest and ruff.

The rebuild script uses the current checkout without upgrading dependencies:

```bash
cd /home/q00473782/inductor/pto_qcy/pypto
source ../env.sh
source .claude/skills/testing/load-env.sh
python -m pip install --no-build-isolation -e runtime \
    --config-settings=build.targets=build_package_a5
python -m pip install --no-build-isolation -e .
python -m pip install --no-build-isolation --no-deps -e ../../inductor_pto
python -m pip check
```

The testing loader defaults to two build jobs. Both native extensions are built
from this checkout; no `.so` is copied from `pto/`. Build provenance is recorded
in `runtime/build/lib/pto_isa_build.json`.

## Hardware checks

The machine reports Ascend950PR and npu-smi 25.7.rc1.2. Its driver rejects `-c 0`;
the architecture precheck retries the board query without `-c` only for that
specific error and identifies A5 using CANN's `Short_SoC_version=Ascend950`.
Use `npu-smi info` to choose an idle card. Validation used physical card 1,
mapped to logical card 0 by `ASCEND_RT_VISIBLE_DEVICES=1`, with serial direct
execution because `task-submit` is unavailable. Other sessions were not reset.

**PyPTO's `RunConfig()` defaults to `a2a3sim` and does not read `PTO_PLATFORM`.**
Direct calls must specify `RunConfig(platform="a5", device_id=0)`.
The existing `tile_add`, `tile_softmax` and `matmul_64` examples passed A5 TRB
numerical checks, with normal process exit; see `pto_qcy/logs/pypto-a5-smoke.log`.
Reproduce using the existing JIT definitions:

```bash
cd /home/q00473782/inductor/pto_qcy/pypto
source ../env.sh
ASCEND_RT_VISIBLE_DEVICES=1 python - <<'PY'
import importlib
import torch
from pypto.runtime import RunConfig

cfg = RunConfig(platform="a5", device_id=0, runtime="tensormap_and_ringbuffer")
torch.manual_seed(0)
cases = [
    ("examples.beginner.01_hello_world", "tile_add", 128, lambda a, b: a + b, 1e-5),
    ("examples.intermediate.02_softmax", "tile_softmax", 64, None, 1e-5),
    ("examples.beginner.05_matmul", "matmul_64", 64, lambda a, b: a @ b, 1e-3),
]
for module, name, size, golden, tolerance in cases:
    fn = getattr(importlib.import_module(module), name)
    a, b = torch.randn(size, size), torch.randn(size, size)
    out = torch.zeros_like(a)
    if golden is None:
        fn(a, out, config=cfg)
        expected = torch.softmax(a, dim=-1)
    else:
        fn(a, b, out, config=cfg)
        expected = golden(a, b)
    torch.testing.assert_close(out, expected, rtol=tolerance, atol=tolerance)
    print("PASS A5", name)
PY
```

TRB and HBG both build and have representative hardware checks. Unavailable
CANN CPU_TOPO queries use device-side OCCUPY and the shipped A5 topology fallback.

## Same-process device argument ABI

Public `Worker.run` requires address-free Buffer identities. A manually created
raw-pointer `DeviceTensor` cannot enter that protocol.
`CompiledProgram.build_chip_args(*args)` and `compiled[name].build_chip_args`
provide explicit packing for `simpler.task_interface.ChipWorker.run` in the
same process. They return `(chip_args, owners, return_style)` and retain the
existing argument-count, shape, dtype and scalar-type validation.

`DeviceTensor` addresses use `child_memory=True`; tensors and scalars occupy
separate ordered pools. Callers must retain original storage and the returned
owners through the blocking run, synchronize producer/consumer streams, and
ensure addresses belong to the selected device. Public Worker ownership guards
remain enforced. Two focused unit checks cover pointer/scalar packing and shape
rejection before packing.

See `inductor_pto/docs/a5-fork-runtime.md` for the integration and six representative
Inductor checks on the L2 path. Current L1 integration is described below.

## A5 HBG L1 and both frontends

The runtime gitlink includes the native HBG borrowed-stream lifecycle, immutable
graph restore and A5 platform launch ABI. The public `@pl.jit(execution="l1",
runtime="host_build_graph")` accepts `RunConfig(platform="a5", device_id=0,
runtime="host_build_graph")`. Specify that config on every direct PyPTO call;
the no-config PyPTO JIT default remains A2/A3. See the [L1 guide](../user/04-l1-aclgraph.md).

Inductor and the adapter for official `pytorch/helion` share HBG L1/TaskQueue as
the A5 default. `PTO_RUNTIME=tensormap_and_ringbuffer` explicitly selects L2.
The official Helion checkout is `/home/q00473782/inductor/helion`; no upstream
`pto/` runtime or separate LLVM build is used.

Focused A5 checks passed: low-level L1 eager plus three graph replays; public
JIT eager output allocation, appending a second callable after warmup, and a
PyTorch/add/mul/PyTorch captured chain; both frontends' guarded L1 transport;
Helion 64x64 matmul; explicit Inductor L2/TRB. The two JIT CPU checks retain the
A2/A3 default and allow explicit A5 lowering. Logs are under
`pto_qcy/logs/a5-l1-*` and `a5-l2-trb-regression.json`.

The runtime pin retains the fork's A2/A3 post-close worker retirement fix
(`06c8b29`). Its shared L1 report layout is 128 bytes. A5 AIC/AIV custom
build commands now track runtime and platform headers so an incremental build
cannot reuse objects compiled against the old 64-byte layout. After this merge
and rebuild, the public JIT eager allocator and ACLGraph replay check passed
again in 8.94 seconds; see `pto_qcy/logs/pypto-push-header-a5.log`.
The runtime commit is on `nalinaly/simpler` branch `a5-l1-hbg`.

Reproduce the public JIT hardware check on an idle physical card 1:

```bash
cd /home/q00473782/inductor/pto_qcy/pypto
source ../env.sh
source .claude/skills/testing/load-env.sh
export ASCEND_PROCESS_LOG_PATH="$PWD/../logs/a5-l1-public-jit-device"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
ASCEND_RT_VISIBLE_DEVICES=1 PYPTO_L1_JIT_TEST_RUNTIME=host_build_graph \
  python -m pytest tests/st/runtime/l1/test_l1_jit_aclgraph.py \
  --platform=a5 --device=0 -q
```

A5 L1 currently uses HBG's scheduler path. Direct-AIV optimization, A5 TRB L1,
concurrent replay, SDMA, distributed execution and DFX are not included. These
checks do not claim full coverage, simulator regression or performance results.
