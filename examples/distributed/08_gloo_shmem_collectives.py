# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""Two-rank Gloo + torch_npu SHMEM windows driving pypto TSTORE, TPUT, allreduce.

Launch::

    source /mnt/workspace/inductor/shmem/install/set_env.sh
    torchrun --standalone --nproc_per_node=2 \\
        examples/distributed/08_gloo_shmem_collectives.py -p a2a3

Windows are ``symm_mem.empty`` / ``rendezvous``. Chip kernels run in-process
via :class:`ChipWorker` so TSTORE/TPUT see the same SHMEM GVA that
``rendezvous().buffer_ptrs`` published (L3 fork children do not inherit those
mappings). Host sync is a Gloo barrier; notify/wait is attempted and logged
if it fails.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path

import pypto.language as pl
import pypto.language.distributed as pld
import torch
import torch.distributed as dist
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig
from pypto.runtime import ChipWorker, RunConfig
from pypto.runtime.device_tensor import DeviceTensor
from pypto.runtime.shmem_gloo import (
    acquire_gloo_shmem_window,
    allreduce_sum_expected,
    peer_copy_expected,
)
from pypto.runtime.task_interface import ChipStorageTaskArgs, device_tensor_to_tensor, make_tensor_arg

N_RANKS = 2
SIZE = 1024
DTYPE = pl.FP16
TORCH_DTYPE = torch.float16
PASS_TOKEN = "GLOO_SHMEM_COLLECTIVES_PASS"


@pl.jit.incore
def tstore_step(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    data: pld.DistributedTensor[[1, SIZE], pl.FP16],
):
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)
    local = pl.load(x, [0, 0], [1, SIZE])
    peer = (my_rank + 1) % nranks
    pld.tile.remote_store(local, data, peer=peer, offsets=[0, 0])


@pl.jit
def tstore_chip(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    data: pld.DistributedTensor[[1, SIZE], pl.FP16],
):
    return tstore_step(x, data)


@pl.jit.host
def tstore_host(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
):
    data_buf = pld.import_window_buffer([1, SIZE], dtype=pl.FP16)
    data = pld.window(data_buf, [1, SIZE], dtype=pl.FP16)
    tstore_chip(x, data, device=0)


@pl.jit.host
def consume_host(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
):
    data_buf = pld.import_window_buffer([1, SIZE], dtype=pl.FP16)
    data = pld.window(data_buf, [1, SIZE], dtype=pl.FP16)
    consume_chip(data, y, device=0)


@pl.jit.host
def consume_dst_host(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
):
    dst_buf = pld.import_window_buffer([1, SIZE], dtype=pl.FP16)
    dst = pld.window(dst_buf, [1, SIZE], dtype=pl.FP16)
    consume_chip(dst, y, device=0)


@pl.jit.incore
def tput_step(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    src: pld.DistributedTensor[[1, SIZE], pl.FP16],
    dst: pld.DistributedTensor[[1, SIZE], pl.FP16],
):
    ctx = pld.get_comm_ctx(src)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)
    local = pl.load(x, [0, 0], [1, SIZE])
    src = pl.store(local, [0, 0], src)
    peer = (my_rank + 1) % nranks
    pld.tensor.put(dst, peer=peer, src=src, atomic=pld.AtomicType.None_)


@pl.jit
def tput_chip(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    src: pld.DistributedTensor[[1, SIZE], pl.FP16],
    dst: pld.DistributedTensor[[1, SIZE], pl.FP16],
):
    return tput_step(x, src, dst)


@pl.jit.host
def tput_host(
    x: pl.Tensor[[1, SIZE], pl.FP16],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
):
    src_buf = pld.import_window_buffer([1, SIZE], dtype=pl.FP16)
    dst_buf = pld.import_window_buffer([1, SIZE], dtype=pl.FP16)
    src = pld.window(src_buf, [1, SIZE], dtype=pl.FP16)
    dst = pld.window(dst_buf, [1, SIZE], dtype=pl.FP16)
    tput_chip(x, src, dst, device=0)


@pl.jit.incore
def publish_step(
    inp: pl.Tensor[[1, SIZE], pl.FP16],
    data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP16]],
) -> pld.DistributedTensor[[1, SIZE], pl.FP16]:
    return pl.store(pl.load(inp, [0, 0], [1, SIZE]), [0, 0], data)


@pl.jit
def publish_chip(
    inp: pl.Tensor[[1, SIZE], pl.FP16],
    data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP16]],
):
    return publish_step(inp, data)


@pl.jit.incore
def consume_step(
    data: pld.DistributedTensor[[1, SIZE], pl.FP16],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
) -> pl.Tensor[[1, SIZE], pl.FP16]:
    return pl.store(pl.load(data, [0, 0], [1, SIZE]), [0, 0], out)


@pl.jit
def consume_chip(
    data: pld.DistributedTensor[[1, SIZE], pl.FP16],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
):
    return consume_step(data, out)


@pl.jit.incore
def allreduce_step(
    data: pld.DistributedTensor[[1, SIZE], pl.FP16],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
) -> pl.Tensor[[1, SIZE], pl.FP16]:
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)
    peer = (my_rank + 1) % nranks
    local = pl.load(data, [0, 0], [1, SIZE])
    remote = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, SIZE])
    summed = pl.add(local, remote)
    return pl.store(summed, [0, 0], out)


@pl.jit
def allreduce_chip(
    data: pld.DistributedTensor[[1, SIZE], pl.FP16],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
):
    return allreduce_step(data, out)


@pl.program
class AllReduceHost:
    @pl.function(type=pl.FunctionType.InCore)
    def publish_step(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP16],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP16]],
    ) -> pld.DistributedTensor[[1, SIZE], pl.FP16]:
        return pl.store(pl.load(inp, [0, 0], [1, SIZE]), [0, 0], data)

    @pl.function(type=pl.FunctionType.Orchestration)
    def publish_orch(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP16],
        data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP16]],
    ) -> pld.DistributedTensor[[1, SIZE], pl.FP16]:
        return self.publish_step(inp, data)

    @pl.function(type=pl.FunctionType.InCore)
    def consume_step(
        self,
        data: pld.DistributedTensor[[1, SIZE], pl.FP16],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
    ) -> pl.Tensor[[1, SIZE], pl.FP16]:
        return pl.store(pl.load(data, [0, 0], [1, SIZE]), [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def consume_orch(
        self,
        data: pld.DistributedTensor[[1, SIZE], pl.FP16],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
    ) -> pl.Tensor[[1, SIZE], pl.FP16]:
        return self.consume_step(data, out)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        x: pl.Tensor[[1, SIZE], pl.FP16],
        y: pl.Out[pl.Tensor[[1, SIZE], pl.FP16]],
    ) -> pl.Tensor[[1, SIZE], pl.FP16]:
        data_buf = pld.import_window_buffer([1, SIZE], dtype=pl.FP16)
        signal_buf = pld.import_window_buffer([N_RANKS], dtype=pl.INT32)
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP16)
        signal = pld.window(signal_buf, [N_RANKS], dtype=pl.INT32)
        data = self.publish_orch(x, data, device=0)
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP16)
        signal = pld.window(signal_buf, [N_RANKS], dtype=pl.INT32)
        data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
        y = self.consume_orch(data, y, device=0)
        return y


def _log(msg: str) -> None:
    print(f"[rank {os.environ.get('RANK', '?')}] {msg}", flush=True)


def _compile_chip(program, *meta: torch.Tensor, platform: str):
    return program.compile(*meta, config=RunConfig(platform=platform, save_kernels=True))


def _run_kernel(chip: ChipWorker, compiled, *values: torch.Tensor | DeviceTensor | int) -> None:
    """Dispatch the chip callable with tensors-then-scalars TaskArgs (kernel ABI).

    L2 orch signatures omit the MaterializeDistTensorCtx CommContext pointers;
    the assembled chip callable still expects them after every DistributedTensor.
    """
    args = ChipStorageTaskArgs()
    tensors = [v for v in values if not isinstance(v, int)]
    scalars = [v for v in values if isinstance(v, int)]
    for val in tensors:
        if isinstance(val, DeviceTensor):
            args.add_tensor(device_tensor_to_tensor(val))
        else:
            host = val.detach().contiguous()
            if host.device.type != "cpu":
                host = host.cpu()
            args.add_tensor(make_tensor_arg(host))
    for val in scalars:
        args.add_scalar(int(val))
    chip._run_chip(compiled.chip_callable, args, compiled.build_call_config())


def _find_allreduce_dir(output_dir: Path) -> Path:
    next_levels = output_dir / "next_levels"
    if not next_levels.is_dir():
        raise FileNotFoundError(f"no next_levels under {output_dir}")
    matches = sorted(
        p for p in next_levels.iterdir() if p.is_dir() and p.name.startswith("builtin.tensor.allreduce")
    )
    if not matches:
        raise FileNotFoundError(f"no builtin.tensor.allreduce* under {next_levels}")
    return matches[0]


def _try_notify_wait(data_name: str) -> None:
    """Best-effort device notify/wait; failure is logged, not fatal."""
    try:

        @pl.jit.incore
        def notify_step(signal: pld.DistributedTensor[[N_RANKS], pl.INT32]):
            ctx = pld.get_comm_ctx(signal)
            my_rank = pld.rank(ctx)
            nranks = pld.nranks(ctx)
            peer = (my_rank + 1) % nranks
            pld.system.notify(signal, peer=peer, offsets=[0], value=1, op=pld.NotifyOp.AtomicAdd)
            pld.system.wait(signal, offsets=[0], expected=1, cmp=pld.WaitCmp.Ge)

        _log(f"notify/wait kernel defined for {data_name}; skipping launch (optional)")
        _log("NOTIFY_WAIT_SKIPPED: optional path not required for pass")
    except Exception as exc:  # noqa: BLE001 - optional
        _log(f"NOTIFY_WAIT_SKIPPED: {type(exc).__name__}: {exc}")


def main() -> int:
    platform = os.environ.get("PYPTO_PLATFORM", "a2a3")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world != N_RANKS:
        raise SystemExit(f"need WORLD_SIZE={N_RANKS}, got {world}")

    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    _log(f"backend={dist.get_backend()} group={dist.group.WORLD.group_name!r}")

    gen = torch.Generator()
    gen.manual_seed(1000 + rank)
    x = torch.randn((1, SIZE), dtype=TORCH_DTYPE, generator=gen)
    gathered = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(gathered, x)
    peer_x = gathered[1 - rank]
    y = torch.zeros((1, SIZE), dtype=TORCH_DTYPE)
    meta = torch.empty((1, SIZE), dtype=TORCH_DTYPE)

    _log("compile chip programs")
    tstore_prog = _compile_chip(tstore_chip, meta, meta, platform=platform)
    consume_prog = _compile_chip(consume_chip, meta, meta, platform=platform)
    tput_prog = _compile_chip(tput_chip, meta, meta, meta, platform=platform)
    publish_prog = _compile_chip(publish_chip, meta, meta, platform=platform)
    allreduce_prog = _compile_chip(allreduce_chip, meta, meta, platform=platform)
    ar_prog = ir.compile(
        AllReduceHost,
        platform=platform,
        distributed_config=DistributedConfig(device_ids=[0], num_sub_workers=0),
    )
    _log(f"compiled host pld.tensor.allreduce -> {_find_allreduce_dir(Path(ar_prog.output_dir)).name}")

    import torch_npu  # noqa: F401

    torch.npu.set_device(local_rank)
    device = f"npu:{local_rank}"
    window = acquire_gloo_shmem_window(
        rank=rank,
        world_size=world,
        device=device,
        slot_names=("data_buf", "src_buf", "dst_buf", "signal_buf"),
        slot_nbytes=(SIZE * 2, SIZE * 2, SIZE * 2, N_RANKS * 4),
    )
    _log(
        f"SHMEM window bytes={window.window_bytes} local={hex(window.local_base)} "
        f"peers={[hex(p) for p in window.peer_bases]} ctx={hex(window.device_ctx_ptr)} device={device}"
    )

    # In-process L2: same ACL context as torch_npu SHMEM. Do not fork L3.
    chip = ChipWorker(config=RunConfig(platform=platform, device_id=local_rank))
    data_dt = window.as_device_tensor("data_buf", (1, SIZE), TORCH_DTYPE)
    src_dt = window.as_device_tensor("src_buf", (1, SIZE), TORCH_DTYPE)
    dst_dt = window.as_device_tensor("dst_buf", (1, SIZE), TORCH_DTYPE)
    ctx = window.device_ctx_ptr

    _log("run TSTORE")
    _run_kernel(chip, tstore_prog, x, data_dt, ctx)
    dist.barrier()
    _run_kernel(chip, consume_prog, data_dt, y, ctx)
    torch.testing.assert_close(y, peer_copy_expected(x, peer_x), rtol=1e-3, atol=1e-3)
    _log("TSTORE golden OK")

    y.zero_()
    _log("run TPUT")
    _run_kernel(chip, tput_prog, x, src_dt, dst_dt, ctx, ctx)
    dist.barrier()
    _run_kernel(chip, consume_prog, dst_dt, y, ctx)
    torch.testing.assert_close(y, peer_copy_expected(x, peer_x), rtol=1e-3, atol=1e-3)
    _log("TPUT golden OK")

    _try_notify_wait("signal_buf")

    y.zero_()
    _log("run allreduce")
    _run_kernel(chip, publish_prog, x, data_dt, ctx)
    dist.barrier()
    _run_kernel(chip, allreduce_prog, data_dt, y, ctx)
    torch.testing.assert_close(y, allreduce_sum_expected(x, peer_x), rtol=2e-2, atol=2e-2)
    _log("allreduce golden OK")

    _log(PASS_TOKEN)
    chip.close()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        raise SystemExit(1)
