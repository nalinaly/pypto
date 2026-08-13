# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Gloo + torch_npu symmetric-memory windows for pypto comm ops.

The host process group is Gloo (CPU). Data windows come from
``torch.distributed._symmetric_memory`` (NPUSHMEM), not from
``orch.allocate_domain`` / HCCL Fabric. Generated host orch still calls
``(_domain_provider or orch.allocate_domain)(...)``; pass
:func:`make_shmem_domain_provider` as ``RunConfig.domain_provider`` so the
HCCL path is never entered.

``CommContext`` packing matches ``runtime/src/common/platform_comm/comm_context.h``
(1056 bytes). Device kernels read ``windowsIn[peer]`` for TPUT/TSTORE.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pypto.runtime.device_tensor import DeviceTensor

# Mirrors comm_context.h. Do not reorder without coordinating pto-isa.
COMM_CONTEXT_SIZE = 1056
COMM_CTX_SLOT = "_comm_ctx"
COMM_MAX_RANK_NUM = 64
_OFF_RANK_ID = 16
_OFF_RANK_NUM = 20
_OFF_WIN_SIZE = 24
_OFF_WINDOWS_IN = 32
_OFF_WINDOWS_OUT = 544

# Same rounding codegen uses in EmitCommDomainAllocations.
COMM_BUFFER_ALIGN = 32


def align_up(nbytes: int, align: int = COMM_BUFFER_ALIGN) -> int:
    """Round *nbytes* up to a multiple of *align*."""
    if nbytes < 0:
        raise ValueError(f"nbytes must be non-negative, got {nbytes}")
    if align <= 0:
        raise ValueError(f"align must be positive, got {align}")
    return ((nbytes + align - 1) // align) * align


def carve_window_layout(slot_nbytes: Sequence[int], *, align: int = COMM_BUFFER_ALIGN) -> tuple[list[int], int]:
    """Return ``(offsets, window_bytes)`` for consecutive aligned slots."""
    offsets: list[int] = []
    cursor = 0
    for n in slot_nbytes:
        if n <= 0:
            raise ValueError(f"slot nbytes must be positive, got {n}")
        offsets.append(cursor)
        cursor += align_up(n, align)
    return offsets, cursor


def pack_comm_context(
    rank: int,
    world_size: int,
    win_size: int,
    window_bases: Sequence[int],
) -> bytes:
    """Pack a host ``CommContext`` blob (little-endian, 1056 bytes)."""
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} out of range for world_size={world_size}")
    if world_size <= 0 or world_size > COMM_MAX_RANK_NUM:
        raise ValueError(f"world_size {world_size} out of range (1..{COMM_MAX_RANK_NUM})")
    if len(window_bases) != world_size:
        raise ValueError(f"expected {world_size} window bases, got {len(window_bases)}")
    if win_size <= 0:
        raise ValueError(f"win_size must be positive, got {win_size}")

    blob = bytearray(COMM_CONTEXT_SIZE)
    struct.pack_into("<I", blob, _OFF_RANK_ID, int(rank))
    struct.pack_into("<I", blob, _OFF_RANK_NUM, int(world_size))
    struct.pack_into("<Q", blob, _OFF_WIN_SIZE, int(win_size))
    for i, base in enumerate(window_bases):
        struct.pack_into("<Q", blob, _OFF_WINDOWS_IN + i * 8, int(base))
        struct.pack_into("<Q", blob, _OFF_WINDOWS_OUT + i * 8, int(base))
    return bytes(blob)


def unpack_comm_context(blob: bytes) -> dict[str, Any]:
    """Decode a packed ``CommContext`` for tests / debug dumps."""
    if len(blob) != COMM_CONTEXT_SIZE:
        raise ValueError(f"CommContext blob must be {COMM_CONTEXT_SIZE} bytes, got {len(blob)}")
    rank = struct.unpack_from("<I", blob, _OFF_RANK_ID)[0]
    world = struct.unpack_from("<I", blob, _OFF_RANK_NUM)[0]
    win_size = struct.unpack_from("<Q", blob, _OFF_WIN_SIZE)[0]
    bases = [struct.unpack_from("<Q", blob, _OFF_WINDOWS_IN + i * 8)[0] for i in range(world)]
    return {"rank": rank, "world_size": world, "win_size": win_size, "window_bases": bases}


@dataclass
class ShmemWindow:
    """One process-lifetime SHMEM allocation plus carved named slices."""

    tensor: Any
    handle: Any
    window_bytes: int
    local_base: int
    peer_bases: list[int]
    offsets: dict[str, int]
    device_ctx_tensor: Any
    device_ctx_ptr: int

    def as_device_tensor(self, slot: str, shape: Sequence[int], dtype: Any) -> DeviceTensor:
        """Wrap a carved SHMEM slice as a worker-resident :class:`DeviceTensor`."""
        from pypto.runtime.device_tensor import DeviceTensor

        if slot not in self.offsets:
            raise KeyError(f"unknown SHMEM slot {slot!r}; have {sorted(self.offsets)}")
        return DeviceTensor(self.local_base + self.offsets[slot], shape, dtype)


def acquire_gloo_shmem_window(
    *,
    rank: int,
    world_size: int,
    device: str,
    slot_names: Sequence[str],
    slot_nbytes: Sequence[int],
    group_name: str | None = None,
) -> ShmemWindow:
    """``empty`` + ``rendezvous`` one uint8 window and carve named slices.

    Requires an initialized Gloo process group. Does not call
    ``orch.allocate_domain``.
    """
    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    if len(slot_names) != len(slot_nbytes):
        raise ValueError("slot_names and slot_nbytes length mismatch")
    if len(set(slot_names)) != len(slot_names):
        raise ValueError(f"duplicate slot names: {slot_names}")
    if COMM_CTX_SLOT in slot_names:
        raise ValueError(f"{COMM_CTX_SLOT!r} is reserved for the packed CommContext")

    # CommContext lives in the SHMEM heap so TSTORE/TPUT can load windowsIn[]
    # from the same GVA space as the data slices (not a private torch.empty).
    all_names = (COMM_CTX_SLOT, *slot_names)
    all_nbytes = (COMM_CONTEXT_SIZE, *slot_nbytes)
    offsets_list, window_bytes = carve_window_layout(all_nbytes)
    offsets = dict(zip(all_names, offsets_list, strict=True))

    if group_name is None:
        group_name = dist.group.WORLD.group_name
    if not symm_mem.is_symm_mem_enabled_for_group(group_name):
        symm_mem.enable_symm_mem_for_group(group_name)

    tensor = symm_mem.empty((window_bytes,), dtype=torch.uint8, device=device)
    handle = symm_mem.rendezvous(tensor, group=group_name)
    peer_bases = [int(p) for p in handle.buffer_ptrs]
    if len(peer_bases) != world_size:
        raise RuntimeError(f"rendezvous returned {len(peer_bases)} ptrs, expected world_size={world_size}")
    local_base = peer_bases[rank]

    ctx_host = pack_comm_context(rank, world_size, window_bytes, peer_bases)
    ctx_off = offsets[COMM_CTX_SLOT]
    device_ctx_tensor = tensor[ctx_off : ctx_off + COMM_CONTEXT_SIZE]
    device_ctx_tensor.copy_(torch.frombuffer(bytearray(ctx_host), dtype=torch.uint8).to(device))
    torch.npu.synchronize()

    return ShmemWindow(
        tensor=tensor,
        handle=handle,
        window_bytes=window_bytes,
        local_base=local_base,
        peer_bases=peer_bases,
        offsets=offsets,
        device_ctx_tensor=device_ctx_tensor,
        device_ctx_ptr=int(device_ctx_tensor.data_ptr()),
    )


def make_shmem_domain_provider(
    window: ShmemWindow,
    *,
    rank: int,
    world_size: int,
    local_chip: int = 0,
    simpler_worker: Any = None,
    slot_nbytes: Mapping[str, int] | None = None,
) -> Callable[..., Any]:
    """Build a ``_domain_provider`` that never calls ``orch.allocate_domain``.

    ``simpler_worker`` is the L3 ``Worker`` (``DistributedWorker._w``). When
    set, SHMEM bases are registered in ``_child_alloc_prov`` so
    ``submit_next_level`` accepts ``child_memory=True`` tensors.
    """
    from simpler.task_interface import ChipDomainContext, CommDomainHandle

    def _release(_handle: Any) -> None:
        return None

    def provider(**kwargs: Any) -> Any:
        buffers = kwargs["buffers"]
        buffer_ptrs: dict[str, int] = {}
        for spec in buffers:
            name = spec.name
            logical = name.split("__ssa_v", 1)[0]
            if logical in window.offsets:
                buffer_ptrs[name] = window.local_base + window.offsets[logical]
            else:
                raise KeyError(
                    f"generated buffer {name!r} (logical {logical!r}) not in SHMEM layout "
                    f"{sorted(window.offsets)}; LHS names of pld.import_window_buffer "
                    "must match slot_names"
                )
        if simpler_worker is not None:
            lock = getattr(simpler_worker, "_child_prov_lock", None)
            record = getattr(simpler_worker, "_child_prov_record_domain", None)
            if record is not None:
                ctx_mgr = lock if lock is not None else nullcontext()
                with ctx_mgr:
                    record(local_chip, window.local_base, 0, window.window_bytes)
                    record(local_chip, window.device_ctx_ptr, 0, COMM_CONTEXT_SIZE)
                    for spec in buffers:
                        logical = spec.name.split("__ssa_v", 1)[0]
                        nbytes = spec.nbytes
                        if slot_nbytes and logical in slot_nbytes:
                            nbytes = slot_nbytes[logical]
                        record(
                            local_chip,
                            window.local_base + window.offsets[logical],
                            0,
                            int(nbytes),
                        )

        ctx = ChipDomainContext(
            name=str(kwargs.get("name", "shmem")),
            domain_rank=int(rank),
            domain_size=int(world_size),
            device_ctx=window.device_ctx_ptr,
            local_window_base=window.local_base,
            actual_window_size=window.window_bytes,
            buffer_ptrs=buffer_ptrs,
        )
        return CommDomainHandle(
            name=str(kwargs.get("name", "shmem")),
            workers=(local_chip,),
            contexts={local_chip: ctx},
            allocation_id=0,
            _release_fn=_release,
            _domain_size=world_size,
            _domain_ranks={local_chip: rank},
        )

    return provider


def peer_copy_expected(local: Any, peer: Any) -> Any:
    """Golden for a ring-shift / TSTORE / TPUT of *peer* into *local*'s view."""
    return peer.detach().clone()


def allreduce_sum_expected(mine: Any, peer: Any) -> Any:
    """Elementwise sum golden for two-rank allreduce."""
    return mine.to(dtype=mine.dtype) + peer.to(dtype=mine.dtype)
