# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Process-owned dispatch registry for ``@pl.jit(execution="l1")``.

The registry is deliberately strong and has no GC/atexit finalizer. Captured
ACLGraph nodes can outlive every Python wrapper that originally launched them,
so only an explicit, externally-quiesced :func:`shutdown` retires an owner.
"""

import threading
from dataclasses import dataclass, field
from typing import Any

import torch

from pypto.ir.compiled_program import CompiledProgram

from .l1 import L1Config, L1Context, L1InitializationError, _tensor_device_index, _tensor_device_type


@dataclass
class _DeviceOwner:
    device: int
    platform: str
    runtime: str
    context: L1Context
    config: L1Config
    owner_thread: int
    state: str = "ready"
    invoke_lock: threading.Lock = field(default_factory=threading.Lock)

    def check_admission(self) -> None:
        if self.state != "ready":
            raise RuntimeError(f"PyPTO L1 device {self.device} is {self.state}; no new calls are accepted")
        current_thread = threading.get_ident()
        if current_thread != self.owner_thread:
            raise RuntimeError(
                "PyPTO L1 owner is thread-affine in v1: "
                f"owner={self.owner_thread}, caller={current_thread}, device={self.device}"
            )


_REGISTRY_LOCK = threading.RLock()
_DEVICE_OWNERS: dict[int, _DeviceOwner] = {}


def _infer_device(arguments: list[Any]) -> int:
    tensors = [value for value in arguments if isinstance(value, torch.Tensor)]
    if not tensors:
        raise TypeError("@pl.jit(execution='l1') requires at least one NPU tensor argument")
    devices: set[int] = set()
    for tensor in tensors:
        if _tensor_device_type(tensor) != "npu":
            raise ValueError(f"PyPTO L1 tensor arguments must be on NPU, got {tensor.device}")
        devices.add(_tensor_device_index(tensor))
    if len(devices) != 1:
        raise ValueError(f"PyPTO L1 arguments span multiple devices: {sorted(devices)}")
    return next(iter(devices))


def _config_from_run_config(run_config: Any | None) -> L1Config:
    if run_config is None:
        return L1Config()
    return L1Config(
        aicpu_thread_num=getattr(run_config, "aicpu_thread_num", None),
        ring_task_window=getattr(run_config, "ring_task_window", None),
        ring_heap=getattr(run_config, "ring_heap", None),
        ring_dep_pool=getattr(run_config, "ring_dep_pool", None),
    )


def _get_or_create_owner(
    program: CompiledProgram,
    arguments: list[Any],
    *,
    runtime: str,
    run_config: Any | None,
) -> _DeviceOwner:
    device = _infer_device(arguments)
    if program.runtime_name != runtime:
        raise ValueError(
            f"compiled program runtime {program.runtime_name!r} does not match "
            f"L1 decorator runtime {runtime!r}"
        )
    with _REGISTRY_LOCK:
        owner = _DEVICE_OWNERS.get(device)
        if owner is not None:
            owner.check_admission()
            if owner.runtime != runtime or owner.platform != program.platform:
                raise RuntimeError(
                    f"device {device} already owns an active PyPTO L1 runtime "
                    f"runtime={owner.runtime!r}, platform={owner.platform!r}; "
                    f"requested runtime={runtime!r}, platform={program.platform!r}. "
                    "Destroy all graphs, externally quiesce the device, then use a new process."
                )
            requested_config = _config_from_run_config(run_config)
            if owner.config != requested_config:
                raise RuntimeError(
                    f"device {device} already owns a PyPTO L1 context with frozen runtime config "
                    f"{owner.config!r}; requested {requested_config!r}"
                )
            return owner

        context_config = _config_from_run_config(run_config)
        context: L1Context | None = None
        try:
            context = L1Context([program], device=device, config=context_config)
        except L1InitializationError as exc:
            cleanup_owner = _DeviceOwner(
                device=device,
                platform=program.platform,
                runtime=runtime,
                context=exc.cleanup_context,
                config=context_config,
                owner_thread=threading.get_ident(),
                state="cleanup-only",
            )
            _DEVICE_OWNERS[device] = cleanup_owner
            raise
        except RuntimeError as exc:
            raise RuntimeError(
                "PyPTO L1 initialization must complete during an ordinary eager call before "
                "ACLGraph capture. Call this specialization outside capture, synchronize "
                f"externally, then capture it. Native initialization error: {exc}"
            ) from exc
        owner = _DeviceOwner(
            device=device,
            platform=program.platform,
            runtime=runtime,
            context=context,
            config=context_config,
            owner_thread=threading.get_ident(),
        )
        _DEVICE_OWNERS[device] = owner
        return owner


def dispatch(
    program: CompiledProgram,
    ordered_arguments: list[Any],
    *,
    runtime: str,
    run_config: Any | None,
) -> object:
    """Prepare/append and enqueue one JIT specialization as a normal L1 op."""
    owner = _get_or_create_owner(program, ordered_arguments, runtime=runtime, run_config=run_config)
    owner.check_admission()
    if not owner.invoke_lock.acquire(blocking=False):
        raise RuntimeError(
            "concurrent PyPTO L1 host invocation is unsupported in v1; serialize calls and graph replays"
        )
    try:
        op = owner.context.add_program(program)
        param_infos, output_indices, _ = program._get_metadata()
        if len(ordered_arguments) != len(param_infos):
            raise TypeError(
                f"L1 JIT dispatch expected {len(param_infos)} bound arguments, got {len(ordered_arguments)}"
            )
        output_set = set(output_indices)
        inputs = tuple(value for index, value in enumerate(ordered_arguments) if index not in output_set)
        outputs = [ordered_arguments[index] for index in output_indices]
        out: object
        if not outputs:
            out = op(*inputs)
        elif len(outputs) == 1:
            out = op(*inputs, out=outputs[0])
        else:
            out = op(*inputs, out=tuple(outputs))
        return out
    finally:
        owner.invoke_lock.release()


def shutdown(*, device: int) -> None:
    """Optionally retire one externally-quiesced L1 device owner.

    The call is idempotent. It never synchronizes and never unloads the CANN
    runtime binary; a failed close retains the owner for an explicit retry.
    """
    if isinstance(device, bool) or not isinstance(device, int):
        raise TypeError("device must be a non-bool integer")
    if device < 0:
        raise ValueError("device must be non-negative")
    with _REGISTRY_LOCK:
        owner = _DEVICE_OWNERS.get(device)
        if owner is None or owner.state == "retired":
            return
        if threading.get_ident() != owner.owner_thread:
            raise RuntimeError(
                f"PyPTO L1 shutdown must run on owner thread {owner.owner_thread}, "
                f"got {threading.get_ident()}"
            )
        if not owner.invoke_lock.acquire(blocking=False):
            raise RuntimeError("cannot shutdown PyPTO L1 while a host invocation is active")
        owner.state = "retiring"
        try:
            owner.context.close()
        except BaseException:
            owner.state = "cleanup-only"
            raise
        finally:
            owner.invoke_lock.release()
        owner.state = "retired"


__all__ = ["dispatch", "shutdown"]
