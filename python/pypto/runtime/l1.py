# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Borrowed-device PyPTO L1 execution for PyTorch and ACLGraph.

L1 is deliberately a single-operator boundary:

* the caller owns the current device, tensors and current torch_npu stream;
* PyPTO keeps its v1 workspace and hidden AICore branch private;
* AICPU is launched on the caller stream and the hidden AICore stream is
  forked/joined with events inside that one operator;
* no method in this module synchronizes a stream/device or inspects capture;
* callables and graph-visible runtime state remain pinned until explicit close.

ACLGraph callers must use the explicit sequence ``prepare() -> warmup() ->
external synchronize -> capture``.  Eager calls offer an automatic first-call
convenience, but that convenience must not be invoked for the first time from
inside capture because PyPTO intentionally cannot detect that situation.

The caller must keep the context and every tensor/storage referenced by a
captured graph alive until that graph can no longer replay.  The default
taskQueue adapter records ordinary torch_npu caching-allocator storage on the
caller stream.  For external/from-blob/custom-allocator storage, allocator
recording may be unavailable, so its external owner must remain alive through
the actual stream completion (and through graph destruction for capture).
"""

from __future__ import annotations

import ctypes
import importlib
import re
import struct
import threading
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from pypto._runtime_names import validate_runtime_name
from pypto.ir.compiled_program import CompiledProgram, _coerce_args, _to_torch_dtype
from pypto.pypto_core.ir import CommCtxType, DistributedTensorType, ParamDirection

from .runner import RunConfig
from .task_interface import (
    CHIP_MAX_SCALAR_ARGS,  # pyright: ignore[reportAttributeAccessIssue]
    CHIP_MAX_TENSOR_ARGS,  # pyright: ignore[reportAttributeAccessIssue]
    MAX_REGISTERED_CALLABLE_IDS,  # pyright: ignore[reportAttributeAccessIssue]
    MAX_TENSOR_DIMS,  # pyright: ignore[reportAttributeAccessIssue]
    ArgDirection,  # pyright: ignore[reportAttributeAccessIssue]
    ChipStorageTaskArgs,  # pyright: ignore[reportAttributeAccessIssue]
    ChipTensor,  # pyright: ignore[reportAttributeAccessIssue]
    torch_dtype_to_datatype,  # pyright: ignore[reportAttributeAccessIssue]
)

_MISSING = object()
_UINT32_MAX = 2**32 - 1
_QUEUE_CALL_ABI_VERSION = 1
_SUPPORTED_L1_SCALAR_DTYPES = frozenset(
    {
        "fp16",
        "fp32",
        "fp64",
        "bfloat16",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bool",
        "index",
    }
)


class L1InitializationError(RuntimeError):
    """L1 init failed while native cleanup ownership remained live.

    ``cleanup_context`` cannot prepare or launch; it exists solely so the
    caller can externally quiesce the device and retry ``close()``.  Keeping
    this owner on the exception is intentional: native rollback failure must
    never turn into an unreachable device claim/DSO leak.
    """

    def __init__(self, message: str, cleanup_context: L1Context) -> None:
        super().__init__(message)
        self.cleanup_context = cleanup_context


@dataclass(frozen=True)
class L1Config:
    """Context-wide configuration for the first L1 implementation.

    DFX, SDMA, simulator, distributed execution and concurrent invocation are
    intentionally absent. The runtime is selected when each program is
    compiled; one context accepts either TRB or HBG programs, but never mixes
    the two. Ring sizing and the AICPU thread count are provisioned once because
    the context shares persistent Runtime/workspace state across all declared
    operators.

    ``use_task_queue=False`` is a low-level bring-up/debug escape hatch.  The
    default taskQueue adapter is required for the supported PyTorch/ACLGraph
    path because it preserves torch_npu queue ordering and allocator lifetime.
    Direct mode obtains ``current_stream(...).npu_stream`` through the Python
    API; when the global torch_npu taskQueue is enabled that getter drains its
    pending host queue, so direct mode is not a zero-side-effect production
    replacement for the adapter.
    """

    aicpu_thread_num: int | None = None
    ring_task_window: int | list[int] | tuple[int, ...] | None = None
    ring_heap: int | list[int] | tuple[int, ...] | None = None
    ring_dep_pool: int | list[int] | tuple[int, ...] | None = None
    use_task_queue: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.use_task_queue, bool):
            raise TypeError("L1Config.use_task_queue must be bool")


@dataclass
class _OperatorState:
    program: CompiledProgram
    callable_id: int
    chip_callable: Any
    param_infos: list[Any]
    output_indices: tuple[int, ...]
    op_name: str
    warmed: bool = False
    bound_tensor_metadata: tuple[tuple[tuple[int, ...], torch.dtype, tuple[int, ...]], ...] | None = None


def _load_torch_npu():
    try:
        return importlib.import_module("torch_npu")
    except ImportError as exc:
        raise RuntimeError("PyPTO L1 requires torch_npu") from exc


def _load_task_queue_adapter():
    try:
        return importlib.import_module("pypto._torch_npu_l1")
    except ImportError as exc:
        raise RuntimeError(
            "PyPTO was built without the optional torch_npu L1 adapter; "
            "reinstall it in an environment containing matching torch and torch_npu headers/libraries"
        ) from exc


def _validate_task_queue_adapter(adapter: Any, torch_npu: Any) -> None:
    if not callable(getattr(adapter, "enqueue", None)):
        raise RuntimeError("pypto._torch_npu_l1 does not export enqueue()")
    actual_abi = getattr(adapter, "QUEUE_CALL_ABI_VERSION", None)
    if actual_abi != _QUEUE_CALL_ABI_VERSION:
        raise RuntimeError(
            "incompatible pypto._torch_npu_l1 queue-call ABI: "
            f"built={actual_abi!r}, expected={_QUEUE_CALL_ABI_VERSION}"
        )

    runtime_versions = {
        "torch": str(torch.__version__),
        "torch_npu": str(getattr(torch_npu, "__version__", "unknown")),
    }
    build_versions = {
        "torch": str(getattr(adapter, "BUILD_TORCH_VERSION", "unknown")),
        "torch_npu": str(getattr(adapter, "BUILD_TORCH_NPU_VERSION", "unknown")),
    }
    mismatches = [
        f"{name}: built={build_versions[name]!r}, runtime={runtime_versions[name]!r}"
        for name in runtime_versions
        if build_versions[name] not in ("", "unknown")
        and runtime_versions[name] not in ("", "unknown")
        and build_versions[name] != runtime_versions[name]
    ]
    if mismatches:
        raise RuntimeError(
            "pypto._torch_npu_l1 was compiled against different framework versions; " + "; ".join(mismatches)
        )


def _current_device(torch_npu: Any) -> int:
    return int(torch_npu.npu.current_device())


def _current_stream(torch_npu: Any, device: int):
    # Passing the explicit device avoids an accidental fallback to another
    # process-visible card in direct-debug mode.
    return torch_npu.npu.current_stream(device)


def _build_runtime_binaries(platform: str, runtime: str):
    runtime_builder = importlib.import_module("simpler_setup.runtime_builder")
    return runtime_builder.RuntimeBuilder(platform).get_binaries(runtime)


def _make_native_worker():
    task_interface = importlib.import_module("simpler.task_interface")
    return task_interface.ChipWorker()


def _contains_distributed_types(program: CompiledProgram) -> bool:
    return any(
        isinstance(param.type, (CommCtxType, DistributedTensorType))
        for func in program.program.functions.values()
        for param in func.params
    )


_PARAM_TO_RUNTIME_DIRECTION = {
    ParamDirection.In: ArgDirection.IN,
    ParamDirection.Out: ArgDirection.OUT,
    ParamDirection.InOut: ArgDirection.INOUT,
}


def _validate_final_callable_signature(chip_callable: Any, param_infos: Sequence[Any]) -> None:
    tensor_infos = [info for info in param_infos if info.shape is not None]
    scalar_infos = [info for info in param_infos if info.shape is None]
    actual_tensor_count = int(chip_callable.sig_count)
    actual_scalar_count = int(chip_callable.scalar_count)
    if actual_tensor_count != len(tensor_infos) or actual_scalar_count != len(scalar_infos):
        raise ValueError(
            "assembled L1 callable signature does not match program metadata: "
            f"callable tensors/scalars={actual_tensor_count}/{actual_scalar_count}, "
            f"metadata tensors/scalars={len(tensor_infos)}/{len(scalar_infos)}; "
            "the final lowering may have materialized an unsupported CommCtx/distributed argument"
        )
    for index, info in enumerate(tensor_infos):
        expected = _PARAM_TO_RUNTIME_DIRECTION.get(info.direction)
        if expected is None:
            raise ValueError(f"unsupported L1 parameter direction {info.direction!r} for {info.name!r}")
        actual = chip_callable.sig(index)
        # The public program signature owns the API-level mutability contract,
        # while the assembled signature owns the runtime dependency/access
        # contract.  Outlining a store into an already-existing output buffer
        # deliberately represents the child slot as INOUT even when generated
        # AICore code never reads the old bytes.  That conservative read bit is
        # harmless for L1's direct-device binder, but changing a read-only
        # public input into any writer (or losing writability) is not.
        expected_writable = expected in (ArgDirection.OUT, ArgDirection.INOUT)
        actual_writable = actual in (ArgDirection.OUT, ArgDirection.INOUT)
        if expected_writable != actual_writable:
            raise ValueError(
                f"assembled L1 tensor writability mismatch for {info.name!r}: "
                f"callable={actual!r}, metadata={expected!r}; final lowering must preserve "
                "the public read-only versus writable contract"
            )


def _pack_l1_scalar(value: ctypes._SimpleCData, dtype: Any) -> int:
    """Pack one coerced scalar according to its declared orchestration type.

    The device uses ``from_u64<declared C++ type>`` and therefore reads the
    low ``sizeof(type)`` bytes verbatim.  In particular, FP16/BF16 parameters
    must carry a 16-bit value; reusing the generic c_float packer would put an
    FP32 bit pattern in the slot and silently expose its unrelated low 16 bits.
    """

    dtype_name = str(dtype)
    if dtype_name not in _SUPPORTED_L1_SCALAR_DTYPES:
        raise ValueError(f"unsupported PyPTO L1 scalar dtype {dtype_name!r}")
    if not isinstance(value, ctypes._SimpleCData):
        raise TypeError(f"L1 scalar was not coerced to a ctypes value: {type(value).__name__}")

    scalar = value.value
    if dtype_name == "fp16":
        return struct.unpack("<H", struct.pack("<e", float(scalar)))[0]
    if dtype_name == "fp32":
        return struct.unpack("<I", struct.pack("<f", float(scalar)))[0]
    if dtype_name == "fp64":
        return struct.unpack("<Q", struct.pack("<d", float(scalar)))[0]
    if dtype_name == "bfloat16":
        # _coerce_args deliberately uses c_float for BF16, so the conversion
        # begins from the exact FP32 value that the public scalar API accepts.
        fp32_bits = struct.unpack("<I", struct.pack("<f", float(scalar)))[0]
        exponent = fp32_bits & 0x7F800000
        mantissa = fp32_bits & 0x007FFFFF
        if exponent == 0x7F800000:
            # Preserve infinities.  Preserve a NaN payload's high bits and
            # force the quiet-NaN bit so truncation cannot turn it into inf.
            result = (fp32_bits >> 16) & 0xFFFF
            if mantissa != 0:
                result |= 0x0040
            return result
        # Round-to-nearest-even, matching the normal FP32 -> BF16 cast.
        rounded = fp32_bits + 0x7FFF + ((fp32_bits >> 16) & 1)
        return (rounded >> 16) & 0xFFFF

    # ctypes has already applied the declared integer width/sign conversion.
    # Sign-extending a negative value into the 64-bit carrier is compatible
    # with from_u64<T>(), which consumes only the low sizeof(T) bytes.
    return int(scalar) & 0xFFFFFFFFFFFFFFFF


def _safe_op_name(program: CompiledProgram, callable_id: int) -> str:
    leaf = re.sub(r"[^0-9A-Za-z_]+", "_", program.output_dir.name).strip("_")
    if not leaf:
        leaf = "operator"
    return f"pypto_l1_{callable_id}_{leaf[:80]}"


def _tensor_device_index(tensor: torch.Tensor) -> int:
    index = tensor.device.index
    if index is None:
        # torch_npu tensors normally carry an explicit index, but get_device()
        # is the authoritative fallback for a device object rendered as "npu".
        index = tensor.get_device()
    return int(index)


def _tensor_device_type(tensor: torch.Tensor) -> str:
    return str(tensor.device.type)


def _normalize_outputs(out: object, count: int) -> list[torch.Tensor]:
    if count == 1 and isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        values = list(out)
        if len(values) != count:
            raise TypeError(f"out expects {count} tensor(s), got {len(values)}")
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError("out must contain only torch.Tensor values")
        return values
    raise TypeError(f"out expects {count} NPU tensor(s)")


class L1Context:
    """One explicitly owned, non-concurrent L1 context bound to one runtime."""

    # Construction is one validation/ownership transaction; keeping its
    # branches together makes the pre-init versus retained-owner boundary auditable.
    def __init__(  # noqa: PLR0912
        self,
        programs: Sequence[CompiledProgram],
        *,
        device: int,
        config: L1Config | None = None,
    ) -> None:
        self._owner_thread = threading.get_ident()
        if isinstance(device, bool) or not isinstance(device, int):
            raise TypeError(f"device must be a non-bool integer, got {type(device).__name__}")
        self._device = device
        if self._device < 0:
            raise ValueError(f"device must be non-negative, got {device}")
        self._config = config if config is not None else L1Config()
        if not isinstance(self._config, L1Config):
            raise TypeError("config must be L1Config or None")

        declared = list(programs)
        if not declared:
            raise ValueError("pypto_init requires at least one CompiledProgram")
        if any(not isinstance(program, CompiledProgram) for program in declared):
            raise TypeError("pypto_init programs must contain only CompiledProgram values")

        # Object identity, not output-path equality, defines membership.  The
        # same object may appear twice in a caller-produced list; prepare it
        # once.  Distinct objects remain distinct declared operators even if
        # they happen to point at the same artifact directory.
        unique_programs: list[CompiledProgram] = []
        seen_ids: set[int] = set()
        for program in declared:
            if id(program) not in seen_ids:
                seen_ids.add(id(program))
                unique_programs.append(program)
        if len(unique_programs) > MAX_REGISTERED_CALLABLE_IDS:
            raise ValueError(
                "PyPTO L1 callable capacity exceeded: "
                f"got {len(unique_programs)} distinct programs, limit={MAX_REGISTERED_CALLABLE_IDS}"
            )

        self._states: list[_OperatorState] = []
        self._states_by_identity: dict[int, _OperatorState] = {}
        self._prepared = False
        self._closed = False
        self._init_failed = False
        self._worker: Any = None
        self._torch_npu = _load_torch_npu()
        self._task_queue_adapter: Any = None

        # The default path cannot become usable without the optional adapter.
        # Import and ABI/version-check it before native context creation so a
        # packaging mismatch never leaves a half-usable borrowed device owner.
        if self._config.use_task_queue:
            self._task_queue_adapter = _load_task_queue_adapter()
            _validate_task_queue_adapter(self._task_queue_adapter, self._torch_npu)

        # Validate the borrowed device before native allocation or DSO init.
        self._check_current_device()
        platform = unique_programs[0].platform
        if platform not in ("a2a3", "a5"):
            raise ValueError(f"PyPTO L1 requires onboard platform 'a2a3' or 'a5', got {platform!r}")
        runtime = validate_runtime_name(
            unique_programs[0].runtime_name,
            parameter="program runtime",
        )
        self._runtime = runtime

        baked_aicpu_counts: set[int] = set()
        for callable_id, program in enumerate(unique_programs):
            if program.platform != platform:
                raise ValueError(
                    f"all L1 programs must target one platform; got {platform!r} and {program.platform!r}"
                )
            if _contains_distributed_types(program):
                raise ValueError("PyPTO L1 v1 does not support CommCtx/DistributedTensor programs")
            program_runtime = validate_runtime_name(
                program.runtime_name,
                parameter=f"program {program.output_dir} runtime",
            )
            if program_runtime != runtime:
                raise ValueError(
                    f"all L1 programs must use the same runtime; got {runtime!r} and {program_runtime!r}"
                )

            # Loading is host-only compile/assembly work.  Do it for every
            # program before initializing the borrowed native context so a bad
            # artifact cannot leave a partially initialized device owner.
            chip_callable = program.chip_callable
            if bool(program.runtime_config.get("enable_sdma", False)):
                raise ValueError("PyPTO L1 v1 does not support SDMA workspace programs")

            param_infos, output_indices, _ = program._get_metadata()
            tensor_count = sum(info.shape is not None for info in param_infos)
            scalar_count = len(param_infos) - tensor_count
            if tensor_count > CHIP_MAX_TENSOR_ARGS or scalar_count > CHIP_MAX_SCALAR_ARGS:
                raise ValueError(
                    "program exceeds the L1 task-argument ABI capacity: "
                    f"tensors={tensor_count}/{CHIP_MAX_TENSOR_ARGS}, "
                    f"scalars={scalar_count}/{CHIP_MAX_SCALAR_ARGS}"
                )
            for info in param_infos:
                if info.shape is not None:
                    if (
                        not info.shape
                        or len(info.shape) > MAX_TENSOR_DIMS
                        or any(dim <= 0 for dim in info.shape)
                    ):
                        raise ValueError(
                            "PyPTO L1 v1 requires rank in "
                            f"[1, {MAX_TENSOR_DIMS}] and a positive static shape "
                            f"for {info.name!r}; got {info.shape}"
                        )
                    if any(dim > _UINT32_MAX for dim in info.shape):
                        raise ValueError(f"shape for {info.name!r} exceeds the L1 uint32 tensor ABI")
                    torch_dtype = _to_torch_dtype(info.dtype)
                    if torch_dtype is None:
                        raise ValueError(f"L1 tensor {info.name!r} uses unsupported dtype {info.dtype}")
                    try:
                        torch_dtype_to_datatype(torch_dtype)
                    except KeyError as exc:
                        raise ValueError(
                            f"L1 tensor {info.name!r} dtype {info.dtype} has no simpler runtime ABI mapping"
                        ) from exc
                elif info.direction == ParamDirection.Out:
                    raise ValueError(
                        f"PyPTO L1 v1 does not support pure-Out scalar parameter {info.name!r}; "
                        "scalar outputs need an explicit future ABI"
                    )
                elif str(info.dtype) not in _SUPPORTED_L1_SCALAR_DTYPES:
                    raise ValueError(
                        f"PyPTO L1 scalar {info.name!r} uses unsupported dtype {info.dtype}; "
                        f"supported dtypes are {sorted(_SUPPORTED_L1_SCALAR_DTYPES)}"
                    )

            _validate_final_callable_signature(chip_callable, param_infos)

            baked_count = program.runtime_config.get("aicpu_thread_num")
            if baked_count is not None:
                baked_aicpu_counts.add(int(baked_count))
            state = _OperatorState(
                program=program,
                callable_id=callable_id,
                chip_callable=chip_callable,
                param_infos=list(param_infos),
                output_indices=tuple(int(index) for index in output_indices),
                op_name=_safe_op_name(program, callable_id),
            )
            self._states.append(state)
            self._states_by_identity[id(program)] = state

        if self._config.aicpu_thread_num is None and len(baked_aicpu_counts) > 1:
            raise ValueError(
                "declared L1 programs use different baked aicpu_thread_num values; "
                "provide one explicit L1Config.aicpu_thread_num"
            )
        resolved_aicpu_count = self._config.aicpu_thread_num
        if resolved_aicpu_count is None and baked_aicpu_counts:
            resolved_aicpu_count = next(iter(baked_aicpu_counts))

        run_config = RunConfig(
            platform=platform,
            device_id=self._device,
            aicpu_thread_num=resolved_aicpu_count,
            ring_task_window=self._config.ring_task_window,
            ring_heap=self._config.ring_heap,
            ring_dep_pool=self._config.ring_dep_pool,
            runtime=runtime,
        )
        call_config = unique_programs[0].build_call_config(
            run_config,
            aicpu_thread_num=resolved_aicpu_count,
        )
        bins = _build_runtime_binaries(platform, runtime)
        worker = _make_native_worker()
        self._worker = worker
        try:
            worker.init_l1(self._device, bins, call_config)
        except BaseException as exc:
            if bool(getattr(worker, "initialized", False)):
                # Native rollback failed and deliberately retained the L1
                # context/DSO.  Expose this exact owner to the caller for an
                # explicit close retry; it is forbidden from prepare/launch.
                self._init_failed = True
                raise L1InitializationError(
                    f"PyPTO L1 initialization failed and retained cleanup ownership: {exc}", self
                ) from exc
            self._worker = None
            self._closed = True
            raise

    @property
    def device(self) -> int:
        return self._device

    @property
    def runtime(self) -> str:
        """Runtime implementation shared by every declared operator."""
        return self._runtime

    @property
    def prepared(self) -> bool:
        """Whether all prepare calls were successfully enqueued, not completed."""
        return self._prepared

    @property
    def closed(self) -> bool:
        return self._closed

    def _check_owner(self) -> None:
        current = threading.get_ident()
        if current != self._owner_thread:
            raise RuntimeError(f"L1Context is thread-affine: owner={self._owner_thread}, caller={current}")

    def _check_open(self) -> None:
        self._check_owner()
        if self._closed or self._worker is None:
            raise RuntimeError("L1Context is closed")
        if self._init_failed:
            raise RuntimeError("L1Context initialization failed; only close() is permitted")

    def _check_current_device(self) -> None:
        current = _current_device(self._torch_npu)
        if current != self._device:
            raise RuntimeError(
                f"PyPTO L1 borrows current NPU device {self._device}, "
                f"but torch_npu current device is {current}; "
                "set the intended device before pypto_init/call (PyPTO will not set it for you)"
            )

    def operator(self, program: CompiledProgram) -> L1Operator:
        self._check_open()
        state = self._states_by_identity.get(id(program))
        if state is None or state.program is not program:
            raise KeyError("program was not declared in this pypto_init(programs=[...]) context")
        return L1Operator(self, state)

    def _enqueue(self, queue_call: Any, tensors: list[torch.Tensor], op_name: str, direct: Any) -> None:
        self._check_current_device()
        if self._config.use_task_queue:
            assert self._task_queue_adapter is not None
            self._task_queue_adapter.enqueue(queue_call, tensors, self._device, op_name)
            return

        stream = _current_stream(self._torch_npu, self._device)
        raw_stream = int(stream.npu_stream)
        if raw_stream == 0:
            raise RuntimeError("torch_npu returned a null current raw stream")
        # The taskQueue adapter performs the stronger C++ allocator record.
        # Preserve equivalent caller-stream storage lifetime in debug-direct
        # mode where torch exposes Tensor.record_stream.
        for tensor in tensors:
            tensor.record_stream(stream)
        direct(raw_stream)

    def prepare(self) -> None:
        """Prepare every declared callable in deterministic order.

        This method is idempotent and asynchronous.  It does not synchronize;
        ACLGraph callers must execute a warmup and then synchronize externally.
        """
        self._check_open()
        if self._prepared:
            return
        assert self._worker is not None
        for state in self._states:
            queue_call = (
                self._worker.l1_make_prepare_queue_call(state.callable_id, state.chip_callable)
                if self._config.use_task_queue
                else None
            )
            self._enqueue(
                queue_call,
                [],
                f"{state.op_name}_prepare",
                lambda raw_stream, state=state: self._worker.l1_prepare_callable(
                    state.callable_id, state.chip_callable, raw_stream
                ),
            )
        self._prepared = True

    # This is the single tensors-first ABI packing transaction; the branches
    # deliberately mirror tensor, scalar, input/inout and explicit-out cases.
    def _build_args(  # noqa: PLR0912
        self,
        state: _OperatorState,
        args: tuple[Any, ...],
        out: object,
    ) -> tuple[
        ChipStorageTaskArgs,
        list[torch.Tensor],
        tuple[tuple[tuple[int, ...], torch.dtype, tuple[int, ...]], ...],
        object,
    ]:
        output_set = set(state.output_indices)
        input_indices = [index for index in range(len(state.param_infos)) if index not in output_set]
        if len(args) != len(input_indices):
            raise TypeError(
                f"{state.op_name} expects {len(input_indices)} positional input/inout argument(s), "
                f"got {len(args)}; parameters={[info.name for info in state.param_infos]}"
            )

        if state.output_indices:
            if out is _MISSING:
                names = [state.param_infos[index].name for index in state.output_indices]
                raise TypeError(f"PyPTO L1 v1 requires explicit out= for output parameter(s) {names}")
            outputs = _normalize_outputs(out, len(state.output_indices))
            returned: object = outputs[0] if len(outputs) == 1 else tuple(outputs)
        else:
            if out is not _MISSING:
                raise TypeError("out= is only valid when the program has pure Out parameters")
            outputs = []
            returned = None

        full_args: list[Any] = [None] * len(state.param_infos)
        for index, value in zip(input_indices, args, strict=True):
            full_args[index] = value
        for index, value in zip(state.output_indices, outputs, strict=True):
            full_args[index] = value

        coerced, _ = _coerce_args(
            tuple(full_args),
            state.param_infos,
            [],
            [],
            caller_name=state.op_name,
        )

        tensor_values: list[torch.Tensor] = []
        tensor_metadata: list[tuple[tuple[int, ...], torch.dtype, tuple[int, ...]]] = []
        packed = ChipStorageTaskArgs()
        # Runtime ABI is tensors-first even when scalar/tensor parameters are
        # interleaved in the original Python signature.
        for info, value in zip(state.param_infos, coerced, strict=True):
            if info.shape is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"L1 tensor parameter {info.name!r} requires torch.Tensor, got {type(value).__name__}"
                )
            if _tensor_device_type(value) != "npu":
                raise ValueError(f"L1 tensor {info.name!r} must be on NPU, got device={value.device}")
            actual_device = _tensor_device_index(value)
            if actual_device != self._device:
                raise ValueError(
                    f"L1 tensor {info.name!r} is on device {actual_device}, expected {self._device}"
                )
            if torch.is_grad_enabled() and value.requires_grad:
                raise RuntimeError(
                    "PyPTO L1 v1 is inference-only and does not register autograd/alias semantics; "
                    f"tensor {info.name!r} has requires_grad=True"
                )
            actual_shape = tuple(int(dim) for dim in value.shape)
            expected_shape = tuple(int(dim) for dim in info.shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"L1 tensor {info.name!r} expects static shape {expected_shape}, got {actual_shape}"
                )
            expected_dtype = _to_torch_dtype(info.dtype)
            if expected_dtype is None:
                raise ValueError(f"L1 tensor {info.name!r} uses unsupported dtype {info.dtype}")
            if value.dtype != expected_dtype:
                raise ValueError(f"L1 tensor {info.name!r} expects dtype {expected_dtype}, got {value.dtype}")
            try:
                runtime_dtype = torch_dtype_to_datatype(value.dtype)
            except KeyError as exc:
                raise ValueError(
                    f"L1 tensor {info.name!r} dtype {value.dtype} has no simpler runtime ABI mapping"
                ) from exc
            strides = tuple(int(stride) for stride in value.stride())
            if any(stride <= 0 or stride > _UINT32_MAX for stride in strides):
                raise ValueError(f"L1 tensor {info.name!r} requires positive uint32 strides, got {strides}")
            packed.add_tensor(
                ChipTensor.make_strided(
                    data=value.data_ptr(),
                    shapes=actual_shape,
                    strides=strides,
                    dtype=runtime_dtype,
                    child_memory=True,
                )
            )
            tensor_values.append(value)
            tensor_metadata.append((actual_shape, value.dtype, strides))

        for info, value in zip(state.param_infos, coerced, strict=True):
            if info.shape is not None:
                continue
            if not isinstance(value, ctypes._SimpleCData):
                raise TypeError(f"L1 scalar parameter {info.name!r} was not coerced to a ctypes scalar")
            packed.add_scalar(_pack_l1_scalar(value, info.dtype))
        return packed, tensor_values, tuple(tensor_metadata), returned

    def _invoke(
        self, state: _OperatorState, args: tuple[Any, ...], out: object, *, explicit_warmup: bool
    ) -> object:
        self._check_open()
        # Finish all pure host validation before eager auto-prepare. A malformed
        # first call must not mutate the taskQueue/native prepare state.
        packed, tensors, tensor_metadata, returned = self._build_args(state, args, out)
        if state.bound_tensor_metadata is not None and tensor_metadata != state.bound_tensor_metadata:
            raise ValueError(
                f"{state.op_name} tensor layout changed after first successful enqueue: "
                f"expected={state.bound_tensor_metadata!r}, got={tensor_metadata!r}"
            )
        if not self._prepared:
            # Eager convenience.  Graph users are required to call prepare()
            # before capture; PyPTO deliberately does not query capture state.
            self.prepare()
        assert self._worker is not None
        queue_call = (
            self._worker.l1_make_launch_queue_call(state.callable_id, packed)
            if self._config.use_task_queue
            else None
        )
        self._enqueue(
            queue_call,
            tensors,
            state.op_name,
            lambda raw_stream: self._worker.l1_launch(state.callable_id, packed, raw_stream),
        )
        if state.bound_tensor_metadata is None:
            state.bound_tensor_metadata = tensor_metadata
        # This means "one ordinary invocation was successfully enqueued", not
        # "device work completed".  The caller owns warmup synchronization.
        if explicit_warmup or not state.warmed:
            state.warmed = True
        return returned

    def close(self) -> None:
        """Release PyPTO-owned L1 resources without synchronizing the device.

        The caller must first destroy every ACLGraph that can replay this
        context and externally prove device quiescence.  On native teardown
        failure the context remains open and ownership is retained for retry.
        """
        self._check_owner()
        if self._closed:
            return
        self._check_current_device()
        assert self._worker is not None
        self._worker.finalize()
        self._closed = True
        self._init_failed = False
        self._worker = None

    def __del__(self) -> None:
        if getattr(self, "_worker", None) is not None and not getattr(self, "_closed", True):
            warnings.warn(
                "L1Context was not explicitly closed; native L1 resources are intentionally pinned/leaked "
                "rather than risking invalidation of a live ACLGraph",
                ResourceWarning,
                stacklevel=2,
            )


class L1Operator:
    """AscendC-like callable view of one program in an :class:`L1Context`."""

    def __init__(self, context: L1Context, state: _OperatorState) -> None:
        self._context = context
        self._state = state

    @property
    def prepared(self) -> bool:
        """Whether the owning context's prepare calls were successfully enqueued."""
        return self._context.prepared

    @property
    def warmed(self) -> bool:
        """Whether one invocation was successfully enqueued, not device-complete."""
        return self._state.warmed

    def prepare(self) -> None:
        """Prepare all programs declared in the owning context."""
        self._context.prepare()

    def warmup(self, *args: Any, out: object = _MISSING) -> object:
        """Enqueue one explicit warmup; synchronization remains caller-owned."""
        return self._context._invoke(self._state, args, out, explicit_warmup=True)

    def __call__(self, *args: Any, out: object = _MISSING) -> object:
        """Enqueue one asynchronous L1 operator invocation."""
        return self._context._invoke(self._state, args, out, explicit_warmup=False)


def pypto_init(
    *,
    programs: Sequence[CompiledProgram],
    device: int,
    config: L1Config | None = None,
) -> L1Context:
    """Create one borrowed-device L1 context for the declared programs.

    ``device`` is mandatory and must already be the torch_npu current device;
    PyPTO never switches devices on behalf of the caller.

    Every program must target the same onboard platform and the same Simpler
    runtime. Compile them with ``RunConfig(runtime=...)`` or
    ``ir.compile(..., runtime=...)`` before constructing the context.
    """
    return L1Context(programs, device=device, config=config)


__all__ = ["L1Config", "L1Context", "L1InitializationError", "L1Operator", "pypto_init"]
