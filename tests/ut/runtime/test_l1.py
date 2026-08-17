# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""No-hardware contract tests for the PyPTO L1/ACLGraph-facing API."""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from pypto.ir.compiled_program import CompiledProgram, _ParamInfo
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection

l1_mod = importlib.import_module("pypto.runtime.l1")


class _FakeNpuApi:
    def __init__(self, device: int) -> None:
        self.device = device
        self.current_stream_calls = 0
        self.stream: Any = None

    def current_device(self) -> int:
        return self.device

    def current_stream(self, _device: int):
        self.current_stream_calls += 1
        if self.stream is None:
            raise AssertionError("taskQueue path must not ask Python for a raw stream")
        return self.stream


class _FakeWorker:
    def __init__(self) -> None:
        self.init_calls: list[tuple[Any, ...]] = []
        self.prepare_capsules: list[int] = []
        self.launch_args: list[tuple[int, Any]] = []
        self.finalize_calls = 0
        self.fail_finalize_once = False
        self.initialized = False
        self.fail_init_with_retained_owner = False
        self.init_exception: BaseException | None = None
        self.direct_prepare_calls: list[tuple[Any, ...]] = []
        self.direct_launch_calls: list[tuple[Any, ...]] = []

    def init_l1(self, *args: Any) -> None:
        self.init_calls.append(args)
        self.initialized = True
        if self.fail_init_with_retained_owner:
            raise RuntimeError("injected init cleanup failure; explicit finalize required")
        if self.init_exception is not None:
            raise self.init_exception

    def l1_make_prepare_queue_call(self, callable_id: int, _callable: Any):
        self.prepare_capsules.append(callable_id)
        return ("prepare", callable_id)

    def l1_make_launch_queue_call(self, callable_id: int, args: Any):
        self.launch_args.append((callable_id, args))
        return ("launch", callable_id, len(self.launch_args))

    def l1_prepare_callable(self, *args: Any) -> None:
        self.direct_prepare_calls.append(args)

    def l1_launch(self, *args: Any) -> None:
        self.direct_launch_calls.append(args)

    def finalize(self) -> None:
        self.finalize_calls += 1
        if self.fail_finalize_once and self.finalize_calls == 1:
            raise RuntimeError("injected close failure")
        self.initialized = False


class _FakeAdapter:
    QUEUE_CALL_ABI_VERSION = 1
    BUILD_TORCH_VERSION = "unknown"
    BUILD_TORCH_NPU_VERSION = "unknown"

    def __init__(self) -> None:
        self.calls: list[tuple[Any, list[torch.Tensor], int, str]] = []
        self.fail_launch_enqueue_once = False

    def enqueue(self, capsule: Any, tensors: list[torch.Tensor], device: int, op_name: str) -> None:
        if self.fail_launch_enqueue_once and capsule[0] == "launch":
            self.fail_launch_enqueue_once = False
            raise RuntimeError("injected taskQueue launch enqueue failure")
        self.calls.append((capsule, list(tensors), device, op_name))


class _FakeCallable:
    def __init__(self, param_infos: list[_ParamInfo]) -> None:
        tensor_infos = [info for info in param_infos if info.shape is not None]
        self.sig_count = len(tensor_infos)
        self.scalar_count = len(param_infos) - len(tensor_infos)
        self._signature = [l1_mod._PARAM_TO_RUNTIME_DIRECTION[info.direction] for info in tensor_infos]

    def sig(self, index: int):
        return self._signature[index]


def _compiled(
    tmp_path: Path,
    name: str,
    *,
    platform: str = "a2a3",
    runtime_name: str = "tensormap_and_ringbuffer",
    runtime_config: dict[str, Any] | None = None,
    param_infos: list[_ParamInfo] | None = None,
    output_indices: list[int] | None = None,
) -> CompiledProgram:
    program = object.__new__(CompiledProgram)
    program._program = cast(Any, SimpleNamespace(functions={}))
    program._output_dir = tmp_path / name
    program._platform = platform
    resolved_param_infos = param_infos or [
        _ParamInfo("x", ParamDirection.In, [2, 3], DataType.FP32),
        _ParamInfo("scale", ParamDirection.In, None, DataType.FP32),
        _ParamInfo("y", ParamDirection.Out, [2, 3], DataType.FP32),
    ]
    program._chip_callable = _FakeCallable(resolved_param_infos)
    program._runtime_name = runtime_name
    program._runtime_config = dict(runtime_config or {})
    program._sub_chip_dirs = {}
    program._param_infos = resolved_param_infos
    program._output_indices = [2] if output_indices is None else list(output_indices)
    program._return_types = []
    return program


@pytest.fixture
def l1_fakes(monkeypatch: pytest.MonkeyPatch):
    npu_api = _FakeNpuApi(device=1)
    worker = _FakeWorker()
    adapter = _FakeAdapter()
    monkeypatch.setattr(l1_mod, "_load_torch_npu", lambda: SimpleNamespace(npu=npu_api))
    monkeypatch.setattr(l1_mod, "_build_runtime_binaries", lambda _platform: object())
    monkeypatch.setattr(l1_mod, "_make_native_worker", lambda: worker)
    monkeypatch.setattr(l1_mod, "_load_task_queue_adapter", lambda: adapter)
    # Real CPU tensors give us truthful shapes/strides/data_ptr without NPU
    # access. These two seams isolate only the device classification query.
    monkeypatch.setattr(l1_mod, "_tensor_device_type", lambda _tensor: "npu")
    monkeypatch.setattr(l1_mod, "_tensor_device_index", lambda _tensor: 1)
    return npu_api, worker, adapter


def test_context_prepares_all_declared_programs_once(tmp_path: Path, l1_fakes) -> None:
    npu_api, worker, adapter = l1_fakes
    first = _compiled(tmp_path, "first")
    second = _compiled(tmp_path, "second")
    ctx = l1_mod.pypto_init(programs=[first, second, first], device=1)

    assert len(worker.init_calls) == 1
    first_op = ctx.operator(first)
    first_op.prepare()
    ctx.prepare()

    assert worker.prepare_capsules == [0, 1]
    assert [call[0] for call in adapter.calls] == [("prepare", 0), ("prepare", 1)]
    assert all(call[1] == [] and call[2] == 1 for call in adapter.calls)
    assert npu_api.current_stream_calls == 0
    assert ctx.prepared and first_op.prepared
    ctx.close()


def test_context_rejects_callable_capacity_before_native_init(tmp_path: Path, l1_fakes) -> None:
    _, worker, _ = l1_fakes
    programs = [
        _compiled(tmp_path, f"program_{index}") for index in range(l1_mod.MAX_REGISTERED_CALLABLE_IDS + 1)
    ]

    with pytest.raises(ValueError, match="callable capacity exceeded"):
        l1_mod.pypto_init(programs=programs, device=1)

    assert worker.init_calls == []


def test_warmup_packs_tensors_then_scalar_and_returns_explicit_out(tmp_path: Path, l1_fakes) -> None:
    npu_api, worker, adapter = l1_fakes
    program = _compiled(tmp_path, "scale")
    ctx = l1_mod.pypto_init(programs=[program], device=1)
    op = ctx.operator(program)
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    y = torch.zeros((2, 3), dtype=torch.float32)

    returned = op.warmup(x, 2.5, out=y)

    assert returned is y
    assert op.warmed and ctx.prepared
    assert worker.prepare_capsules == [0]
    assert len(worker.launch_args) == 1
    packed = worker.launch_args[0][1]
    assert packed.tensor_count() == 2
    assert packed.scalar_count() == 1
    assert packed.tensor(0).data == x.data_ptr()
    assert packed.tensor(1).data == y.data_ptr()
    assert tuple(packed.tensor(0).strides) == x.stride()
    assert adapter.calls[-1][1] == [x, y]
    assert npu_api.current_stream_calls == 0
    ctx.close()


@pytest.mark.parametrize(
    ("dtype", "value", "expected_bits"),
    [
        (DataType.FP16, 1.0, 0x3C00),
        (DataType.FP16, 1.1, 0x3C66),
        (DataType.BF16, 1.0, 0x3F80),
        (DataType.BF16, 1.1, 0x3F8D),
        (DataType.FP32, 1.0, 0x3F800000),
        ("fp64", 1.0, 0x3FF0000000000000),
        (DataType.INT8, -2, 0xFFFFFFFFFFFFFFFE),
        (DataType.INT16, -2, 0xFFFFFFFFFFFFFFFE),
        (DataType.INT32, -2, 0xFFFFFFFFFFFFFFFE),
        (DataType.INT64, -2, 0xFFFFFFFFFFFFFFFE),
        (DataType.UINT8, 0xFE, 0xFE),
        (DataType.UINT16, 0xFFFE, 0xFFFE),
        (DataType.UINT32, 0xFFFFFFFE, 0xFFFFFFFE),
        (DataType.UINT64, 0xFFFFFFFFFFFFFFFE, 0xFFFFFFFFFFFFFFFE),
        (DataType.BOOL, True, 1),
        (DataType.INDEX, -2, 0xFFFFFFFFFFFFFFFE),
    ],
)
def test_scalar_is_packed_by_declared_device_dtype(
    tmp_path: Path,
    l1_fakes,
    dtype: Any,
    value: object,
    expected_bits: int,
) -> None:
    _, worker, _ = l1_fakes
    param_infos = [
        _ParamInfo("x", ParamDirection.In, [2, 3], DataType.FP32),
        _ParamInfo("scalar", ParamDirection.In, None, dtype),
        _ParamInfo("y", ParamDirection.Out, [2, 3], DataType.FP32),
    ]
    program = _compiled(tmp_path, f"scalar_{dtype}", param_infos=param_infos)
    ctx = l1_mod.pypto_init(programs=[program], device=1)
    x = torch.ones((2, 3), dtype=torch.float32)
    y = torch.empty((2, 3), dtype=torch.float32)

    ctx.operator(program).warmup(x, value, out=y)

    packed = worker.launch_args[-1][1]
    assert packed.scalar(0) == expected_bits
    ctx.close()


@pytest.mark.parametrize("dtype", [DataType.FP4, DataType.FP8E4M3FN, DataType.HF8])
def test_unsupported_scalar_dtype_is_rejected_before_native_init(
    tmp_path: Path,
    l1_fakes,
    dtype: DataType,
) -> None:
    _, worker, _ = l1_fakes
    program = _compiled(
        tmp_path,
        f"unsupported_scalar_{dtype}",
        param_infos=[
            _ParamInfo("x", ParamDirection.In, [2, 3], DataType.FP32),
            _ParamInfo("scalar", ParamDirection.In, None, dtype),
            _ParamInfo("y", ParamDirection.Out, [2, 3], DataType.FP32),
        ],
    )

    with pytest.raises(ValueError, match="scalar .* uses unsupported dtype"):
        l1_mod.pypto_init(programs=[program], device=1)

    assert worker.init_calls == []


def test_first_eager_call_is_auto_prepare_warmup_but_out_is_mandatory(tmp_path: Path, l1_fakes) -> None:
    _, worker, adapter = l1_fakes
    program = _compiled(tmp_path, "auto")
    ctx = l1_mod.pypto_init(programs=[program], device=1)
    op = ctx.operator(program)
    x = torch.ones((2, 3), dtype=torch.float32)
    y = torch.empty((2, 3), dtype=torch.float32)

    with pytest.raises(TypeError, match="requires explicit out="):
        op(x, 1.0)
    assert not ctx.prepared
    assert adapter.calls == []

    assert op(x, 1.0, out=y) is y
    assert op.warmed
    assert worker.prepare_capsules == [0]
    assert len([call for call in adapter.calls if call[0][0] == "launch"]) == 1
    ctx.close()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda tensor: tensor.reshape(3, 2), "expects static shape"),
        (lambda tensor: tensor.to(torch.int32), "expects dtype"),
        (lambda tensor: tensor[:1].expand(2, 3), "positive uint32 strides"),
    ],
)
def test_tensor_metadata_is_validated_before_enqueue(
    tmp_path: Path,
    l1_fakes,
    mutation,
    message: str,
) -> None:
    _, _, adapter = l1_fakes
    program = _compiled(tmp_path, "metadata")
    ctx = l1_mod.pypto_init(programs=[program], device=1)
    op = ctx.operator(program)
    x = mutation(torch.ones((2, 3), dtype=torch.float32))
    y = torch.empty((2, 3), dtype=torch.float32)
    before = len(adapter.calls)

    with pytest.raises(ValueError, match=message):
        op.warmup(x, 1.0, out=y)

    # Pure host validation precedes eager auto-prepare, so a bad invocation
    # must not enqueue either prepare or launch.
    assert len(adapter.calls) == before
    assert not ctx.prepared
    ctx.close()


def test_wrong_current_device_fails_before_native_worker_creation(
    tmp_path: Path,
    l1_fakes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    npu_api, worker, _ = l1_fakes
    npu_api.device = 0
    program = _compiled(tmp_path, "wrong_device")

    with pytest.raises(RuntimeError, match="current NPU device 1"):
        l1_mod.pypto_init(programs=[program], device=1)
    assert worker.init_calls == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"platform": "a2a3sim"}, "requires onboard platform"),
        ({"runtime_name": "host_build_graph"}, "requires runtime"),
        ({"runtime_config": {"enable_sdma": True}}, "does not support SDMA"),
        (
            {
                "param_infos": [
                    _ParamInfo("x", ParamDirection.In, [-1, 3], DataType.FP32),
                ],
                "output_indices": [],
            },
            "positive static shape",
        ),
    ],
)
def test_v1_eligibility_rejections(tmp_path: Path, l1_fakes, kwargs: dict[str, Any], message: str) -> None:
    program = _compiled(tmp_path, "unsupported", **kwargs)
    with pytest.raises(ValueError, match=message):
        l1_mod.pypto_init(programs=[program], device=1)


def test_context_is_thread_affine_and_close_is_retryable(tmp_path: Path, l1_fakes) -> None:
    _, worker, _ = l1_fakes
    program = _compiled(tmp_path, "lifecycle")
    ctx = l1_mod.pypto_init(programs=[program], device=1)
    errors: list[BaseException] = []

    thread = threading.Thread(target=lambda: _capture_error(lambda: ctx.operator(program), errors))
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert "thread-affine" in str(errors[0])

    worker.fail_finalize_once = True
    with pytest.raises(RuntimeError, match="injected close failure"):
        ctx.close()
    assert not ctx.closed
    ctx.close()
    ctx.close()
    assert ctx.closed
    assert worker.finalize_calls == 2


def test_first_successful_enqueue_binds_layout_but_not_tensor_address(tmp_path: Path, l1_fakes) -> None:
    _, worker, adapter = l1_fakes
    program = _compiled(tmp_path, "layout")
    ctx = l1_mod.pypto_init(programs=[program], device=1)
    op = ctx.operator(program)
    x = torch.ones((2, 3), dtype=torch.float32)
    y = torch.empty((2, 3), dtype=torch.float32)
    op.warmup(x, 1.0, out=y)

    x_same_layout = torch.zeros((2, 3), dtype=torch.float32)
    y_same_layout = torch.empty((2, 3), dtype=torch.float32)
    op(x_same_layout, 2.0, out=y_same_layout)
    assert len(worker.launch_args) == 2

    x_different_layout = torch.empty_strided((2, 3), (1, 2), dtype=torch.float32)
    before = len(adapter.calls)
    with pytest.raises(ValueError, match="tensor layout changed"):
        op(x_different_layout, 3.0, out=y_same_layout)
    assert len(adapter.calls) == before
    assert len(worker.launch_args) == 2
    ctx.close()


def test_failed_first_launch_enqueue_does_not_bind_layout_or_warmed(tmp_path: Path, l1_fakes) -> None:
    _, worker, adapter = l1_fakes
    program = _compiled(tmp_path, "failed_first_launch")
    ctx = l1_mod.pypto_init(programs=[program], device=1)
    op = ctx.operator(program)
    x_contiguous = torch.ones((2, 3), dtype=torch.float32)
    x_strided = torch.empty_strided((2, 3), (1, 2), dtype=torch.float32)
    y = torch.empty((2, 3), dtype=torch.float32)
    adapter.fail_launch_enqueue_once = True

    with pytest.raises(RuntimeError, match="launch enqueue failure"):
        op.warmup(x_contiguous, 1.0, out=y)

    # Context-wide prepare was accepted, but the failed launch must not commit
    # per-operator warmup state or make its candidate layout authoritative.
    assert ctx.prepared
    assert not op.warmed
    assert [call[0][0] for call in adapter.calls] == ["prepare"]

    op.warmup(x_strided, 2.0, out=y)
    assert op.warmed
    assert [call[0][0] for call in adapter.calls] == ["prepare", "launch"]
    assert len(worker.launch_args) == 2

    # The first accepted launch, not the failed attempt, fixed the layout.
    with pytest.raises(ValueError, match="tensor layout changed"):
        op(x_contiguous, 3.0, out=y)
    assert len(worker.launch_args) == 2
    ctx.close()


def test_first_successful_enqueue_binds_output_layout(tmp_path: Path, l1_fakes) -> None:
    _, worker, adapter = l1_fakes
    program = _compiled(tmp_path, "output_layout")
    ctx = l1_mod.pypto_init(programs=[program], device=1)
    op = ctx.operator(program)
    x = torch.ones((2, 3), dtype=torch.float32)
    y = torch.empty((2, 3), dtype=torch.float32)
    op.warmup(x, 1.0, out=y)

    y_different_layout = torch.empty_strided((2, 3), (1, 2), dtype=torch.float32)
    before = len(adapter.calls)
    with pytest.raises(ValueError, match="tensor layout changed"):
        op(x, 2.0, out=y_different_layout)

    assert len(adapter.calls) == before
    assert len(worker.launch_args) == 1
    ctx.close()


def test_task_queue_adapter_failure_precedes_native_init(
    tmp_path: Path,
    l1_fakes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, worker, _ = l1_fakes
    program = _compiled(tmp_path, "missing_adapter")

    def fail_adapter():
        raise RuntimeError("adapter unavailable")

    monkeypatch.setattr(l1_mod, "_load_task_queue_adapter", fail_adapter)
    with pytest.raises(RuntimeError, match="adapter unavailable"):
        l1_mod.pypto_init(programs=[program], device=1)
    assert worker.init_calls == []


@pytest.mark.parametrize(
    ("attribute", "bad_value", "runtime_npu_version", "message"),
    [
        ("QUEUE_CALL_ABI_VERSION", 999, None, "queue-call ABI"),
        ("BUILD_TORCH_VERSION", "0.invalid", None, "different framework versions"),
        ("BUILD_TORCH_NPU_VERSION", "build.invalid", "runtime.valid", "different framework versions"),
    ],
)
def test_task_queue_adapter_abi_and_framework_mismatch_precede_native_init(
    tmp_path: Path,
    l1_fakes,
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    bad_value: object,
    runtime_npu_version: str | None,
    message: str,
) -> None:
    npu_api, worker, adapter = l1_fakes
    setattr(adapter, attribute, bad_value)
    if runtime_npu_version is not None:
        monkeypatch.setattr(
            l1_mod,
            "_load_torch_npu",
            lambda: SimpleNamespace(npu=npu_api, __version__=runtime_npu_version),
        )
    program = _compiled(tmp_path, f"bad_adapter_{attribute}")

    with pytest.raises(RuntimeError, match=message):
        l1_mod.pypto_init(programs=[program], device=1)

    assert worker.init_calls == []


def test_init_cleanup_failure_exposes_close_only_recovery_context(tmp_path: Path, l1_fakes) -> None:
    _, worker, _ = l1_fakes
    program = _compiled(tmp_path, "retained_init")
    worker.fail_init_with_retained_owner = True

    with pytest.raises(l1_mod.L1InitializationError) as raised:
        l1_mod.pypto_init(programs=[program], device=1)

    cleanup = raised.value.cleanup_context
    with pytest.raises(RuntimeError, match=r"only close\(\) is permitted"):
        cleanup.operator(program)
    worker.fail_finalize_once = True
    with pytest.raises(RuntimeError, match="injected close failure"):
        cleanup.close()
    assert not cleanup.closed
    cleanup.close()
    assert cleanup.closed
    assert worker.finalize_calls == 2


def test_keyboard_interrupt_after_native_init_preserves_cleanup_owner(tmp_path: Path, l1_fakes) -> None:
    _, worker, _ = l1_fakes
    program = _compiled(tmp_path, "interrupted_init")
    worker.init_exception = KeyboardInterrupt("injected interrupt after native ownership")

    with pytest.raises(l1_mod.L1InitializationError) as raised:
        l1_mod.pypto_init(programs=[program], device=1)

    assert isinstance(raised.value.__cause__, KeyboardInterrupt)
    cleanup = raised.value.cleanup_context
    with pytest.raises(RuntimeError, match=r"only close\(\) is permitted"):
        cleanup.prepare()
    cleanup.close()
    assert cleanup.closed
    assert worker.finalize_calls == 1


def test_final_callable_signature_mismatch_precedes_native_init(tmp_path: Path, l1_fakes) -> None:
    _, worker, _ = l1_fakes
    program = _compiled(tmp_path, "signature_mismatch")
    program._chip_callable.scalar_count = 0

    with pytest.raises(ValueError, match="assembled L1 callable signature"):
        l1_mod.pypto_init(programs=[program], device=1)
    assert worker.init_calls == []


def test_inference_only_rejects_requires_grad_before_prepare(tmp_path: Path, l1_fakes) -> None:
    _, worker, adapter = l1_fakes
    program = _compiled(tmp_path, "autograd")
    ctx = l1_mod.pypto_init(programs=[program], device=1)
    op = ctx.operator(program)
    x = torch.ones((2, 3), dtype=torch.float32, requires_grad=True)
    y = torch.empty((2, 3), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="inference-only"):
        op(x, 1.0, out=y)
    assert worker.prepare_capsules == []
    assert adapter.calls == []
    ctx.close()


def test_direct_prepare_does_not_construct_queue_capsule(tmp_path: Path, l1_fakes) -> None:
    npu_api, worker, _ = l1_fakes
    npu_api.stream = SimpleNamespace(npu_stream=0x1234)
    program = _compiled(tmp_path, "direct")
    ctx = l1_mod.pypto_init(
        programs=[program],
        device=1,
        config=l1_mod.L1Config(use_task_queue=False),
    )

    ctx.prepare()

    assert worker.prepare_capsules == []
    assert len(worker.direct_prepare_calls) == 1
    assert worker.direct_prepare_calls[0][2] == 0x1234
    ctx.close()


def test_direct_launch_uses_raw_stream_and_records_tensor_lifetime(
    tmp_path: Path,
    l1_fakes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    npu_api, worker, adapter = l1_fakes
    stream = SimpleNamespace(npu_stream=0x5678)
    npu_api.stream = stream
    recorded: list[tuple[torch.Tensor, object]] = []

    def record_stream(tensor: torch.Tensor, used_stream: object) -> None:
        recorded.append((tensor, used_stream))

    monkeypatch.setattr(torch.Tensor, "record_stream", record_stream)
    program = _compiled(tmp_path, "direct_launch")
    ctx = l1_mod.pypto_init(
        programs=[program],
        device=1,
        config=l1_mod.L1Config(use_task_queue=False),
    )
    op = ctx.operator(program)
    x = torch.ones((2, 3), dtype=torch.float32)
    y = torch.empty((2, 3), dtype=torch.float32)

    assert op.warmup(x, 1.0, out=y) is y

    assert op.warmed and ctx.prepared
    assert worker.prepare_capsules == []
    assert worker.launch_args == []
    assert len(worker.direct_prepare_calls) == 1
    assert len(worker.direct_launch_calls) == 1
    callable_id, packed, raw_stream = worker.direct_launch_calls[0]
    assert callable_id == 0
    assert raw_stream == 0x5678
    assert packed.tensor(0).data == x.data_ptr()
    assert packed.tensor(1).data == y.data_ptr()
    assert len(recorded) == 2
    assert recorded[0][0] is x and recorded[0][1] is stream
    assert recorded[1][0] is y and recorded[1][1] is stream
    assert npu_api.current_stream_calls == 2
    assert adapter.calls == []
    ctx.close()


def _capture_error(call, errors: list[BaseException]) -> None:
    try:
        call()
    except BaseException as exc:  # noqa: BLE001 - test transfers the exact failure across a thread
        errors.append(exc)
