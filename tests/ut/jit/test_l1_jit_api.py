# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""No-hardware contracts for the Triton-style PyPTO L1 JIT facade."""

from types import SimpleNamespace
from typing import Any

import pypto.language as pl
import pytest
import torch
from pypto.jit import decorator as decorator_mod
from pypto.runtime import l1_jit


@pl.jit(execution="l1")
def _explicit_l1_kernel(
    x: pl.Tensor[[2, 3], pl.FP32],
    scale: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[2, 3], pl.FP32]],
):
    del x, scale, out


@pl.jit(execution="l1", runtime="host_build_graph")
def _implicit_hbg_kernel(
    x: pl.Tensor[[2, 3], pl.FP32],
    out: pl.Out[pl.Tensor[[2, 3], pl.FP32]],
):
    del x, out


@pl.jit(execution="l1")
def _lowerable_l1_kernel(
    lhs: pl.Tensor[[2, 128], pl.FP32],
    rhs: pl.Tensor[[2, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[2, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs_tile = pl.load(lhs, [0, 0], [2, 128])
        rhs_tile = pl.load(rhs, [0, 0], [2, 128])
        pl.store(pl.add(lhs_tile, rhs_tile), [0, 0], out)
    return out


def test_l1_decorator_metadata_and_invalid_combinations() -> None:
    assert _explicit_l1_kernel._execution == "l1"
    assert _explicit_l1_kernel._execution_runtime == "tensormap_and_ringbuffer"
    assert _implicit_hbg_kernel._execution_runtime == "host_build_graph"

    with pytest.raises(ValueError, match="execution"):

        @pl.jit(execution="invalid")
        def _invalid(x: pl.Tensor[[1], pl.FP32]):
            del x

    with pytest.raises(ValueError, match="only valid"):

        @pl.jit(runtime="host_build_graph")
        def _runtime_without_l1(x: pl.Tensor[[1], pl.FP32]):
            del x


def test_l1_lower_defaults_to_a2a3_without_public_init() -> None:
    lowered = _lowerable_l1_kernel.lower()

    assert lowered is not None


def test_l1_explicit_output_dispatch_returns_same_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    x = torch.ones((2, 3), dtype=torch.float32)
    out = torch.empty((2, 3), dtype=torch.float32)
    config = object()
    compiled = object()
    resolved: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    dispatched: list[tuple[Any, list[Any], str, Any]] = []

    monkeypatch.setattr(
        _explicit_l1_kernel,
        "_l1_validate_run_config",
        lambda _config, _arguments, **_kwargs: config,
    )

    def resolve(args: tuple[Any, ...], kwargs: dict[str, Any], **_ignored: Any):
        resolved.append((args, kwargs))
        return compiled, [x, 2.0, out], config

    monkeypatch.setattr(_explicit_l1_kernel, "_resolve_compiled", resolve)

    def dispatch(program: Any, arguments: list[Any], *, runtime: str, run_config: Any):
        dispatched.append((program, arguments, runtime, run_config))
        return arguments[-1]

    monkeypatch.setattr(l1_jit, "dispatch", dispatch)

    returned = _explicit_l1_kernel(x, 2.0, out=out)

    assert returned is out
    assert resolved and resolved[0][1]["config"] is config
    assert dispatched == [(compiled, [x, 2.0, out], "tensormap_and_ringbuffer", config)]


def test_l1_omitted_output_uses_wrapper_allocator(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTensor:
        def __init__(self, device: object) -> None:
            self.device = device

    fake_device = object()
    x = FakeTensor(fake_device)
    allocations: list[tuple[tuple[int, ...], Any, object]] = []
    allocated = FakeTensor(fake_device)
    fake_torch = SimpleNamespace(
        Tensor=FakeTensor,
        empty=lambda shape, *, dtype, device: allocations.append((tuple(shape), dtype, device)) or allocated,
    )
    config = object()

    monkeypatch.setattr(decorator_mod, "_get_torch", lambda: fake_torch)
    monkeypatch.setattr(
        _implicit_hbg_kernel,
        "_l1_validate_run_config",
        lambda _config, _arguments, **_kwargs: config,
    )
    monkeypatch.setattr(
        _implicit_hbg_kernel,
        "_resolve_compiled",
        lambda _args, _kwargs, **_ignored: (object(), [x, allocated], config),
    )
    monkeypatch.setattr(l1_jit, "dispatch", lambda _program, arguments, **_kwargs: arguments[-1])

    returned = _implicit_hbg_kernel(x)

    assert returned is allocated
    assert allocations == [((2, 3), torch.float32, fake_device)]


def test_l1_partial_output_omission_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    @pl.jit(execution="l1")
    def two_outputs(
        x: pl.Tensor[[2, 3], pl.FP32],
        first: pl.Out[pl.Tensor[[2, 3], pl.FP32]],
        second: pl.Out[pl.Tensor[[2, 3], pl.FP32]],
    ):
        del x, first, second

    x = torch.ones((2, 3), dtype=torch.float32)
    first = torch.empty_like(x)
    with pytest.raises(TypeError, match="all-or-none"):
        two_outputs(x, first=first)


class _FakeContext:
    instances: list["_FakeContext"] = []

    def __init__(self, programs, *, device: int, config: Any) -> None:
        self.programs = list(programs)
        self.device = device
        self.config = config
        self.close_calls = 0
        self.fail_close_once = False
        self.calls: list[tuple[Any, tuple[Any, ...], Any]] = []
        self.instances.append(self)

    def add_program(self, program: Any):
        if program not in self.programs:
            self.programs.append(program)

        def op(*args: Any, out: Any = None):
            self.calls.append((program, args, out))
            return out

        return op

    def close(self) -> None:
        self.close_calls += 1
        if self.fail_close_once and self.close_calls == 1:
            raise RuntimeError("injected close failure")


@pytest.fixture
def hidden_registry(monkeypatch: pytest.MonkeyPatch):
    l1_jit._DEVICE_OWNERS.clear()
    _FakeContext.instances.clear()
    monkeypatch.setattr(l1_jit, "L1Context", _FakeContext)
    monkeypatch.setattr(l1_jit, "_infer_device", lambda _arguments: 1)
    yield
    l1_jit._DEVICE_OWNERS.clear()


def _fake_program(runtime: str = "tensormap_and_ringbuffer") -> Any:
    return SimpleNamespace(
        platform="a2a3",
        runtime_name=runtime,
        _get_metadata=lambda: ([object(), object()], [1], []),
    )


def _fake_run_config() -> Any:
    return SimpleNamespace(
        aicpu_thread_num=None,
        ring_task_window=None,
        ring_heap=None,
        ring_dep_pool=None,
    )


def test_hidden_registry_appends_and_shutdown_is_retryable(hidden_registry) -> None:
    first = _fake_program()
    second = _fake_program()
    config = _fake_run_config()
    first_out = object()
    second_out = object()

    assert (
        l1_jit.dispatch(
            first,
            [object(), first_out],
            runtime="tensormap_and_ringbuffer",
            run_config=config,
        )
        is first_out
    )
    assert (
        l1_jit.dispatch(
            second,
            [object(), second_out],
            runtime="tensormap_and_ringbuffer",
            run_config=config,
        )
        is second_out
    )

    context = _FakeContext.instances[0]
    assert context.programs == [first, second]
    context.fail_close_once = True
    with pytest.raises(RuntimeError, match="close failure"):
        l1_jit.shutdown(device=1)
    assert l1_jit._DEVICE_OWNERS[1].state == "cleanup-only"

    l1_jit.shutdown(device=1)
    l1_jit.shutdown(device=1)
    assert context.close_calls == 2
    assert l1_jit._DEVICE_OWNERS[1].state == "retired"


def test_hidden_registry_adds_capture_warmup_context_to_init_errors(
    hidden_registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingContext:
        def __init__(self, _programs, *, device: int, config: Any) -> None:
            del device, config
            raise RuntimeError("injected native init failure")

    monkeypatch.setattr(l1_jit, "L1Context", FailingContext)

    with pytest.raises(RuntimeError, match="ordinary eager call before ACLGraph capture") as exc_info:
        l1_jit.dispatch(
            _fake_program(),
            [object(), object()],
            runtime="tensormap_and_ringbuffer",
            run_config=_fake_run_config(),
        )

    assert "injected native init failure" in str(exc_info.value)
    assert l1_jit._DEVICE_OWNERS == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
