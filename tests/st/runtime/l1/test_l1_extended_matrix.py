# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Extended onboard acceptance matrix for PyPTO L1 task/package lifetime."""

from collections.abc import Sequence
from typing import Any

import pypto.language as pl
import pytest
import torch
import torch_npu
from harness.core.harness import ONBOARD_PLATFORMS
from pypto.l1 import L1InitializationError, pypto_init
from pypto.runtime import RunConfig

_ROWS = 64
_COLS = 128
_SHAPE = (_ROWS, _COLS)
_RUNTIMES = ("tensormap_and_ringbuffer", "host_build_graph")


@pl.jit
def _l1_add_scalar(
    source: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    output: pl.Out[pl.Tensor[[_ROWS, _COLS], pl.FP32]],
    scalar: pl.Scalar[pl.FP32] = pl.RUNTIME,
):
    with pl.at(level=pl.Level.CORE_GROUP):
        source_tile = pl.load(source, [0, 0], [_ROWS, _COLS])
        pl.store(pl.add(source_tile, scalar), [0, 0], output)
    return output


@pl.jit
def _l1_multi_output(
    lhs: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    rhs: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    sum_output: pl.Out[pl.Tensor[[_ROWS, _COLS], pl.FP32]],
    diff_output: pl.Out[pl.Tensor[[_ROWS, _COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs_tile = pl.load(lhs, [0, 0], [_ROWS, _COLS])
        rhs_tile = pl.load(rhs, [0, 0], [_ROWS, _COLS])
        pl.store(pl.add(lhs_tile, rhs_tile), [0, 0], sum_output)
        pl.store(pl.sub(lhs_tile, rhs_tile), [0, 0], diff_output)
    return sum_output, diff_output


@pl.jit.incore
def _child_add(
    lhs: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    rhs: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    output: pl.Out[pl.Tensor[[_ROWS, _COLS], pl.FP32]],
):
    lhs_tile = pl.load(lhs, [0, 0], [_ROWS, _COLS])
    rhs_tile = pl.load(rhs, [0, 0], [_ROWS, _COLS])
    pl.store(pl.add(lhs_tile, rhs_tile), [0, 0], output)
    return output


@pl.jit.incore
def _child_scale(
    source: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    scalar: pl.Scalar[pl.FP32],
    output: pl.Out[pl.Tensor[[_ROWS, _COLS], pl.FP32]],
):
    source_tile = pl.load(source, [0, 0], [_ROWS, _COLS])
    pl.store(pl.mul(source_tile, scalar), [0, 0], output)
    return output


@pl.jit
def _l1_multi_child(
    lhs: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    rhs: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    output: pl.Out[pl.Tensor[[_ROWS, _COLS], pl.FP32]],
):
    intermediate = pl.create_tensor([_ROWS, _COLS], dtype=pl.FP32)
    intermediate = _child_add(lhs, rhs, intermediate)
    output = _child_scale(intermediate, 2.0, output)
    return output


def _device_tensor(device_id: int, value: float) -> torch.Tensor:
    return torch.full(_SHAPE, value, dtype=torch.float32, device=torch.device(f"npu:{device_id}"))


def _empty_device_tensor(device_id: int) -> torch.Tensor:
    return torch.empty(_SHAPE, dtype=torch.float32, device=torch.device(f"npu:{device_id}"))


def _cleanup_context(context: Any, graphs: Sequence[Any], device_id: int) -> None:
    if context is None:
        return
    # The caller, not PyPTO, establishes quiescence and destroys every graph
    # before releasing graph-visible runtime state.
    torch_npu.npu.synchronize(device_id)
    for graph in reversed(graphs):
        graph.reset()
    context.close()


def _compile(kernel: Any, *, platform: str, device_id: int, runtime: str):
    kernel._cache.clear()
    return kernel.compile(config=RunConfig(platform=platform, device_id=device_id, runtime=runtime))


@pytest.mark.parametrize("runtime", _RUNTIMES)
@pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
def test_l1_async_tensor_and_scalar_snapshots_do_not_alias(
    test_config: RunConfig,
    platform: str,
    runtime: str,
) -> None:
    """Several unsynchronized Host calls retain distinct addresses and scalars."""
    device_id = test_config.device_id
    torch_npu.npu.set_device(device_id)
    program = _compile(_l1_add_scalar, platform=platform, device_id=device_id, runtime=runtime)

    context = None
    try:
        try:
            context = pypto_init(programs=[program], device=device_id)
        except L1InitializationError as exc:
            context = exc.cleanup_context
            raise
        op = context.operator(program)
        context.prepare()

        warmup_input = _device_tensor(device_id, 0.0)
        warmup_output = _empty_device_tensor(device_id)
        op.warmup(warmup_input, 0.0, out=warmup_output)
        torch_npu.npu.synchronize(device_id)

        cases = ((1.0, 0.5), (3.0, -2.25), (-4.0, 7.0), (9.0, -0.125))
        inputs = [_device_tensor(device_id, value) for value, _ in cases]
        outputs = [_empty_device_tensor(device_id) for _ in cases]
        for input_tensor, output_tensor, (_, scalar) in zip(inputs, outputs, cases, strict=True):
            op(input_tensor, scalar, out=output_tensor)

        # Deliberately synchronize only after every Host invocation has
        # returned and its temporary queue-call/HostArgs container is gone.
        torch_npu.npu.synchronize(device_id)
        for output_tensor, (value, scalar) in zip(outputs, cases, strict=True):
            torch.testing.assert_close(
                output_tensor.cpu(),
                torch.full(_SHAPE, value + scalar, dtype=torch.float32),
            )
    finally:
        _cleanup_context(context, (), device_id)

    assert context is not None
    assert context.closed


@pytest.mark.parametrize("runtime", _RUNTIMES)
@pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
def test_l1_multi_output_multi_child_workspace_aclgraph(
    test_config: RunConfig,
    platform: str,
    runtime: str,
) -> None:
    """Capture multi-output and multi-child calls sharing internal workspace."""
    device_id = test_config.device_id
    torch_npu.npu.set_device(device_id)
    multi_output_program = _compile(
        _l1_multi_output,
        platform=platform,
        device_id=device_id,
        runtime=runtime,
    )
    multi_child_program = _compile(
        _l1_multi_child,
        platform=platform,
        device_id=device_id,
        runtime=runtime,
    )

    context = None
    graph = None
    try:
        try:
            context = pypto_init(
                programs=[multi_output_program, multi_child_program],
                device=device_id,
            )
        except L1InitializationError as exc:
            context = exc.cleanup_context
            raise

        multi_output_op = context.operator(multi_output_program)
        multi_child_op = context.operator(multi_child_program)
        lhs = _device_tensor(device_id, 2.0)
        rhs = _device_tensor(device_id, 3.0)
        sum_output = _empty_device_tensor(device_id)
        diff_output = _empty_device_tensor(device_id)
        final_output = _empty_device_tensor(device_id)
        capture_stream = torch_npu.npu.Stream(device=device_id)

        context.prepare()
        returned = multi_output_op.warmup(lhs, rhs, out=(sum_output, diff_output))
        assert isinstance(returned, tuple)
        assert returned[0] is sum_output
        assert returned[1] is diff_output
        multi_child_op.warmup(sum_output, diff_output, out=final_output)
        torch_npu.npu.synchronize(device_id)
        torch.testing.assert_close(final_output.cpu(), torch.full(_SHAPE, 8.0))

        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph, stream=capture_stream):
            multi_output_op(lhs, rhs, out=(sum_output, diff_output))
            multi_child_op(sum_output, diff_output, out=final_output)

        for value in (1.0, -5.0, 7.25):
            with torch_npu.npu.stream(capture_stream):
                lhs.fill_(value)
                graph.replay()
            capture_stream.synchronize()
            # (lhs + rhs + lhs - rhs) * 2 == 4 * lhs.
            torch.testing.assert_close(final_output.cpu(), torch.full(_SHAPE, 4.0 * value))
    finally:
        _cleanup_context(context, () if graph is None else (graph,), device_id)

    assert context is not None
    assert context.closed


@pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
def test_hbg_two_graphs_retain_distinct_addresses_and_scalars(
    test_config: RunConfig,
    platform: str,
) -> None:
    """Alternating graph A/B replay must not collapse to the latest HBG blob."""
    device_id = test_config.device_id
    torch_npu.npu.set_device(device_id)
    program = _compile(
        _l1_add_scalar,
        platform=platform,
        device_id=device_id,
        runtime="host_build_graph",
    )

    context = None
    graph_a = None
    graph_b = None
    try:
        try:
            context = pypto_init(programs=[program], device=device_id)
        except L1InitializationError as exc:
            context = exc.cleanup_context
            raise
        op = context.operator(program)
        input_a = _device_tensor(device_id, 0.0)
        input_b = _device_tensor(device_id, 0.0)
        output_a = _empty_device_tensor(device_id)
        output_b = _empty_device_tensor(device_id)
        capture_stream = torch_npu.npu.Stream(device=device_id)

        context.prepare()
        op.warmup(input_a, 0.0, out=output_a)
        torch_npu.npu.synchronize(device_id)

        graph_a = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph_a, stream=capture_stream):
            op(input_a, 1.25, out=output_a)
        graph_b = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph_b, stream=capture_stream):
            op(input_b, -3.5, out=output_b)

        replay_cases = (
            (graph_a, input_a, output_a, 2.0, 1.25),
            (graph_b, input_b, output_b, -4.0, -3.5),
            (graph_a, input_a, output_a, 8.0, 1.25),
            (graph_b, input_b, output_b, 6.5, -3.5),
        )
        for graph, input_tensor, output_tensor, value, scalar in replay_cases:
            with torch_npu.npu.stream(capture_stream):
                input_tensor.fill_(value)
                graph.replay()
            capture_stream.synchronize()
            torch.testing.assert_close(
                output_tensor.cpu(),
                torch.full(_SHAPE, value + scalar, dtype=torch.float32),
            )
    finally:
        graphs = tuple(graph for graph in (graph_a, graph_b) if graph is not None)
        _cleanup_context(context, graphs, device_id)

    assert context is not None
    assert context.closed


def _run_one_hbg_context(
    program: Any,
    args: tuple[Any, ...],
    output: torch.Tensor,
    expected: torch.Tensor,
    device_id: int,
) -> None:
    context = None
    graph = None
    try:
        try:
            context = pypto_init(programs=[program], device=device_id)
        except L1InitializationError as exc:
            context = exc.cleanup_context
            raise
        op = context.operator(program)
        context.prepare()
        op.warmup(*args, out=output)
        torch_npu.npu.synchronize(device_id)
        torch.testing.assert_close(output.cpu(), expected)

        capture_stream = torch_npu.npu.Stream(device=device_id)
        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph, stream=capture_stream):
            op(*args, out=output)
        with torch_npu.npu.stream(capture_stream):
            graph.replay()
        capture_stream.synchronize()
        torch.testing.assert_close(output.cpu(), expected)
    finally:
        _cleanup_context(context, () if graph is None else (graph,), device_id)

    assert context is not None
    assert context.closed


@pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
def test_hbg_sequential_context_generation_resets_resident_registries(
    test_config: RunConfig,
    platform: str,
) -> None:
    """A second context may register a different callable at callable_id zero."""
    device_id = test_config.device_id
    torch_npu.npu.set_device(device_id)
    scalar_program = _compile(
        _l1_add_scalar,
        platform=platform,
        device_id=device_id,
        runtime="host_build_graph",
    )
    child_program = _compile(
        _l1_multi_child,
        platform=platform,
        device_id=device_id,
        runtime="host_build_graph",
    )

    source = _device_tensor(device_id, 2.0)
    rhs = _device_tensor(device_id, 4.0)
    first_output = _empty_device_tensor(device_id)
    second_output = _empty_device_tensor(device_id)

    _run_one_hbg_context(
        scalar_program,
        (source, 5.0),
        first_output,
        torch.full(_SHAPE, 7.0),
        device_id,
    )
    _run_one_hbg_context(
        child_program,
        (source, rhs),
        second_output,
        torch.full(_SHAPE, 12.0),
        device_id,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
