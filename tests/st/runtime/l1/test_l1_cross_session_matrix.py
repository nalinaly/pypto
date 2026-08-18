# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cross-session A2/A3 regressions distilled from the sibling L1 implementation.

These cases intentionally keep GPT's stricter lifetime contract: every graph is
externally quiescent and reset before its L1 context is closed.
"""

from collections.abc import Sequence
from typing import Any

import pypto.language as pl
import pytest
import torch
import torch_npu
from examples.beginner.activation import fused_add_relu, silu
from examples.beginner.elementwise import tile_add_64
from examples.intermediate.fused_linear import fused_matmul_bias
from harness.core.harness import ONBOARD_PLATFORMS
from pypto.l1 import L1InitializationError, pypto_init
from pypto.runtime import RunConfig

_ROWS = 64
_COLS = 128
_SHAPE = (_ROWS, _COLS)
_RUNTIMES = ("tensormap_and_ringbuffer", "host_build_graph")


@pl.jit
def _l1_add_f32(
    lhs: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    rhs: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    output: pl.Out[pl.Tensor[[_ROWS, _COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs_tile = pl.load(lhs, [0, 0], [_ROWS, _COLS])
        rhs_tile = pl.load(rhs, [0, 0], [_ROWS, _COLS])
        pl.store(pl.add(lhs_tile, rhs_tile), [0, 0], output)
    return output


@pl.jit
def _l1_mul_f16(
    lhs: pl.Tensor[[_ROWS, _COLS], pl.FP16],
    rhs: pl.Tensor[[_ROWS, _COLS], pl.FP16],
    output: pl.Out[pl.Tensor[[_ROWS, _COLS], pl.FP16]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs_tile = pl.load(lhs, [0, 0], [_ROWS, _COLS])
        rhs_tile = pl.load(rhs, [0, 0], [_ROWS, _COLS])
        pl.store(pl.mul(lhs_tile, rhs_tile), [0, 0], output)
    return output


def _compile(kernel: Any, *, platform: str, device_id: int, runtime: str):
    kernel._cache.clear()
    return kernel.compile(config=RunConfig(platform=platform, device_id=device_id, runtime=runtime))


def _compile_with_args(
    kernel: Any,
    *args: torch.Tensor,
    platform: str,
    device_id: int,
    runtime: str,
):
    kernel._cache.clear()
    return kernel.compile(
        *args,
        config=RunConfig(platform=platform, device_id=device_id, runtime=runtime),
    )


def _cleanup_context(context: Any, graphs: Sequence[Any], device_id: int) -> None:
    if context is None:
        return
    torch_npu.npu.synchronize(device_id)
    for graph in reversed(graphs):
        graph.reset()
    context.close()


@pytest.mark.parametrize("runtime", _RUNTIMES)
@pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
def test_same_callable_twice_in_one_graph_with_nonuniform_replays(
    test_config: RunConfig,
    platform: str,
    runtime: str,
) -> None:
    """Each captured node must retain its own args despite sharing a callable."""
    device_id = test_config.device_id
    torch_npu.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    program = _compile(_l1_add_f32, platform=platform, device_id=device_id, runtime=runtime)

    context = None
    graph = None
    try:
        try:
            context = pypto_init(programs=(program,), device=device_id)
        except L1InitializationError as exc:
            context = exc.cleanup_context
            raise
        op = context.operator(program)
        lhs = torch.zeros(_SHAPE, dtype=torch.float32, device=device)
        rhs = torch.zeros(_SHAPE, dtype=torch.float32, device=device)
        bias = torch.zeros(_SHAPE, dtype=torch.float32, device=device)
        intermediate = torch.empty(_SHAPE, dtype=torch.float32, device=device)
        output = torch.empty(_SHAPE, dtype=torch.float32, device=device)
        capture_stream = torch_npu.npu.Stream(device=device_id)

        context.prepare()
        op.warmup(lhs, rhs, out=intermediate)
        op(intermediate, bias, out=output)
        torch_npu.npu.synchronize(device_id)

        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph, stream=capture_stream):
            op(lhs, rhs, out=intermediate)
            op(intermediate, bias, out=output)

        element_count = _ROWS * _COLS
        base = torch.arange(element_count, dtype=torch.float32).reshape(_SHAPE) * 0.01
        for replay_index in range(8):
            host_lhs = base + float(replay_index)
            host_rhs = torch.flip(base, dims=(1,)) * 0.5
            host_bias = torch.full(_SHAPE, 0.25 * float(replay_index - 2))
            with torch_npu.npu.stream(capture_stream):
                lhs.copy_(host_lhs)
                rhs.copy_(host_rhs)
                bias.copy_(host_bias)
                graph.replay()
            capture_stream.synchronize()
            torch.testing.assert_close(output.cpu(), host_lhs + host_rhs + host_bias)
    finally:
        _cleanup_context(context, () if graph is None else (graph,), device_id)


@pytest.mark.parametrize("runtime", _RUNTIMES)
@pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
def test_fp16_graph_replay_uses_current_tensor_contents(
    test_config: RunConfig,
    platform: str,
    runtime: str,
) -> None:
    """L1 tensor descriptors and graph replay are not accidentally FP32-only."""
    device_id = test_config.device_id
    torch_npu.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    program = _compile(_l1_mul_f16, platform=platform, device_id=device_id, runtime=runtime)

    context = None
    graph = None
    try:
        try:
            context = pypto_init(programs=(program,), device=device_id)
        except L1InitializationError as exc:
            context = exc.cleanup_context
            raise
        op = context.operator(program)
        lhs = torch.ones(_SHAPE, dtype=torch.float16, device=device)
        rhs = torch.ones(_SHAPE, dtype=torch.float16, device=device)
        output = torch.empty(_SHAPE, dtype=torch.float16, device=device)
        capture_stream = torch_npu.npu.Stream(device=device_id)

        context.prepare()
        op.warmup(lhs, rhs, out=output)
        torch_npu.npu.synchronize(device_id)

        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph, stream=capture_stream):
            op(lhs, rhs, out=output)

        for lhs_value, rhs_value in ((1.5, 4.0), (0.5, 0.25), (-2.0, 3.0)):
            with torch_npu.npu.stream(capture_stream):
                lhs.fill_(lhs_value)
                rhs.fill_(rhs_value)
                graph.replay()
            capture_stream.synchronize()
            expected = torch.full(_SHAPE, lhs_value * rhs_value, dtype=torch.float16)
            torch.testing.assert_close(output.cpu(), expected, rtol=1e-3, atol=1e-3)
    finally:
        _cleanup_context(context, () if graph is None else (graph,), device_id)


@pytest.mark.parametrize("runtime", _RUNTIMES)
@pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
def test_two_graphs_on_distinct_streams_replay_sequentially(
    test_config: RunConfig,
    platform: str,
    runtime: str,
) -> None:
    """Externally quiescent stream changes must not import an old graph tail."""
    device_id = test_config.device_id
    torch_npu.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    program = _compile(_l1_add_f32, platform=platform, device_id=device_id, runtime=runtime)

    context = None
    graph_a = None
    graph_b = None
    try:
        try:
            context = pypto_init(programs=(program,), device=device_id)
        except L1InitializationError as exc:
            context = exc.cleanup_context
            raise
        op = context.operator(program)
        lhs = torch.zeros(_SHAPE, dtype=torch.float32, device=device)
        rhs = torch.zeros(_SHAPE, dtype=torch.float32, device=device)
        intermediate = torch.empty(_SHAPE, dtype=torch.float32, device=device)
        tail = torch.zeros(_SHAPE, dtype=torch.float32, device=device)
        output = torch.empty(_SHAPE, dtype=torch.float32, device=device)
        stream_a = torch_npu.npu.Stream(device=device_id)
        stream_b = torch_npu.npu.Stream(device=device_id)
        assert stream_a.npu_stream != stream_b.npu_stream

        context.prepare()
        op.warmup(lhs, rhs, out=intermediate)
        op(intermediate, tail, out=output)
        torch_npu.npu.synchronize(device_id)

        graph_a = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph_a, stream=stream_a):
            op(lhs, rhs, out=intermediate)
        stream_a.synchronize()

        graph_b = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph_b, stream=stream_b):
            op(intermediate, tail, out=output)
        stream_b.synchronize()

        for lhs_value, rhs_value, tail_value in ((2.0, 3.0, 4.0), (-5.0, 1.5, 2.0), (8.0, -3.0, -1.0)):
            with torch_npu.npu.stream(stream_a):
                lhs.fill_(lhs_value)
                rhs.fill_(rhs_value)
                graph_a.replay()
            stream_a.synchronize()
            with torch_npu.npu.stream(stream_b):
                tail.fill_(tail_value)
                graph_b.replay()
            stream_b.synchronize()
            expected = torch.full(_SHAPE, lhs_value + rhs_value + tail_value)
            torch.testing.assert_close(output.cpu(), expected)
    finally:
        graphs = tuple(graph for graph in (graph_a, graph_b) if graph is not None)
        _cleanup_context(context, graphs, device_id)


@pytest.mark.parametrize("runtime", _RUNTIMES)
@pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
def test_distinct_nonlinear_programs_share_one_graph(
    test_config: RunConfig,
    platform: str,
    runtime: str,
) -> None:
    """Two unlike vector programs keep independent callable-local state."""
    device_id = test_config.device_id
    torch_npu.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    relu_lhs = torch.zeros((128, 128), dtype=torch.float32, device=device)
    relu_rhs = torch.zeros((128, 128), dtype=torch.float32, device=device)
    relu_output = torch.empty((128, 128), dtype=torch.float32, device=device)
    silu_input = torch.zeros((32, 128), dtype=torch.float32, device=device)
    silu_output = torch.empty((32, 128), dtype=torch.float32, device=device)
    relu_program = _compile_with_args(
        fused_add_relu,
        relu_lhs,
        relu_rhs,
        relu_output,
        platform=platform,
        device_id=device_id,
        runtime=runtime,
    )
    silu_program = _compile_with_args(
        silu,
        silu_input,
        silu_output,
        platform=platform,
        device_id=device_id,
        runtime=runtime,
    )

    context = None
    graph = None
    try:
        try:
            context = pypto_init(programs=(relu_program, silu_program), device=device_id)
        except L1InitializationError as exc:
            context = exc.cleanup_context
            raise
        relu_op = context.operator(relu_program)
        silu_op = context.operator(silu_program)
        capture_stream = torch_npu.npu.Stream(device=device_id)

        context.prepare()
        relu_op.warmup(relu_lhs, relu_rhs, out=relu_output)
        silu_op.warmup(silu_input, out=silu_output)
        torch_npu.npu.synchronize(device_id)

        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph, stream=capture_stream):
            relu_op(relu_lhs, relu_rhs, out=relu_output)
            silu_op(silu_input, out=silu_output)

        relu_count = relu_lhs.numel()
        silu_count = silu_input.numel()
        for offset in (-2.5, 0.0, 3.0):
            host_relu_lhs = torch.linspace(-4.0, 4.0, relu_count).reshape(relu_lhs.shape)
            host_relu_rhs = torch.full(relu_rhs.shape, offset)
            host_silu = torch.linspace(-3.0 + offset, 2.0 + offset, silu_count).reshape(silu_input.shape)
            with torch_npu.npu.stream(capture_stream):
                relu_lhs.copy_(host_relu_lhs)
                relu_rhs.copy_(host_relu_rhs)
                silu_input.copy_(host_silu)
                graph.replay()
            capture_stream.synchronize()
            torch.testing.assert_close(relu_output.cpu(), torch.relu(host_relu_lhs + host_relu_rhs))
            torch.testing.assert_close(
                silu_output.cpu(),
                host_silu * torch.sigmoid(host_silu),
                rtol=2e-3,
                atol=2e-3,
            )
    finally:
        _cleanup_context(context, () if graph is None else (graph,), device_id)


@pytest.mark.parametrize("runtime", _RUNTIMES)
@pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
def test_multichild_matmul_then_distinct_small_tile_program(
    test_config: RunConfig,
    platform: str,
    runtime: str,
) -> None:
    """A cube+vector multi-child callable composes with a separate 64x64 callable."""
    device_id = test_config.device_id
    torch_npu.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    shape = (64, 64)
    lhs = torch.zeros(shape, dtype=torch.float32, device=device)
    rhs = torch.eye(shape[0], dtype=torch.float32, device=device)
    bias = torch.zeros(shape, dtype=torch.float32, device=device)
    matmul_output = torch.empty(shape, dtype=torch.float32, device=device)
    tail = torch.zeros(shape, dtype=torch.float32, device=device)
    output = torch.empty(shape, dtype=torch.float32, device=device)
    matmul_program = _compile_with_args(
        fused_matmul_bias,
        lhs,
        rhs,
        bias,
        matmul_output,
        platform=platform,
        device_id=device_id,
        runtime=runtime,
    )
    assert matmul_program.chip_callable.child_count >= 2
    add_program = _compile_with_args(
        tile_add_64,
        matmul_output,
        tail,
        output,
        platform=platform,
        device_id=device_id,
        runtime=runtime,
    )

    context = None
    graph = None
    try:
        try:
            context = pypto_init(programs=(matmul_program, add_program), device=device_id)
        except L1InitializationError as exc:
            context = exc.cleanup_context
            raise
        matmul_op = context.operator(matmul_program)
        add_op = context.operator(add_program)
        capture_stream = torch_npu.npu.Stream(device=device_id)

        context.prepare()
        matmul_op.warmup(lhs, rhs, bias, out=matmul_output)
        add_op.warmup(matmul_output, tail, out=output)
        torch_npu.npu.synchronize(device_id)

        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph, stream=capture_stream):
            matmul_op(lhs, rhs, bias, out=matmul_output)
            add_op(matmul_output, tail, out=output)

        element_count = shape[0] * shape[1]
        host_lhs_base = torch.linspace(-1.0, 1.0, element_count).reshape(shape)
        for scale, bias_value, tail_value in ((0.5, -1.0, 2.0), (2.0, 0.25, -0.5), (-1.0, 3.0, 1.0)):
            host_lhs = host_lhs_base + scale
            host_rhs = torch.eye(shape[0], dtype=torch.float32) * scale
            host_bias = torch.full(shape, bias_value)
            host_tail = torch.linspace(tail_value, tail_value + 1.0, element_count).reshape(shape)
            with torch_npu.npu.stream(capture_stream):
                lhs.copy_(host_lhs)
                rhs.copy_(host_rhs)
                bias.copy_(host_bias)
                tail.copy_(host_tail)
                graph.replay()
            capture_stream.synchronize()
            expected = host_lhs * scale + host_bias + host_tail
            torch.testing.assert_close(output.cpu(), expected, rtol=1e-3, atol=1e-3)
    finally:
        _cleanup_context(context, () if graph is None else (graph,), device_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
