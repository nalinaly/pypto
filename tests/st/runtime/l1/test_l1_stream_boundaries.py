# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A2/A3 onboard ordering stress for borrowed-stream PyPTO L1 calls."""

from typing import Any

import pypto.language as pl
import pytest
import torch
import torch_npu
from pypto.l1 import L1InitializationError, pypto_init
from pypto.runtime import RunConfig

_ROWS = 64
_COLS = 128
_SHAPE = (_ROWS, _COLS)
_TORCH_CHAIN_DEPTH = 24
_L1_CHILD_COUNT = 8
_REPLAY_VALUES = (2.0, -3.5, 7.25)


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


@pl.jit
def _l1_eight_child_chain(
    source: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    bias: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    output: pl.Out[pl.Tensor[[_ROWS, _COLS], pl.FP32]],
):
    stage_0 = pl.create_tensor([_ROWS, _COLS], dtype=pl.FP32)
    stage_1 = pl.create_tensor([_ROWS, _COLS], dtype=pl.FP32)
    stage_2 = pl.create_tensor([_ROWS, _COLS], dtype=pl.FP32)
    stage_3 = pl.create_tensor([_ROWS, _COLS], dtype=pl.FP32)
    stage_4 = pl.create_tensor([_ROWS, _COLS], dtype=pl.FP32)
    stage_5 = pl.create_tensor([_ROWS, _COLS], dtype=pl.FP32)
    stage_6 = pl.create_tensor([_ROWS, _COLS], dtype=pl.FP32)
    stage_0 = _child_add(source, bias, stage_0)
    stage_1 = _child_add(stage_0, bias, stage_1)
    stage_2 = _child_add(stage_1, bias, stage_2)
    stage_3 = _child_add(stage_2, bias, stage_3)
    stage_4 = _child_add(stage_3, bias, stage_4)
    stage_5 = _child_add(stage_4, bias, stage_5)
    stage_6 = _child_add(stage_5, bias, stage_6)
    output = _child_add(stage_6, bias, output)
    return output


def _enqueue_torch_chain(
    source: torch.Tensor,
    increment: torch.Tensor,
    buffers: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    current = source
    for index in range(_TORCH_CHAIN_DEPTH):
        destination = buffers[index % len(buffers)]
        torch.add(current, increment, out=destination)
        current = destination
    return current


def _expected(value: float, bias: float) -> torch.Tensor:
    result = value + 2 * _TORCH_CHAIN_DEPTH + _L1_CHILD_COUNT * bias
    return torch.full(_SHAPE, result, dtype=torch.float32)


def _close(context: Any, graph: Any, device_id: int) -> None:
    if context is None:
        return
    torch_npu.npu.synchronize(device_id)
    if graph is not None:
        graph.reset()
    context.close()


@pytest.mark.parametrize("runtime", ("tensormap_and_ringbuffer", "host_build_graph"))
@pytest.mark.parametrize("platform", ("a2a3",))
def test_l1_delayed_predecessor_and_hidden_tail_stay_inside_operator_boundary(
    test_config: RunConfig,
    platform: str,
    runtime: str,
) -> None:
    """Stress both L1 stream boundaries in eager execution and graph replay."""
    device_id = test_config.device_id
    torch_npu.npu.set_device(device_id)
    _l1_eight_child_chain._cache.clear()
    program = _l1_eight_child_chain.compile(
        config=RunConfig(platform=platform, device_id=device_id, runtime=runtime)
    )

    context = None
    graph = None
    try:
        try:
            context = pypto_init(programs=[program], device=device_id)
        except L1InitializationError as exc:
            context = exc.cleanup_context
            raise
        op = context.operator(program)
        device = torch.device(f"npu:{device_id}")
        source = torch.full(_SHAPE, _REPLAY_VALUES[0], dtype=torch.float32, device=device)
        increment = torch.ones(_SHAPE, dtype=torch.float32, device=device)
        bias_value = 0.25
        bias = torch.full(_SHAPE, bias_value, dtype=torch.float32, device=device)
        predecessor_buffers = (
            torch.empty(_SHAPE, dtype=torch.float32, device=device),
            torch.empty(_SHAPE, dtype=torch.float32, device=device),
        )
        l1_output = torch.empty(_SHAPE, dtype=torch.float32, device=device)
        successor_buffers = (
            torch.empty(_SHAPE, dtype=torch.float32, device=device),
            torch.empty(_SHAPE, dtype=torch.float32, device=device),
        )

        context.prepare()
        predecessor = _enqueue_torch_chain(source, increment, predecessor_buffers)
        assert op.warmup(predecessor, bias, out=l1_output) is l1_output
        eager_output = _enqueue_torch_chain(l1_output, increment, successor_buffers)
        torch_npu.npu.synchronize(device_id)
        torch.testing.assert_close(eager_output.cpu(), _expected(_REPLAY_VALUES[0], bias_value))

        capture_stream = torch_npu.npu.Stream(device=device_id)
        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph, stream=capture_stream):
            predecessor = _enqueue_torch_chain(source, increment, predecessor_buffers)
            op(predecessor, bias, out=l1_output)
            graph_output = _enqueue_torch_chain(l1_output, increment, successor_buffers)

        for value in _REPLAY_VALUES:
            with torch_npu.npu.stream(capture_stream):
                source.fill_(value)
                graph.replay()
            capture_stream.synchronize()
            torch.testing.assert_close(graph_output.cpu(), _expected(value, bias_value))
    finally:
        _close(context, graph, device_id)

    assert context is not None
    assert context.closed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
