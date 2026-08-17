# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Onboard acceptance test for borrowed-stream PyPTO L1 and ACLGraph."""

import pypto.language as pl
import pytest
import torch
import torch_npu
from harness.core.harness import ONBOARD_PLATFORMS
from pypto.l1 import L1InitializationError, pypto_init
from pypto.runtime import RunConfig

_ROWS = 64
_COLS = 128
_REPLAY_INPUTS = (2.0, -4.0, 7.5)


@pl.jit
def _l1_add(
    lhs: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    rhs: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[_ROWS, _COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs_tile = pl.load(lhs, [0, 0], [_ROWS, _COLS])
        rhs_tile = pl.load(rhs, [0, 0], [_ROWS, _COLS])
        pl.store(pl.add(lhs_tile, rhs_tile), [0, 0], out)
    return out


def _expected(value: float) -> torch.Tensor:
    # Captured topology: add(input, 1) -> PyPTO add(rhs=3) -> multiply by 2.
    return torch.full((_ROWS, _COLS), (value + 1.0 + 3.0) * 2.0, dtype=torch.float32)


def _allocate_graph_tensors(device_id: int) -> tuple[torch.Tensor, ...]:
    device = torch.device(f"npu:{device_id}")
    return (
        torch.full((_ROWS, _COLS), _REPLAY_INPUTS[0], dtype=torch.float32, device=device),
        torch.ones((_ROWS, _COLS), dtype=torch.float32, device=device),
        torch.full((_ROWS, _COLS), 3.0, dtype=torch.float32, device=device),
        torch.empty((_ROWS, _COLS), dtype=torch.float32, device=device),
        torch.empty((_ROWS, _COLS), dtype=torch.float32, device=device),
        torch.empty((_ROWS, _COLS), dtype=torch.float32, device=device),
    )


@pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
def test_l1_eager_warmup_and_aclgraph_replay(test_config: RunConfig, platform: str) -> None:
    """Run one ordinary warmup, then capture and replay the same L1 op."""
    device_id = test_config.device_id
    torch_npu.npu.set_device(device_id)
    assert torch_npu.npu.current_device() == device_id

    _l1_add._cache.clear()
    compiled = _l1_add.compile(config=RunConfig(platform=platform, device_id=device_id))
    context = None
    graph = None
    try:
        try:
            context = pypto_init(programs=[compiled], device=device_id)
        except L1InitializationError as exc:
            context = exc.cleanup_context
            raise

        op = context.operator(compiled)
        static_input, bias, rhs, pre_l1, l1_output, final_output = _allocate_graph_tensors(device_id)

        warmup_stream = torch_npu.npu.current_stream(device_id)
        capture_stream = torch_npu.npu.Stream(device=device_id)
        assert capture_stream.npu_stream != warmup_stream.npu_stream

        context.prepare()
        torch.add(static_input, bias, out=pre_l1)
        assert op.warmup(pre_l1, rhs, out=l1_output) is l1_output
        torch.mul(l1_output, 2.0, out=final_output)

        # PyPTO does not synchronize. The caller establishes a completed
        # warmup tail before switching from the ordinary stream to capture.
        torch_npu.npu.synchronize(device_id)
        torch.testing.assert_close(final_output.cpu(), _expected(_REPLAY_INPUTS[0]))
        assert context.prepared
        assert op.warmed

        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph, stream=capture_stream):
            torch.add(static_input, bias, out=pre_l1)
            captured_output = op(pre_l1, rhs, out=l1_output)
            torch.mul(l1_output, 2.0, out=final_output)
        assert captured_output is l1_output

        for value in _REPLAY_INPUTS:
            with torch_npu.npu.stream(capture_stream):
                static_input.fill_(value)
                graph.replay()
            capture_stream.synchronize()
            torch.testing.assert_close(final_output.cpu(), _expected(value))
    finally:
        # A graph may retain PyPTO events, streams, runtime state and tensor
        # addresses. All graph-bound tensor locals remain alive until the graph
        # is quiescent and reset, and the L1 context is closed only afterwards.
        if context is not None:
            torch_npu.npu.synchronize(device_id)
            if graph is not None:
                graph.reset()
            context.close()

    assert context is not None
    assert context.closed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
