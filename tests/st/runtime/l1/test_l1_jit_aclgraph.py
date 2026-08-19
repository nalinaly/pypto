# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A2/A3 acceptance test for the Triton-style L1 JIT facade.

Set ``PYPTO_L1_JIT_TEST_RUNTIME`` to run one runtime per fresh pytest process.
The default is TRB; HBG is intentionally a separate process because L1 code
and CANN function handles remain process-pinned after ``shutdown()``.
"""

import os

import pypto
import pypto.language as pl
import pytest
import torch
import torch_npu
from pypto.runtime import RunConfig

_ROWS = 64
_COLS = 128
_RUNTIME = os.environ.get("PYPTO_L1_JIT_TEST_RUNTIME", "tensormap_and_ringbuffer")
_REPLAY_INPUTS = (2.0, -4.0, 7.5)


@pl.jit(execution="l1", runtime=_RUNTIME)
def _jit_add(
    lhs: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    rhs: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[_ROWS, _COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs_tile = pl.load(lhs, [0, 0], [_ROWS, _COLS])
        rhs_tile = pl.load(rhs, [0, 0], [_ROWS, _COLS])
        pl.store(pl.add(lhs_tile, rhs_tile), [0, 0], out)
    return out


@pl.jit(execution="l1", runtime=_RUNTIME)
def _jit_mul(
    lhs: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    rhs: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[_ROWS, _COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs_tile = pl.load(lhs, [0, 0], [_ROWS, _COLS])
        rhs_tile = pl.load(rhs, [0, 0], [_ROWS, _COLS])
        pl.store(pl.mul(lhs_tile, rhs_tile), [0, 0], out)
    return out


def _expected(value: float) -> torch.Tensor:
    return torch.full((_ROWS, _COLS), (value + 1.0 + 3.0) * 2.0 + 1.0, dtype=torch.float32)


def test_l1_jit_eager_allocator_and_aclgraph_replay(test_config: RunConfig) -> None:
    """Use only the public JIT call shape for eager and captured execution."""
    assert test_config.platform == "a2a3"
    device_id = test_config.device_id
    torch_npu.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")

    source = torch.full((_ROWS, _COLS), _REPLAY_INPUTS[0], dtype=torch.float32, device=device)
    bias = torch.ones((_ROWS, _COLS), dtype=torch.float32, device=device)
    rhs = torch.full((_ROWS, _COLS), 3.0, dtype=torch.float32, device=device)
    factor = torch.full((_ROWS, _COLS), 2.0, dtype=torch.float32, device=device)
    pre_l1 = torch.empty((_ROWS, _COLS), dtype=torch.float32, device=device)
    add_output = torch.empty((_ROWS, _COLS), dtype=torch.float32, device=device)
    mul_output = torch.empty((_ROWS, _COLS), dtype=torch.float32, device=device)
    final_output = torch.empty((_ROWS, _COLS), dtype=torch.float32, device=device)
    graph = None

    try:
        # First ordinary call implicitly initializes/prepares and lets the
        # PyTorch wrapper allocate the pure output through torch.empty().
        eager_output = _jit_add(source, rhs)
        torch_npu.npu.synchronize(device_id)
        torch.testing.assert_close(eager_output.cpu(), torch.full_like(_expected(0.0), 5.0))

        # Discover a second callable only after the first one has executed.
        # This exercises append-after-warmup instead of a batch-prepared table.
        eager_mul = _jit_mul(eager_output, factor)
        torch_npu.npu.synchronize(device_id)
        torch.testing.assert_close(eager_mul.cpu(), torch.full_like(_expected(0.0), 10.0))

        capture_stream = torch_npu.npu.Stream(device=device_id)
        assert capture_stream.npu_stream != torch_npu.npu.current_stream(device_id).npu_stream
        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph, stream=capture_stream):
            torch.add(source, bias, out=pre_l1)
            captured_add = _jit_add(pre_l1, rhs, out=add_output)
            captured_mul = _jit_mul(captured_add, factor, out=mul_output)
            torch.add(captured_mul, 1.0, out=final_output)
        assert captured_add is add_output
        assert captured_mul is mul_output

        for value in _REPLAY_INPUTS:
            with torch_npu.npu.stream(capture_stream):
                source.fill_(value)
                graph.replay()
            capture_stream.synchronize()
            torch.testing.assert_close(final_output.cpu(), _expected(value))
    finally:
        # shutdown() never synchronizes. The caller proves quiescence and
        # destroys the graph before explicitly retiring the device owner.
        torch_npu.npu.synchronize(device_id)
        if graph is not None:
            graph.reset()
        pypto.l1.shutdown(device=device_id)


if __name__ == "__main__":
    pytest.main([__file__])
