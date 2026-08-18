# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device-only no-reset fault matrix for HBG L1 generation teardown."""

import os

import pypto.language as pl
import pytest
import torch
import torch_npu
from harness.core.harness import ONBOARD_PLATFORMS
from pypto.l1 import L1InitializationError, pypto_init
from pypto.runtime import RunConfig

_ROWS = 64
_COLS = 128
_FAULT_ENV = "SIMPLER_INTERNAL_HBG_L1_TEST_FAULT"
_FAULT_STAGES = (
    "restore_copy",
    "restore_publish",
    "after_scheduler_init",
    "before_classify",
    "before_dispatch",
    "shutdown",
    "runtime_destroy",
    "scheduler_init",
    "scheduler_assign",
    "scheduler_dispatch",
    "platform_bridge",
    "affinity_inputs",
    "kernel_args_runtime",
    "physical_core_mapping",
    "physical_core_id",
    "slot_fallback_control",
)


@pl.jit
def _fault_add(
    lhs: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    rhs: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[_ROWS, _COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs_tile = pl.load(lhs, [0, 0], [_ROWS, _COLS])
        rhs_tile = pl.load(rhs, [0, 0], [_ROWS, _COLS])
        pl.store(pl.add(lhs_tile, rhs_tile), [0, 0], out)
    return out


@pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
def test_hbg_l1_generation_faults_recover_without_device_reset(
    test_config: RunConfig,
    platform: str,
) -> None:
    """Every controlled abort reaches the caller tail and leaves one reusable context."""
    device_id = test_config.device_id
    torch_npu.npu.set_device(device_id)
    _fault_add._cache.clear()
    compiled = _fault_add.compile(
        config=RunConfig(platform=platform, device_id=device_id, runtime="host_build_graph")
    )

    context = None
    graph = None
    try:
        try:
            context = pypto_init(programs=[compiled], device=device_id)
        except L1InitializationError as exc:
            context = exc.cleanup_context
            raise

        op = context.operator(compiled)
        device = torch.device(f"npu:{device_id}")
        lhs = torch.full((_ROWS, _COLS), 2.0, dtype=torch.float32, device=device)
        rhs = torch.full((_ROWS, _COLS), 5.0, dtype=torch.float32, device=device)
        out = torch.empty((_ROWS, _COLS), dtype=torch.float32, device=device)
        expected = torch.full((_ROWS, _COLS), 7.0, dtype=torch.float32)

        context.prepare()
        op.warmup(lhs, rhs, out=out)
        torch_npu.npu.synchronize(device_id)
        torch.testing.assert_close(out.cpu(), expected)

        for stage in _FAULT_STAGES:
            out.fill_(-777.0)
            os.environ[_FAULT_ENV] = stage
            op(lhs, rhs, out=out)
            # Keep the environment set until taskQueue has executed the
            # callback and the caller tail proves hidden-AICore completion.
            torch_npu.npu.synchronize(device_id)
            os.environ.pop(_FAULT_ENV, None)

            if stage != "scheduler_dispatch":
                torch.testing.assert_close(out.cpu(), torch.full_like(expected, -777.0))

            # The immediately following generation must fully restore the
            # working slot and execute without context/device reset.
            out.fill_(-333.0)
            op(lhs, rhs, out=out)
            torch_npu.npu.synchronize(device_id)
            torch.testing.assert_close(out.cpu(), expected)

        capture_stream = torch_npu.npu.Stream(device=device_id)
        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph, stream=capture_stream):
            op(lhs, rhs, out=out)
        for value in (11.0, -3.0):
            with torch_npu.npu.stream(capture_stream):
                lhs.fill_(value)
                graph.replay()
            capture_stream.synchronize()
            torch.testing.assert_close(out.cpu(), torch.full_like(expected, value + 5.0))
    finally:
        os.environ.pop(_FAULT_ENV, None)
        if context is not None:
            torch_npu.npu.synchronize(device_id)
            if graph is not None:
                graph.reset()
            context.close()

    assert context is not None
    assert context.closed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
