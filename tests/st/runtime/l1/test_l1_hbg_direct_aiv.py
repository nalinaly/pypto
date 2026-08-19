# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to LICENSE in the root of the software repository for the full text of the License.

"""A2/A3 onboard coverage for the homogeneous HBG direct-AIV L1 path."""

import time

import pypto.language as pl
import pytest
import torch
import torch_npu
from pypto.l1 import L1InitializationError, pypto_init
from pypto.runtime import RunConfig

_TASK_COUNT = 50
_ELEMENTS_PER_TASK = 128
_REPLAY_VALUES = (2.0, -4.0, 7.5)


@pl.jit
def _hbg_direct_50_task_add(
    lhs: pl.Tensor[[_TASK_COUNT, _ELEMENTS_PER_TASK], pl.FP32],
    rhs: pl.Tensor[[_TASK_COUNT, _ELEMENTS_PER_TASK], pl.FP32],
    out: pl.Out[pl.Tensor[[_TASK_COUNT, _ELEMENTS_PER_TASK], pl.FP32]],
):
    # This A3 launches 48 AIV blocks. Fifty independent tasks force blocks 0
    # and 1 to execute a second grid-stride iteration instead of relying on a
    # one-task-per-core coincidence. This is not the A5 Scalar-worker ABI.
    for task_id in pl.spmd(_TASK_COUNT):
        lhs_tile = pl.load(lhs, [task_id, 0], [1, _ELEMENTS_PER_TASK])
        rhs_tile = pl.load(rhs, [task_id, 0], [1, _ELEMENTS_PER_TASK])
        pl.store(pl.add(lhs_tile, rhs_tile), [task_id, 0], out)
    return out


@pytest.mark.platforms("a2a3")
def test_hbg_direct_aiv_50_tasks_eager_and_aclgraph(
    test_config: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute 50 independent child tasks on 48 AIV lanes and replay them."""
    device_id = test_config.device_id
    torch_npu.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    monkeypatch.setenv("SIMPLER_INTERNAL_HBG_L1_REQUIRE_DIRECT_AIV", "1")
    _hbg_direct_50_task_add._cache.clear()
    compiled = _hbg_direct_50_task_add.compile(
        config=RunConfig(platform="a2a3", device_id=device_id, runtime="host_build_graph")
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
        lhs = torch.full(
            (_TASK_COUNT, _ELEMENTS_PER_TASK),
            _REPLAY_VALUES[0],
            dtype=torch.float32,
            device=device,
        )
        rhs = torch.arange(_TASK_COUNT, dtype=torch.float32, device=device).view(_TASK_COUNT, 1)
        rhs = rhs.expand(_TASK_COUNT, _ELEMENTS_PER_TASK).contiguous()
        added = torch.empty_like(lhs)
        final = torch.empty_like(lhs)
        capture_stream = torch_npu.npu.Stream(device=device_id)

        context.prepare()
        assert op.warmup(lhs, rhs, out=added) is added
        torch.mul(added, 2.0, out=final)
        torch_npu.npu.synchronize(device_id)
        torch.testing.assert_close(final.cpu(), ((lhs + rhs) * 2.0).cpu())

        graph = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(graph, stream=capture_stream):
            op(lhs, rhs, out=added)
            torch.mul(added, 2.0, out=final)

        replay_latencies_us = []
        for value in _REPLAY_VALUES:
            with torch_npu.npu.stream(capture_stream):
                lhs.fill_(value)
                start = time.perf_counter_ns()
                graph.replay()
                capture_stream.synchronize()
                replay_latencies_us.append((time.perf_counter_ns() - start) / 1_000.0)
            torch.testing.assert_close(final.cpu(), ((lhs + rhs) * 2.0).cpu())

        print(
            "A2/A3 HBG direct-AIV 50-task ACLGraph replay latency(us): "
            + ", ".join(f"{latency:.1f}" for latency in replay_latencies_us)
        )
    finally:
        if context is not None:
            torch_npu.npu.synchronize(device_id)
            if graph is not None:
                graph.reset()
            context.close()

    assert context is not None
    assert context.closed


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
