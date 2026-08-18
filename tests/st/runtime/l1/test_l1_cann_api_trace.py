# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A2/A3 CANN-symbol trace for the borrowed-L1 launch contract."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path
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
_TRACE_ABI_VERSION = 1
_TRACE_RECORD_CAPACITY = 32
_TRACE_CHILD_ENV = "PYPTO_L1_CANN_TRACE_CHILD"
_TRACE_LIBRARY_ENV = "PYPTO_L1_CANN_TRACE_LIBRARY"

_MEMSET = 1
_RECORD_EVENT = 2
_WAIT_EVENT = 3
_LAUNCH_AICPU = 4
_LAUNCH_AICORE = 5
_QUERY_EVENT = 6
_FORK_JOIN_OPERATIONS = (
    _MEMSET,
    _MEMSET,
    _RECORD_EVENT,
    _WAIT_EVENT,
    _LAUNCH_AICORE,
    _RECORD_EVENT,
    _LAUNCH_AICPU,
    _WAIT_EVENT,
    _RECORD_EVENT,
)


class _TraceRecord(ctypes.Structure):
    _fields_ = [
        ("operation", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("stream", ctypes.c_uint64),
        ("object", ctypes.c_uint64),
    ]


class _TraceSnapshot(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint64),
        ("expected_caller_stream", ctypes.c_uint64),
        ("stream_sync_calls", ctypes.c_uint64),
        ("device_sync_calls", ctypes.c_uint64),
        ("capture_api_calls", ctypes.c_uint64),
        ("model_attach_calls", ctypes.c_uint64),
        ("resource_lifecycle_calls", ctypes.c_uint64),
        ("device_allocation_calls", ctypes.c_uint64),
        ("aicpu_launch_calls", ctypes.c_uint64),
        ("aicore_launch_calls", ctypes.c_uint64),
        ("private_aicpu_stream_calls", ctypes.c_uint64),
        ("caller_stream_aicore_calls", ctypes.c_uint64),
        ("early_aicpu_launch_calls", ctypes.c_uint64),
        ("record_count", ctypes.c_uint64),
        ("record_overflow", ctypes.c_uint64),
        ("records", _TraceRecord * _TRACE_RECORD_CAPACITY),
    ]


class _TraceLibrary:
    def __init__(self, path: Path) -> None:
        self._library = ctypes.CDLL(str(path))
        self._library.pypto_l1_cann_trace_begin.argtypes = [ctypes.c_uint64]
        self._library.pypto_l1_cann_trace_begin.restype = None
        self._library.pypto_l1_cann_trace_end.argtypes = [
            ctypes.POINTER(_TraceSnapshot),
            ctypes.c_size_t,
        ]
        self._library.pypto_l1_cann_trace_end.restype = ctypes.c_int

    def begin(self, caller_stream: int) -> None:
        self._library.pypto_l1_cann_trace_begin(caller_stream)

    def end(self) -> _TraceSnapshot:
        snapshot = _TraceSnapshot()
        rc = self._library.pypto_l1_cann_trace_end(
            ctypes.byref(snapshot),
            ctypes.sizeof(snapshot),
        )
        assert rc == 0
        return snapshot


@pl.jit
def _l1_add(
    lhs: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    rhs: pl.Tensor[[_ROWS, _COLS], pl.FP32],
    output: pl.Out[pl.Tensor[[_ROWS, _COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs_tile = pl.load(lhs, [0, 0], [_ROWS, _COLS])
        rhs_tile = pl.load(rhs, [0, 0], [_ROWS, _COLS])
        pl.store(pl.add(lhs_tile, rhs_tile), [0, 0], output)
    return output


def _build_trace_library(output: Path) -> None:
    compiler = shutil.which("g++")
    assert compiler is not None, "g++ is required to build the L1 CANN API tracer"
    source = Path(__file__).with_name("support") / "l1_cann_api_trace.cpp"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-shared",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source),
            "-ldl",
            "-o",
            str(output),
        ],
        check=True,
    )


def _run_preloaded_child(test_config: RunConfig, trace_library: Path) -> None:
    environment = os.environ.copy()
    environment[_TRACE_CHILD_ENV] = "1"
    environment[_TRACE_LIBRARY_ENV] = str(trace_library)
    existing_preload = environment.get("LD_PRELOAD")
    environment["LD_PRELOAD"] = (
        f"{trace_library}:{existing_preload}" if existing_preload else str(trace_library)
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-s",
            str(Path(__file__).resolve()),
            "--platform=a2a3",
            f"--device={test_config.device_id}",
        ],
        env=environment,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    assert result.returncode == 0, result.stdout + result.stderr


def _assert_trace(
    snapshot: _TraceSnapshot,
    expected_caller_stream: int,
    *,
    expect_tail_query: bool,
    label: str,
) -> None:
    assert snapshot.abi_version == _TRACE_ABI_VERSION, label
    assert snapshot.expected_caller_stream == expected_caller_stream, label
    forbidden_counts = {
        "stream_sync": snapshot.stream_sync_calls,
        "device_sync": snapshot.device_sync_calls,
        "capture_api": snapshot.capture_api_calls,
        "model_attach": snapshot.model_attach_calls,
        "resource_lifecycle": snapshot.resource_lifecycle_calls,
        "device_allocation": snapshot.device_allocation_calls,
        "private_aicpu_stream": snapshot.private_aicpu_stream_calls,
        "caller_stream_aicore": snapshot.caller_stream_aicore_calls,
        "early_aicpu_launch": snapshot.early_aicpu_launch_calls,
    }
    assert forbidden_counts == dict.fromkeys(forbidden_counts, 0), f"{label}: {forbidden_counts}"
    assert snapshot.aicpu_launch_calls == 1, label
    assert snapshot.aicore_launch_calls == 1, label
    assert snapshot.record_overflow == 0, label
    assert snapshot.record_count <= _TRACE_RECORD_CAPACITY, label

    records = list(snapshot.records[: snapshot.record_count])
    operations = tuple(record.operation for record in records)
    expected_operations = ((_QUERY_EVENT,) if expect_tail_query else ()) + _FORK_JOIN_OPERATIONS
    assert operations == expected_operations, f"{label}: {operations}"
    if expect_tail_query:
        records = records[1:]

    caller = expected_caller_stream
    hidden = records[3].stream
    assert hidden not in (0, caller), label
    assert records[0].stream == caller and records[1].stream == caller, label
    assert records[2].stream == caller, label
    assert records[3].stream == hidden and records[4].stream == hidden and records[5].stream == hidden, label
    assert records[6].stream == caller and records[7].stream == caller and records[8].stream == caller, label
    assert records[2].object == records[3].object, label
    assert records[5].object == records[7].object, label


def _close_context(context: Any, graph: Any, device_id: int) -> None:
    if context is None:
        return
    torch_npu.npu.synchronize(device_id)
    if graph is not None:
        graph.reset()
    context.close()


def _trace_runtime(runtime: str, test_config: RunConfig, trace: _TraceLibrary) -> None:
    device_id = test_config.device_id
    _l1_add._cache.clear()
    program = _l1_add.compile(config=RunConfig(platform="a2a3", device_id=device_id, runtime=runtime))
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
        lhs = torch.full(_SHAPE, 2.0, dtype=torch.float32, device=device)
        rhs = torch.full(_SHAPE, 3.0, dtype=torch.float32, device=device)
        output = torch.empty(_SHAPE, dtype=torch.float32, device=device)

        context.prepare()
        op.warmup(lhs, rhs, out=output)
        torch_npu.npu.synchronize(device_id)

        eager_stream = torch_npu.npu.current_stream(device_id)
        trace.begin(eager_stream.npu_stream)
        op(lhs, rhs, out=output)
        torch_npu.npu.synchronize(device_id)
        eager_snapshot = trace.end()
        _assert_trace(
            eager_snapshot,
            eager_stream.npu_stream,
            expect_tail_query=False,
            label=f"{runtime} eager",
        )
        torch.testing.assert_close(output.cpu(), torch.full(_SHAPE, 5.0))

        capture_stream = torch_npu.npu.Stream(device=device_id)
        graph = torch_npu.npu.NPUGraph()
        trace.begin(capture_stream.npu_stream)
        with torch_npu.npu.graph(graph, stream=capture_stream):
            op(lhs, rhs, out=output)
        capture_snapshot = trace.end()
        _assert_trace(
            capture_snapshot,
            capture_stream.npu_stream,
            expect_tail_query=True,
            label=f"{runtime} capture",
        )

        with torch_npu.npu.stream(capture_stream):
            lhs.fill_(7.0)
            graph.replay()
        capture_stream.synchronize()
        torch.testing.assert_close(output.cpu(), torch.full(_SHAPE, 10.0))
    finally:
        _close_context(context, graph, device_id)

    assert context is not None
    assert context.closed


@pytest.mark.parametrize("platform", ("a2a3",))
def test_l1_launch_cann_api_trace_is_capture_transparent(
    test_config: RunConfig,
    tmp_path: Path,
    platform: str,
) -> None:
    """Trace real host-runtime CANN calls without counting torch_npu graph APIs."""
    assert platform == "a2a3"
    if os.environ.get(_TRACE_CHILD_ENV) != "1":
        trace_library = tmp_path / "libpypto_l1_cann_api_trace.so"
        _build_trace_library(trace_library)
        _run_preloaded_child(test_config, trace_library)
        return

    trace_library = Path(os.environ[_TRACE_LIBRARY_ENV])
    trace = _TraceLibrary(trace_library)
    torch_npu.npu.set_device(test_config.device_id)
    for runtime in ("tensormap_and_ringbuffer", "host_build_graph"):
        _trace_runtime(runtime, test_config, trace)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
