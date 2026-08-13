# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for Gloo + torch_npu SHMEM window helpers and live-input goldens."""

import torch
from pypto.runtime.shmem_gloo import (
    COMM_CONTEXT_SIZE,
    align_up,
    allreduce_sum_expected,
    carve_window_layout,
    pack_comm_context,
    peer_copy_expected,
    unpack_comm_context,
)


def test_align_up() -> None:
    assert align_up(1) == 32
    assert align_up(32) == 32
    assert align_up(33) == 64


def test_carve_window_layout_matches_codegen_alignment() -> None:
    offsets, total = carve_window_layout([2048, 2048, 8])
    assert offsets == [0, 2048, 4096]
    assert total == 4096 + 32


def test_pack_comm_context_roundtrip() -> None:
    bases = [0x1000, 0x2000]
    blob = pack_comm_context(rank=1, world_size=2, win_size=4128, window_bases=bases)
    assert len(blob) == COMM_CONTEXT_SIZE
    decoded = unpack_comm_context(blob)
    assert decoded["rank"] == 1
    assert decoded["world_size"] == 2
    assert decoded["win_size"] == 4128
    assert decoded["window_bases"] == bases


def test_peer_copy_golden_from_live_inputs() -> None:
    mine = torch.arange(8, dtype=torch.float16)
    peer = torch.arange(8, 16, dtype=torch.float16)
    got = peer_copy_expected(mine, peer)
    assert torch.equal(got, peer)
    assert not torch.equal(got, mine)


def test_import_window_buffer_is_exported() -> None:
    import pypto.language.distributed as pld

    assert hasattr(pld, "import_window_buffer")
    assert hasattr(pld.tensor, "import_window_buffer")


def test_allreduce_sum_golden_from_live_inputs() -> None:
    mine = torch.arange(8, dtype=torch.float16)
    peer = torch.ones(8, dtype=torch.float16)
    got = allreduce_sum_expected(mine, peer)
    assert torch.allclose(got, mine + peer)
