# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Call a tiled (non-SPMD) add as a torch_npu L1 op, then chain torch.mul.

Latest PyPTO exposes this path as :func:`pypto.l1.pypto_init` + :class:`L1Operator`.
The caller owns the NPU device, tensors and stream; PyPTO only enqueues.

L1 v1 requires a positive static shape on the compiled artifact
(``pypto/runtime/l1.py``). ``pl.dynamic("M")`` lowers to ``-1`` and is rejected
at ``pypto_init``. Host-known shape change is done here by specializing the
``@pl.jit`` kernel for this launch's ``ROWS`` (a new compile / cache key), then
unrolling ``pl.range`` tiles — the same orch-for pattern as the dynamic add,
but with a concrete extent L1 will accept.

    result = torch.mul(pypto_tiled_add(a, b), scale)

Run (onboard only; L1 has no simulator)::

    source /mnt/workspace/inductor/env.sh
    python examples/runtime/l1_tiled_add_then_mul.py --device 0
"""

from __future__ import annotations

import argparse
import importlib

import pypto.language as pl
import torch
from pypto.l1 import L1InitializationError, pypto_init
from pypto.runtime import RunConfig

COLS = 128
TILE = 128


def make_tiled_add(rows: int, cols: int = COLS, tile: int = TILE):
    """Build a non-SPMD add that unrolls ``ceil(rows/tile)`` single-block tasks."""

    n_tiles = (rows + tile - 1) // tile

    @pl.jit
    def tiled_add(
        a: pl.Tensor[[rows, cols], pl.FP32],
        b: pl.Tensor[[rows, cols], pl.FP32],
        out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
    ):
        for i in pl.range(n_tiles):
            remain = rows - i * tile
            vrows = tile if remain >= tile else remain
            with pl.at(level=pl.Level.CORE_GROUP):
                tile_a = pl.load(a, [i * tile, 0], [tile, cols], valid_shape=[vrows, cols])
                tile_b = pl.load(b, [i * tile, 0], [tile, cols], valid_shape=[vrows, cols])
                pl.store(pl.add(tile_a, tile_b), [i * tile, 0], out, valid_shape=[vrows, cols])
        return out

    tiled_add._cache.clear()
    return tiled_add


def main() -> None:
    torch_npu = importlib.import_module("torch_npu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=256, help="Host-known M for this compile")
    parser.add_argument(
        "--runtime",
        default="host_build_graph",
        choices=("host_build_graph", "tensormap_and_ringbuffer"),
    )
    parser.add_argument("--platform", default="a2a3", choices=("a2a3", "a5"))
    parser.add_argument("--scale", type=float, default=2.0)
    args = parser.parse_args()

    device = args.device
    rows = args.rows
    torch_npu.npu.set_device(device)
    npu = torch.device(f"npu:{device}")

    tiled_add = make_tiled_add(rows)
    compiled = tiled_add.compile(
        config=RunConfig(platform=args.platform, device_id=device, runtime=args.runtime)
    )

    a = torch.full((rows, COLS), 2.0, dtype=torch.float32, device=npu)
    b = torch.full((rows, COLS), 3.0, dtype=torch.float32, device=npu)
    added = torch.empty((rows, COLS), dtype=torch.float32, device=npu)
    result = torch.empty((rows, COLS), dtype=torch.float32, device=npu)

    ctx = None
    try:
        try:
            ctx = pypto_init(programs=[compiled], device=device)
        except L1InitializationError as exc:
            ctx = exc.cleanup_context
            raise

        op = ctx.operator(compiled)
        ctx.prepare()
        op.warmup(a, b, out=added)
        torch.mul(added, args.scale, out=result)
        torch_npu.npu.synchronize(device)

        expected = (a + b) * args.scale
        torch.testing.assert_close(result.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)
        print(
            f"OK  runtime={ctx.runtime}  M={rows}  "
            f"tiles={(rows + TILE - 1) // TILE}  "
            f"result[0,0]={result[0, 0].item()}"
        )
    finally:
        torch_npu.npu.synchronize(device)
        if ctx is not None:
            ctx.close()


if __name__ == "__main__":
    main()
