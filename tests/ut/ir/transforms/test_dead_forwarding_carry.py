# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Dead forwarding cycles must not retain tiles; semantic users still keep carries."""

import pypto.language as pl
from pypto import ir, passes


class _Inventory(ir.IRVisitor):
    def __init__(self):
        super().__init__()
        self.loops = []
        self.assigns = []

    def visit_for_stmt(self, stmt):
        self.loops.append(stmt)
        super().visit_for_stmt(stmt)

    def visit_while_stmt(self, stmt):
        self.loops.append(stmt)
        super().visit_while_stmt(stmt)

    def visit_assign_stmt(self, stmt):
        self.assigns.append(stmt)
        super().visit_assign_stmt(stmt)


def _simplified(program):
    after = passes.simplify()(program)
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.UseAfterDef)
    errors = [
        diagnostic
        for diagnostic in passes.PropertyVerifierRegistry.verify(props, after)
        if diagnostic.severity == passes.DiagnosticSeverity.Error
    ]
    assert not errors, [diagnostic.message for diagnostic in errors]
    ir.assert_structural_equal(passes.simplify()(after), after)
    inventory = _Inventory()
    for function in after.functions.values():
        inventory.visit_stmt(function.body)
    return after, inventory


def test_dead_tile_forwarded_through_if_phi_is_removed():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, src: pl.Tensor[[1, 32], pl.FP32], n: pl.Scalar[pl.INDEX],
                   out: pl.Tensor[[1, 32], pl.FP32]):
            seed = pl.tile.load(src, [0, 0], [1, 32])
            ghost = pl.tile.create([6, 32], dtype=pl.FP32)
            for i, (live, unused) in pl.range(n, init_values=(seed, ghost)):
                if i % 2 == 0:
                    forwarded = pl.yield_(unused)
                else:
                    forwarded = pl.yield_(unused)
                updated = pl.tile.add(live, 1.0)
                result, _ghost_result = pl.yield_(updated, forwarded)
            pl.tile.store(result, [0, 0], out)

    _, inventory = _simplified(Before)
    assert len(inventory.loops) == 1
    assert len(inventory.loops[0].iter_args) == 1
    create_name = ir.get_op("tile.create").name
    assert not any(
        isinstance(stmt.value, ir.Call) and stmt.value.op.name == create_name
        for stmt in inventory.assigns
    )


def test_dead_tile_forwarded_through_plain_alias_is_removed():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, src: pl.Tensor[[1, 32], pl.FP32], n: pl.Scalar[pl.INDEX],
                   out: pl.Tensor[[1, 32], pl.FP32]):
            seed = pl.tile.load(src, [0, 0], [1, 32])
            ghost = pl.tile.create([6, 32], dtype=pl.FP32)
            for i, (live, unused) in pl.range(n, init_values=(seed, ghost)):
                renamed = unused
                updated = pl.tile.add(live, 1.0)
                result, _ghost_result = pl.yield_(updated, renamed)
            pl.tile.store(result, [0, 0], out)

    _, inventory = _simplified(Before)
    assert len(inventory.loops[0].iter_args) == 1


def test_tile_read_in_loop_keeps_forwarded_state():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, src: pl.Tensor[[6, 32], pl.FP32], n: pl.Scalar[pl.INDEX],
                   out: pl.Tensor[[6, 32], pl.FP32]):
            seed = pl.tile.load(src, [0, 0], [6, 32])
            for i, (live,) in pl.range(n, init_values=(seed,)):
                renamed = live
                pl.tile.store(renamed, [0, 0], out)
                _ignored_result = pl.yield_(renamed)

    _, inventory = _simplified(Before)
    assert len(inventory.loops[0].iter_args) == 1


def test_post_loop_read_keeps_initializer_alias_for_zero_iterations():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, src: pl.Tensor[[1, 32], pl.FP32], n: pl.Scalar[pl.INDEX],
                   out: pl.Tensor[[1, 32], pl.FP32]):
            seed = pl.tile.load(src, [0, 0], [1, 32])
            initial = seed
            for i, (live,) in pl.range(n, init_values=(initial,)):
                fresh = pl.tile.full([1, 32], dtype=pl.FP32, value=2.0)
                result = pl.yield_(fresh)
            pl.tile.store(result, [0, 0], out)

    _, inventory = _simplified(Before)
    assert len(inventory.loops[0].iter_args) == 1
    assert any(stmt.var.name_hint == "initial" for stmt in inventory.assigns)


def test_while_condition_keeps_scalar_but_drops_dead_tile():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, n: pl.Scalar[pl.INDEX], out: pl.Tensor[[1], pl.INDEX]):
            zero: pl.Scalar[pl.INDEX] = 0
            ghost = pl.tile.create([6, 32], dtype=pl.FP32)
            for (i, unused) in pl.while_(init_values=(zero, ghost)):
                pl.cond(i < n)
                renamed = unused
                next_i: pl.Scalar[pl.INDEX] = i + 1
                final_i, _ghost_result = pl.yield_(next_i, renamed)
            pl.tensor.write(out, [0], final_i)

    _, inventory = _simplified(Before)
    assert len(inventory.loops) == 1
    assert isinstance(inventory.loops[0], ir.WhileStmt)
    assert len(inventory.loops[0].iter_args) == 1


def test_unused_task_id_channel_survives():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, tid: pl.Scalar[pl.TASK_ID], n: pl.Scalar[pl.INDEX]):
            for i, (carried_tid,) in pl.range(n, init_values=(tid,)):
                renamed = carried_tid
                _final_tid = pl.yield_(renamed)

    _, inventory = _simplified(Before)
    assert len(inventory.loops[0].iter_args) == 1


def test_spmd_core_count_is_a_semantic_use():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, count: pl.Scalar[pl.INDEX], n: pl.Scalar[pl.INDEX],
                 out: pl.Tensor[[32], pl.INDEX]):
            for i, (workers,) in pl.range(n, init_values=(count,)):
                renamed = workers
                with pl.spmd(renamed):
                    block = pl.tile.get_block_idx()
                    pl.tensor.write(out, [block], pl.const(1, pl.INDEX))
                _final_workers = pl.yield_(renamed)

    _, inventory = _simplified(Before)
    assert len(inventory.loops[0].iter_args) == 1
