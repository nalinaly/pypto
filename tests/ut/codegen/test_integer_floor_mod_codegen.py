# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Execute generated scalar C++ to check floor remainder, including negative inputs.

The host harness only removes the device header/preamble. Arithmetic and control
flow emitted by PyPTO/PTOAS are compiled unchanged; this is not an NPU test.
"""

import ctypes
import os
import re
import shutil
import subprocess

import pytest
from pypto import DataType, backend, codegen, ir
from pypto.arith import Analyzer
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.op import tensor as tensor_ops


@pytest.fixture(autouse=True)
def _backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _compile_scalar(tmp_path, dtype, expression):
    compiler = os.environ.get("CXX") or shutil.which("c++")
    if not compiler:
        pytest.skip("Host C++ compiler is unavailable")
    ptoas = find_ptoas_binary()
    if not ptoas:
        pytest.skip("PTOAS is unavailable")
    span = ir.Span.unknown()
    scalar = ir.ScalarType(dtype)
    a = ir.Var("a", scalar, span)
    b = ir.Var("b", scalar, span)
    output_dtype = DataType.INT64 if dtype == DataType.INDEX else dtype
    tensor_type = ir.TensorType([1], output_dtype)
    out = ir.Var("out", tensor_type, span)
    result = ir.Var("result", scalar, span)
    written = ir.Var("written", tensor_type, span)
    body = ir.SeqStmts(
        [ir.AssignStmt(result, expression(a, b, span), span),
         ir.AssignStmt(written, tensor_ops.write(out, [0], ir.Cast(result, output_dtype, span)), span),
         ir.ReturnStmt([written], span)], span
    )
    function = ir.Function("floor_mod", [out, a, b], [tensor_type], body, span, ir.FunctionType.InCore)
    generated = codegen.PTOCodegen().generate(ir.Program([function], "floor_mod_program", span))
    mlir = generated if isinstance(generated, str) else "".join(generated.values())
    pto = tmp_path / "floor_mod.pto"
    cpp = tmp_path / "floor_mod.cpp"
    pto.write_text(mlir)
    subprocess.run([str(ptoas), str(pto), "-o", str(cpp)], check=True, capture_output=True, text=True)
    source = cpp.read_text()
    match = re.search(r"AICORE \w+ floor_mod\(", source)
    assert match, source
    function_cpp = source[match.start():].replace("AICORE ", 'extern "C" ', 1)
    host = tmp_path / "host.cpp"
    host.write_text("""#include <cstdint>
#define __gm__
#define PIPE_ALL 0
enum class cache_line_t {ENTIRE_DATA_CACHE, SINGLE_CACHE_LINE};
using mem_dsb_t = int;
inline void pipe_barrier(int) {}
inline void dcci(void*, cache_line_t) {}
inline void dsb(mem_dsb_t) {}
""" + function_cpp)
    library = tmp_path / "host.so"
    subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", str(host), "-o", str(library)],
                   check=True, capture_output=True, text=True)
    loaded = ctypes.CDLL(str(library))
    fn = loaded.floor_mod
    ctype = ctypes.c_int32 if dtype == DataType.INT32 else ctypes.c_int64
    fn.argtypes = [ctypes.POINTER(ctype), ctype, ctype]
    fn.restype = None
    fn._library = loaded

    def execute(left, right):
        output = ctype()
        fn(ctypes.byref(output), left, right)
        return output.value

    return execute


@pytest.mark.parametrize("dtype", [DataType.INT32, DataType.INT64, DataType.INDEX])
@pytest.mark.parametrize("divisor", [None, 24, -24])
def test_signed_floor_mod_matches_python(tmp_path, dtype, divisor):
    def expr(a, b, span):
        right = b if divisor is None else ir.ConstInt(divisor, dtype, span)
        return ir.FloorMod(a, right, dtype, span)

    fn = _compile_scalar(tmp_path, dtype, expr)
    for left in [-97, -48, -47, -25, -24, -1, 0, 1, 23, 24, 25, 47, 48, 97]:
        for right in ([divisor] if divisor is not None else [-24, -7, -1, 1, 7, 24]):
            assert fn(left, right) == left % right, (left, right)
    bits = 32 if dtype == DataType.INT32 else 64
    for left, right in [(-(2 ** (bits - 1)), 24), (2 ** (bits - 1) - 1, -24)]:
        actual_right = divisor if divisor is not None else right
        assert fn(left, actual_right) == left % actual_right


def test_simplified_query_owner_preserves_nonnegative_offset(tmp_path):
    dtype = DataType.INDEX

    def expr(worker, begin, span):
        n = ir.ConstInt(24, dtype, span)
        difference = ir.Sub(worker, begin, dtype, span)
        wrapped = ir.FloorMod(ir.Add(ir.FloorMod(difference, n, dtype, span), n, dtype, span),
                              n, dtype, span)
        return Analyzer().simplify(ir.Add(begin, wrapped, dtype, span))

    fn = _compile_scalar(tmp_path, dtype, expr)
    for begin in [0, 8, 16, 24, 40, 48, 72, 96, 128]:
        for worker in range(24):
            expected = begin + (worker - begin) % 24
            assert fn(worker, begin) == expected, (worker, begin)
