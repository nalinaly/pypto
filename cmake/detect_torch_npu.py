# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Print build metadata for the optional torch_npu L1 adapter.

The script imports torch (an ordinary PyPTO dependency) but deliberately does
not import torch_npu: importing torch_npu can initialize platform state during a
wheel build. Its package root and installed version are discovered as metadata.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path

import torch


def main() -> None:
    spec = importlib.util.find_spec("torch_npu")
    if spec is None or spec.origin is None:
        raise SystemExit(2)

    try:
        torch_npu_version = importlib.metadata.version("torch-npu")
    except importlib.metadata.PackageNotFoundError:
        torch_npu_version = "unknown"

    values = (
        Path(torch.__file__).resolve().parent,
        Path(spec.origin).resolve().parent,
        int(torch._C._GLIBCXX_USE_CXX11_ABI),
        torch.__version__,
        torch_npu_version,
    )
    print("\n".join(str(value) for value in values))


if __name__ == "__main__":
    main()
