# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Canonical Simpler runtime names and compile-time selection rules."""

DEFAULT_RUNTIME = "tensormap_and_ringbuffer"
HOST_BUILD_GRAPH_RUNTIME = "host_build_graph"
SUPPORTED_RUNTIMES = frozenset({DEFAULT_RUNTIME, HOST_BUILD_GRAPH_RUNTIME})


def validate_runtime_name(runtime: object, *, parameter: str = "runtime") -> str:
    """Return a supported runtime name or raise a user-facing validation error."""
    if not isinstance(runtime, str):
        raise TypeError(f"{parameter} must be a string, got {type(runtime).__name__}")
    if runtime not in SUPPORTED_RUNTIMES:
        expected = ", ".join(repr(name) for name in sorted(SUPPORTED_RUNTIMES))
        raise ValueError(f"Invalid {parameter} {runtime!r}. Expected one of: {expected}.")
    return runtime


def resolve_runtime_name(
    runtime: object | None,
    *,
    inherited_runtime: object | None = None,
    inherited_parameter: str = "distributed_config.runtime",
) -> str:
    """Resolve one runtime while rejecting conflicting explicit/inherited sources.

    ``runtime=None`` inherits the runtime carried by a distributed configuration;
    without either source the historical TRB runtime remains the default.  When
    both are present they must agree so generated manifests and execution
    configuration cannot silently select different runtime implementations.
    """
    explicit = None if runtime is None else validate_runtime_name(runtime)
    inherited = (
        None
        if inherited_runtime is None
        else validate_runtime_name(inherited_runtime, parameter=inherited_parameter)
    )
    if explicit is not None and inherited is not None and explicit != inherited:
        raise ValueError(
            f"runtime {explicit!r} conflicts with {inherited_parameter} {inherited!r}; "
            "generated artifacts and execution must use the same runtime"
        )
    return explicit or inherited or DEFAULT_RUNTIME


__all__ = [
    "DEFAULT_RUNTIME",
    "HOST_BUILD_GRAPH_RUNTIME",
    "SUPPORTED_RUNTIMES",
    "resolve_runtime_name",
    "validate_runtime_name",
]
