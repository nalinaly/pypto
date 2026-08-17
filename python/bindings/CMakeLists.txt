# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# nanobind Python bindings

# Collect all binding source files
set(PYPTO_BINDING_SOURCES
    bindings.cpp
    modules/error.cpp
    modules/core.cpp
    modules/logging.cpp
    modules/testing.cpp
    modules/ir.cpp
    modules/ir_builder.cpp
    modules/passes.cpp
    modules/codegen.cpp
    modules/backend.cpp
    modules/arith.cpp
    modules/functor.cpp
)

# Make paths absolute
list(TRANSFORM PYPTO_SOURCES PREPEND "${CMAKE_SOURCE_DIR}/")

# Make paths absolute
list(TRANSFORM PYPTO_BINDING_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

# Export to parent scope
set(PYPTO_BINDING_SOURCES ${PYPTO_BINDING_SOURCES} PARENT_SCOPE)

# Set binding include directory
set(PYPTO_BINDING_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

# Create the nanobind module
nanobind_add_module(${LIBRARY_NAME}
    ${PYPTO_BINDING_SOURCES}
    ${PYPTO_SOURCES}
)

# Set visibility to hidden by default (best practice for Python modules)
set_target_properties(${LIBRARY_NAME} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Define PYPTO_DEBUG only for Debug builds so nanobind leak warnings are
# suppressed in RelWithDebInfo (the default) and Release. Uses a generator
# expression for multi-config generator compatibility.
target_compile_definitions(${LIBRARY_NAME} PRIVATE $<$<CONFIG:Debug>:PYPTO_DEBUG>)

# Link with libbacktrace and msgpack.
# runtime/src/common is added so distributed codegen can consume the runtime
# ABI header `platform_comm/comm_context.h` for compile-time layout asserts
# (see include/pypto/codegen/distributed/comm_layout.h).
target_include_directories(${LIBRARY_NAME} PRIVATE
    ${LIBBACKTRACE_INSTALL_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/runtime/src/common
)
target_link_libraries(${LIBRARY_NAME} PRIVATE
    libbacktrace
    msgpackc-cxx
)

# Handle installation based on build mode
if(SKBUILD_MODE)
    # scikit-build-core: install to the wheel
    install(TARGETS ${LIBRARY_NAME} DESTINATION .)
    # Install dSYM debug symbols on macOS
    if(APPLE)
        install(DIRECTORY $<TARGET_FILE:${LIBRARY_NAME}>.dSYM
            DESTINATION .
            OPTIONAL
        )
    endif()
else()
    # Native CMake: output to local python directory for development
    set_target_properties(${LIBRARY_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/pypto
    )
endif()

# Generate dSYM on macOS for libbacktrace debug symbols
pypto_add_apple_dsymutil(${LIBRARY_NAME})

# Optional taskQueue-aware torch_npu adapter for borrowed-device L1. Keep it a
# separate extension: pypto_core and simpler remain free of torch_npu's private
# C++ ABI, and CPU-only builds continue when torch_npu is absent.
option(PYPTO_BUILD_TORCH_NPU_L1_ADAPTER "Build the optional torch_npu L1 adapter" ON)
if(PYPTO_BUILD_TORCH_NPU_L1_ADAPTER AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    execute_process(
        COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/detect_torch_npu.py
        RESULT_VARIABLE PYPTO_TORCH_NPU_DETECT_RESULT
        OUTPUT_VARIABLE PYPTO_TORCH_NPU_METADATA
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(PYPTO_TORCH_NPU_DETECT_RESULT EQUAL 0)
        string(REPLACE "\n" ";" PYPTO_TORCH_NPU_FIELDS "${PYPTO_TORCH_NPU_METADATA}")
        list(LENGTH PYPTO_TORCH_NPU_FIELDS PYPTO_TORCH_NPU_FIELD_COUNT)
    else()
        set(PYPTO_TORCH_NPU_FIELD_COUNT 0)
    endif()

    if(PYPTO_TORCH_NPU_FIELD_COUNT EQUAL 5)
        list(GET PYPTO_TORCH_NPU_FIELDS 0 PYPTO_TORCH_ROOT)
        list(GET PYPTO_TORCH_NPU_FIELDS 1 PYPTO_TORCH_NPU_ROOT)
        list(GET PYPTO_TORCH_NPU_FIELDS 2 PYPTO_TORCH_CXX11_ABI)
        list(GET PYPTO_TORCH_NPU_FIELDS 3 PYPTO_TORCH_VERSION)
        list(GET PYPTO_TORCH_NPU_FIELDS 4 PYPTO_TORCH_NPU_VERSION)

        set(PYPTO_TORCH_LIB_DIR ${PYPTO_TORCH_ROOT}/lib)
        set(PYPTO_TORCH_NPU_LIB_DIR ${PYPTO_TORCH_NPU_ROOT}/lib)
        set(PYPTO_TORCH_NPU_REQUIRED_FILES
            ${PYPTO_TORCH_ROOT}/include/torch/csrc/api/include/torch/torch.h
            ${PYPTO_TORCH_ROOT}/include/torch/csrc/autograd/python_variable.h
            ${PYPTO_TORCH_NPU_ROOT}/include/torch_npu/csrc/core/npu/NPUCachingAllocator.h
            ${PYPTO_TORCH_NPU_ROOT}/include/torch_npu/csrc/core/npu/NPUStream.h
            ${PYPTO_TORCH_NPU_ROOT}/include/torch_npu/csrc/framework/OpCommand.h
            ${PYPTO_TORCH_NPU_LIB_DIR}/libtorch_npu.so
            ${PYPTO_TORCH_LIB_DIR}/libtorch.so
            ${PYPTO_TORCH_LIB_DIR}/libtorch_cpu.so
            ${PYPTO_TORCH_LIB_DIR}/libtorch_python.so
            ${PYPTO_TORCH_LIB_DIR}/libc10.so
        )
        set(PYPTO_TORCH_NPU_BUILDABLE TRUE)
        foreach(PYPTO_TORCH_NPU_FILE IN LISTS PYPTO_TORCH_NPU_REQUIRED_FILES)
            if(NOT EXISTS ${PYPTO_TORCH_NPU_FILE})
                set(PYPTO_TORCH_NPU_BUILDABLE FALSE)
            endif()
        endforeach()
    else()
        set(PYPTO_TORCH_NPU_BUILDABLE FALSE)
    endif()

    if(PYPTO_TORCH_NPU_BUILDABLE)
        nanobind_add_module(_torch_npu_l1 torch_npu_l1_adapter.cpp)
        target_include_directories(_torch_npu_l1 PRIVATE
            ${PYPTO_TORCH_ROOT}/include
            ${PYPTO_TORCH_ROOT}/include/torch/csrc/api/include
            ${PYPTO_TORCH_NPU_ROOT}/include
            ${PYPTO_TORCH_NPU_ROOT}/include/third_party/acl/inc
            ${PYPTO_TORCH_NPU_ROOT}/include/third_party/hccl/inc
            ${CMAKE_SOURCE_DIR}/runtime/src
        )
        target_compile_definitions(_torch_npu_l1 PRIVATE
            _GLIBCXX_USE_CXX11_ABI=${PYPTO_TORCH_CXX11_ABI}
            PYPTO_L1_BUILD_TORCH_VERSION="${PYPTO_TORCH_VERSION}"
            PYPTO_L1_BUILD_TORCH_NPU_VERSION="${PYPTO_TORCH_NPU_VERSION}"
        )
        target_link_libraries(_torch_npu_l1 PRIVATE
            ${PYPTO_TORCH_NPU_LIB_DIR}/libtorch_npu.so
            ${PYPTO_TORCH_LIB_DIR}/libtorch.so
            ${PYPTO_TORCH_LIB_DIR}/libtorch_cpu.so
            ${PYPTO_TORCH_LIB_DIR}/libtorch_python.so
            ${PYPTO_TORCH_LIB_DIR}/libc10.so
        )
        set_target_properties(_torch_npu_l1 PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
            BUILD_RPATH "${PYPTO_TORCH_LIB_DIR};${PYPTO_TORCH_NPU_LIB_DIR}"
            INSTALL_RPATH "$ORIGIN/../torch/lib;$ORIGIN/../torch_npu/lib"
        )
        if(SKBUILD_MODE)
            install(TARGETS _torch_npu_l1 DESTINATION .)
        else()
            set_target_properties(_torch_npu_l1 PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/pypto
            )
        endif()
        message(STATUS "Building optional PyPTO L1 adapter for torch ${PYPTO_TORCH_VERSION}, torch_npu ${PYPTO_TORCH_NPU_VERSION}")
    else()
        message(STATUS "Skipping optional PyPTO L1 torch_npu adapter: compatible headers/libraries were not found")
    endif()
endif()
