/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <dlfcn.h>
#include <pthread.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

constexpr uint64_t kTraceAbiVersion = 1;
constexpr size_t kMaxTraceRecords = 32;

enum class TraceOperation : uint32_t {
  Memset = 1,
  RecordEvent = 2,
  WaitEvent = 3,
  LaunchAicpu = 4,
  LaunchAicore = 5,
  QueryEvent = 6,
};

struct TraceRecord {
  uint32_t operation;
  uint32_t reserved;
  uint64_t stream;
  uint64_t object;
};

struct TraceSnapshot {
  uint64_t abi_version;
  uint64_t expected_caller_stream;
  uint64_t stream_sync_calls;
  uint64_t device_sync_calls;
  uint64_t capture_api_calls;
  uint64_t model_attach_calls;
  uint64_t resource_lifecycle_calls;
  uint64_t device_allocation_calls;
  uint64_t aicpu_launch_calls;
  uint64_t aicore_launch_calls;
  uint64_t private_aicpu_stream_calls;
  uint64_t caller_stream_aicore_calls;
  uint64_t early_aicpu_launch_calls;
  uint64_t record_count;
  uint64_t record_overflow;
  TraceRecord records[kMaxTraceRecords];
};

pthread_mutex_t g_trace_mutex = PTHREAD_MUTEX_INITIALIZER;
std::atomic<bool> g_trace_enabled{false};
TraceSnapshot g_trace{};
bool g_saw_caller_start_record = false;

bool called_from_host_runtime(void* return_address) noexcept {
  if (!g_trace_enabled.load(std::memory_order_acquire)) return false;
  Dl_info info{};
  return dladdr(return_address, &info) != 0 && info.dli_fname != nullptr &&
         std::strstr(info.dli_fname, "libhost_runtime.so") != nullptr;
}

template <typename Function>
Function next_symbol(const char* name) noexcept {
  return reinterpret_cast<Function>(dlsym(RTLD_NEXT, name));
}

void append_record(TraceOperation operation, void* stream, void* object, void* return_address) noexcept {
  if (!called_from_host_runtime(return_address)) return;
  pthread_mutex_lock(&g_trace_mutex);
  if (g_trace_enabled.load(std::memory_order_relaxed)) {
    if (g_trace.record_count < kMaxTraceRecords) {
      TraceRecord& record = g_trace.records[g_trace.record_count];
      record.operation = static_cast<uint32_t>(operation);
      record.reserved = 0;
      record.stream = reinterpret_cast<uint64_t>(stream);
      record.object = reinterpret_cast<uint64_t>(object);
    } else {
      ++g_trace.record_overflow;
    }
    ++g_trace.record_count;
    if (operation == TraceOperation::RecordEvent &&
        reinterpret_cast<uint64_t>(stream) == g_trace.expected_caller_stream) {
      g_saw_caller_start_record = true;
    } else if (operation == TraceOperation::LaunchAicpu) {
      ++g_trace.aicpu_launch_calls;
      if (reinterpret_cast<uint64_t>(stream) != g_trace.expected_caller_stream) {
        ++g_trace.private_aicpu_stream_calls;
      }
      if (!g_saw_caller_start_record) ++g_trace.early_aicpu_launch_calls;
    } else if (operation == TraceOperation::LaunchAicore) {
      ++g_trace.aicore_launch_calls;
      if (reinterpret_cast<uint64_t>(stream) == g_trace.expected_caller_stream) {
        ++g_trace.caller_stream_aicore_calls;
      }
    }
  }
  pthread_mutex_unlock(&g_trace_mutex);
}

enum class ForbiddenKind : uint8_t {
  StreamSync,
  DeviceSync,
  CaptureApi,
  ModelAttach,
  ResourceLifecycle,
  DeviceAllocation,
};

void count_forbidden(ForbiddenKind kind, void* return_address) noexcept {
  if (!called_from_host_runtime(return_address)) return;
  pthread_mutex_lock(&g_trace_mutex);
  if (g_trace_enabled.load(std::memory_order_relaxed)) {
    switch (kind) {
      case ForbiddenKind::StreamSync:
        ++g_trace.stream_sync_calls;
        break;
      case ForbiddenKind::DeviceSync:
        ++g_trace.device_sync_calls;
        break;
      case ForbiddenKind::CaptureApi:
        ++g_trace.capture_api_calls;
        break;
      case ForbiddenKind::ModelAttach:
        ++g_trace.model_attach_calls;
        break;
      case ForbiddenKind::ResourceLifecycle:
        ++g_trace.resource_lifecycle_calls;
        break;
      case ForbiddenKind::DeviceAllocation:
        ++g_trace.device_allocation_calls;
        break;
    }
  }
  pthread_mutex_unlock(&g_trace_mutex);
}

#define PYPTO_RETURN_ADDRESS __builtin_extract_return_addr(__builtin_return_address(0))

}  // namespace

extern "C" void pypto_l1_cann_trace_begin(uint64_t expected_caller_stream) noexcept {
  pthread_mutex_lock(&g_trace_mutex);
  std::memset(&g_trace, 0, sizeof(g_trace));
  g_trace.abi_version = kTraceAbiVersion;
  g_trace.expected_caller_stream = expected_caller_stream;
  g_saw_caller_start_record = false;
  g_trace_enabled.store(true, std::memory_order_release);
  pthread_mutex_unlock(&g_trace_mutex);
}

extern "C" int pypto_l1_cann_trace_end(TraceSnapshot* snapshot, size_t snapshot_size) noexcept {
  if (snapshot == nullptr || snapshot_size != sizeof(TraceSnapshot)) return -1;
  pthread_mutex_lock(&g_trace_mutex);
  g_trace_enabled.store(false, std::memory_order_release);
  std::memcpy(snapshot, &g_trace, sizeof(g_trace));
  pthread_mutex_unlock(&g_trace_mutex);
  return 0;
}

extern "C" int aclrtMemsetAsync(void* device_ptr, size_t max_count, int32_t value, size_t count,
                                void* stream) {
  append_record(TraceOperation::Memset, stream, device_ptr, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(void*, size_t, int32_t, size_t, void*);
  const auto function = next_symbol<Function>("aclrtMemsetAsync");
  return function == nullptr ? -1 : function(device_ptr, max_count, value, count, stream);
}

extern "C" int aclrtRecordEvent(void* event, void* stream) {
  append_record(TraceOperation::RecordEvent, stream, event, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(void*, void*);
  const auto function = next_symbol<Function>("aclrtRecordEvent");
  return function == nullptr ? -1 : function(event, stream);
}

extern "C" int aclrtStreamWaitEvent(void* stream, void* event) {
  append_record(TraceOperation::WaitEvent, stream, event, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(void*, void*);
  const auto function = next_symbol<Function>("aclrtStreamWaitEvent");
  return function == nullptr ? -1 : function(stream, event);
}

extern "C" int aclrtQueryEventStatus(void* event, void* status) {
  append_record(TraceOperation::QueryEvent, nullptr, event, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(void*, void*);
  const auto function = next_symbol<Function>("aclrtQueryEventStatus");
  return function == nullptr ? -1 : function(event, status);
}

extern "C" int aclrtLaunchKernelWithHostArgs(void* function_handle, uint32_t block_count, void* stream,
                                             void* config, void* host_args, size_t args_size,
                                             void* placeholders, size_t placeholder_count) {
  append_record(TraceOperation::LaunchAicpu, stream, function_handle, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(void*, uint32_t, void*, void*, void*, size_t, void*, size_t);
  const auto function = next_symbol<Function>("aclrtLaunchKernelWithHostArgs");
  return function == nullptr ? -1
                             : function(function_handle, block_count, stream, config, host_args, args_size,
                                        placeholders, placeholder_count);
}

extern "C" int rtKernelLaunchWithHandleV2(void* handle, uint64_t tiling_key, uint32_t block_count, void* args,
                                          void* shared_memory, void* stream, const void* config) {
  append_record(TraceOperation::LaunchAicore, stream, handle, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(void*, uint64_t, uint32_t, void*, void*, void*, const void*);
  const auto function = next_symbol<Function>("rtKernelLaunchWithHandleV2");
  return function == nullptr ? -1
                             : function(handle, tiling_key, block_count, args, shared_memory, stream, config);
}

#define PYPTO_WRAP_ONE_ARG(name, kind)                          \
  extern "C" int name(void* argument) {                         \
    count_forbidden(ForbiddenKind::kind, PYPTO_RETURN_ADDRESS); \
    using Function = int (*)(void*);                            \
    const auto function = next_symbol<Function>(#name);         \
    return function == nullptr ? -1 : function(argument);       \
  }

#define PYPTO_WRAP_NO_ARG(name, kind)                           \
  extern "C" int name() {                                       \
    count_forbidden(ForbiddenKind::kind, PYPTO_RETURN_ADDRESS); \
    using Function = int (*)();                                 \
    const auto function = next_symbol<Function>(#name);         \
    return function == nullptr ? -1 : function();               \
  }

#define PYPTO_WRAP_POINTER_INT(name, kind)                       \
  extern "C" int name(void* argument, int32_t value) {           \
    count_forbidden(ForbiddenKind::kind, PYPTO_RETURN_ADDRESS);  \
    using Function = int (*)(void*, int32_t);                    \
    const auto function = next_symbol<Function>(#name);          \
    return function == nullptr ? -1 : function(argument, value); \
  }

PYPTO_WRAP_ONE_ARG(aclrtSynchronizeStream, StreamSync)
PYPTO_WRAP_POINTER_INT(aclrtSynchronizeStreamWithTimeout, StreamSync)
PYPTO_WRAP_NO_ARG(aclrtSynchronizeDevice, DeviceSync)

extern "C" int aclrtSynchronizeDeviceWithTimeout(int32_t timeout) {
  count_forbidden(ForbiddenKind::DeviceSync, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(int32_t);
  const auto function = next_symbol<Function>("aclrtSynchronizeDeviceWithTimeout");
  return function == nullptr ? -1 : function(timeout);
}

PYPTO_WRAP_ONE_ARG(rtStreamSynchronize, StreamSync)
PYPTO_WRAP_POINTER_INT(rtStreamSynchronizeWithTimeout, StreamSync)
PYPTO_WRAP_NO_ARG(rtDeviceSynchronize, DeviceSync)

extern "C" int rtDeviceSynchronizeWithTimeout(int32_t timeout) {
  count_forbidden(ForbiddenKind::DeviceSync, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(int32_t);
  const auto function = next_symbol<Function>("rtDeviceSynchronizeWithTimeout");
  return function == nullptr ? -1 : function(timeout);
}

extern "C" int rtsStreamSynchronize(void* stream, int32_t timeout) {
  count_forbidden(ForbiddenKind::StreamSync, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(void*, int32_t);
  const auto function = next_symbol<Function>("rtsStreamSynchronize");
  return function == nullptr ? -1 : function(stream, timeout);
}

extern "C" int rtsDeviceSynchronize(int32_t timeout) {
  count_forbidden(ForbiddenKind::DeviceSync, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(int32_t);
  const auto function = next_symbol<Function>("rtsDeviceSynchronize");
  return function == nullptr ? -1 : function(timeout);
}

extern "C" int aclmdlRICaptureBegin(void* stream, int32_t mode) {
  count_forbidden(ForbiddenKind::CaptureApi, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(void*, int32_t);
  const auto function = next_symbol<Function>("aclmdlRICaptureBegin");
  return function == nullptr ? -1 : function(stream, mode);
}

extern "C" int aclmdlRICaptureGetInfo(void* stream, void* status, void* model) {
  count_forbidden(ForbiddenKind::CaptureApi, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(void*, void*, void*);
  const auto function = next_symbol<Function>("aclmdlRICaptureGetInfo");
  return function == nullptr ? -1 : function(stream, status, model);
}

extern "C" int aclmdlRICaptureEnd(void* stream, void* model) {
  count_forbidden(ForbiddenKind::CaptureApi, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(void*, void*);
  const auto function = next_symbol<Function>("aclmdlRICaptureEnd");
  return function == nullptr ? -1 : function(stream, model);
}

extern "C" int rtStreamBeginCapture(void* stream, int32_t mode) {
  count_forbidden(ForbiddenKind::CaptureApi, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(void*, int32_t);
  const auto function = next_symbol<Function>("rtStreamBeginCapture");
  return function == nullptr ? -1 : function(stream, mode);
}

extern "C" int rtStreamEndCapture(void* stream, void* model) {
  count_forbidden(ForbiddenKind::CaptureApi, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(void*, void*);
  const auto function = next_symbol<Function>("rtStreamEndCapture");
  return function == nullptr ? -1 : function(stream, model);
}

extern "C" int rtStreamGetCaptureInfo(void* stream, void* status, void* model) {
  count_forbidden(ForbiddenKind::CaptureApi, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(void*, void*, void*);
  const auto function = next_symbol<Function>("rtStreamGetCaptureInfo");
  return function == nullptr ? -1 : function(stream, status, model);
}

extern "C" int rtStreamAddToModel(void* stream, void* model) {
  count_forbidden(ForbiddenKind::ModelAttach, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(void*, void*);
  const auto function = next_symbol<Function>("rtStreamAddToModel");
  return function == nullptr ? -1 : function(stream, model);
}

extern "C" int rtModelBindStream(void* model, void* stream, uint32_t flag) {
  count_forbidden(ForbiddenKind::ModelAttach, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(void*, void*, uint32_t);
  const auto function = next_symbol<Function>("rtModelBindStream");
  return function == nullptr ? -1 : function(model, stream, flag);
}

PYPTO_WRAP_ONE_ARG(aclrtCreateStream, ResourceLifecycle)
PYPTO_WRAP_ONE_ARG(aclrtDestroyStream, ResourceLifecycle)
PYPTO_WRAP_ONE_ARG(aclrtCreateEvent, ResourceLifecycle)
PYPTO_WRAP_ONE_ARG(aclrtDestroyEvent, ResourceLifecycle)

extern "C" int aclrtCreateStreamWithConfig(void* stream, uint32_t priority, uint32_t flag) {
  count_forbidden(ForbiddenKind::ResourceLifecycle, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(void*, uint32_t, uint32_t);
  const auto function = next_symbol<Function>("aclrtCreateStreamWithConfig");
  return function == nullptr ? -1 : function(stream, priority, flag);
}

extern "C" int aclrtCreateEventExWithFlag(void* event, uint32_t flag) {
  count_forbidden(ForbiddenKind::ResourceLifecycle, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(void*, uint32_t);
  const auto function = next_symbol<Function>("aclrtCreateEventExWithFlag");
  return function == nullptr ? -1 : function(event, flag);
}

extern "C" int aclrtMalloc(void* device_ptr, size_t size, int32_t policy) {
  count_forbidden(ForbiddenKind::DeviceAllocation, PYPTO_RETURN_ADDRESS);
  using Function = int (*)(void*, size_t, int32_t);
  const auto function = next_symbol<Function>("aclrtMalloc");
  return function == nullptr ? -1 : function(device_ptr, size, policy);
}

PYPTO_WRAP_ONE_ARG(aclrtFree, DeviceAllocation)

#undef PYPTO_WRAP_POINTER_INT
#undef PYPTO_WRAP_NO_ARG
#undef PYPTO_WRAP_ONE_ARG
#undef PYPTO_RETURN_ADDRESS
