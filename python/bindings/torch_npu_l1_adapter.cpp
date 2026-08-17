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

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/worker/l1_queue_call.h"

#ifndef PYPTO_L1_BUILD_TORCH_VERSION
#define PYPTO_L1_BUILD_TORCH_VERSION "unknown"
#endif

#ifndef PYPTO_L1_BUILD_TORCH_NPU_VERSION
#define PYPTO_L1_BUILD_TORCH_NPU_VERSION "unknown"
#endif

namespace nb = nanobind;

namespace {

class DeferredQueueCall {
 public:
  DeferredQueueCall(const SimplerL1QueueCall& call, std::vector<at::Tensor> tensors)
      : call_(call), tensors_(std::move(tensors)) {
    call_.retain(call_.opaque);
  }
  ~DeferredQueueCall() { call_.release(call_.opaque); }

  DeferredQueueCall(const DeferredQueueCall&) = delete;
  DeferredQueueCall& operator=(const DeferredQueueCall&) = delete;

  int invoke(uint64_t raw_stream) const noexcept { return call_.invoke(call_.opaque, raw_stream); }

 private:
  SimplerL1QueueCall call_;
  // Keep every tensor/storage alive until taskQueue has actually invoked the
  // native callback and enqueued the complete L1 sequence on the ACL stream.
  std::vector<at::Tensor> tensors_;
};

const SimplerL1QueueCall& validated_queue_call(nb::handle capsule) {
  if (!PyCapsule_CheckExact(capsule.ptr()) ||
      PyCapsule_IsValid(capsule.ptr(), SIMPLER_L1_QUEUE_CALL_CAPSULE_NAME) == 0) {
    throw nb::type_error("expected a simpler.l1.queue_call.v1 capsule");
  }
  auto* call = static_cast<SimplerL1QueueCall*>(
      PyCapsule_GetPointer(capsule.ptr(), SIMPLER_L1_QUEUE_CALL_CAPSULE_NAME));
  if (call == nullptr) throw nb::python_error();
  if (call->abi_version != SIMPLER_L1_QUEUE_CALL_ABI_VERSION ||
      call->struct_size < sizeof(SimplerL1QueueCall)) {
    throw nb::value_error("incompatible simpler L1 queue-call ABI");
  }
  if (call->opaque == nullptr || call->retain == nullptr || call->release == nullptr ||
      call->invoke == nullptr) {
    throw nb::value_error("incomplete simpler L1 queue-call descriptor");
  }
  return *call;
}

std::vector<at::Tensor> collect_tensors(nb::iterable tensors, int expected_device) {
  std::vector<at::Tensor> result;
  for (nb::handle value : tensors) {
    if (!THPVariable_Check(value.ptr()))
      throw nb::type_error("tensors must contain only torch.Tensor objects");
    at::Tensor tensor = THPVariable_Unpack(value.ptr());
    if (!tensor.defined() || tensor.device().type() != c10::DeviceType::PrivateUse1 ||
        tensor.get_device() != expected_device) {
      throw nb::value_error("every L1 tensor must be an NPU tensor on expected_device");
    }
    result.push_back(std::move(tensor));
  }
  return result;
}

void enqueue(nb::handle capsule, nb::iterable tensors, int expected_device, const std::string& op_name) {
  const SimplerL1QueueCall& call = validated_queue_call(capsule);
  if (expected_device < 0) throw nb::value_error("expected_device must be non-negative");
  if (op_name.empty()) throw nb::value_error("op_name must be non-empty");
  std::vector<at::Tensor> tensor_keepalive = collect_tensors(tensors, expected_device);

  const c10_npu::NPUStream current_stream = c10_npu::getCurrentNPUStream();
  if (static_cast<int>(current_stream.device_index()) != expected_device) {
    throw nb::value_error("current torch_npu stream belongs to a different device than the PyPTO L1 context");
  }

  // stream(false) returns the raw ACL stream without draining taskQueue.
  // RunOpApiV2 then owns ordering in both queue-enabled and direct modes.
  const aclrtStream raw_stream = current_stream.stream(false);
  if (raw_stream == nullptr) throw std::runtime_error("torch_npu returned a null current ACL stream");

  // Custom raw-pointer launches are invisible to the caching allocator.
  // Record every distinct storage on the exact current stream so an early
  // Python reference drop cannot recycle it while the L1 op is still live.
  std::unordered_set<void*> recorded_storages;
  for (const at::Tensor& tensor : tensor_keepalive) {
    const c10::DataPtr& data_ptr = tensor.storage().data_ptr();
    void* const storage = data_ptr.get();
    if (storage != nullptr && recorded_storages.insert(storage).second) {
      c10_npu::NPUCachingAllocator::recordStream(data_ptr, current_stream);
    }
  }

  auto deferred = std::make_shared<DeferredQueueCall>(call, std::move(tensor_keepalive));
  const uint64_t raw_stream_value = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(raw_stream));
  std::function<int()> callback = [deferred = std::move(deferred), raw_stream_value]() noexcept {
    return deferred->invoke(raw_stream_value);
  };
  at_npu::native::OpCommand::RunOpApiV2(op_name, callback, false);
}

}  // namespace

NB_MODULE(_torch_npu_l1, m) {
  m.doc() = "Internal taskQueue-aware torch_npu adapter for PyPTO L1";
  m.attr("QUEUE_CALL_ABI_VERSION") = SIMPLER_L1_QUEUE_CALL_ABI_VERSION;
  m.attr("BUILD_TORCH_VERSION") = PYPTO_L1_BUILD_TORCH_VERSION;
  m.attr("BUILD_TORCH_NPU_VERSION") = PYPTO_L1_BUILD_TORCH_NPU_VERSION;
  m.def("enqueue", &enqueue, nb::arg("queue_call"), nb::arg("tensors"), nb::arg("expected_device"),
        nb::arg("op_name"), "Enqueue one retained simpler L1 deferred call through torch_npu taskQueue.");
}
