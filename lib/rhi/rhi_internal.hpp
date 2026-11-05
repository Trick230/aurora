#pragma once

#include "rhi.hpp"

#include <atomic>

namespace aurora::rhi {

std::unique_ptr<Device> create_d3d12_device(const DeviceOptions& options, std::string* error);
std::unique_ptr<Device> create_vulkan_device(const DeviceOptions& options, std::string* error);

extern std::atomic<int64_t> g_validation_errors;
void validation_active();

inline constexpr uint32_t MipExtent(uint32_t size, uint32_t mip) { return size >> mip ? size >> mip : 1; }

} // namespace aurora::rhi
