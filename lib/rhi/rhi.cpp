#include "rhi.hpp"

#include "rhi_internal.hpp"

namespace aurora::rhi {

std::atomic<int64_t> g_validation_errors{-1};

void validation_active() {
  int64_t expected = -1;
  g_validation_errors.compare_exchange_strong(expected, 0);
}

int64_t validation_errors() { return g_validation_errors.load(); }

std::unique_ptr<Device> create_device(Backend backend, const DeviceOptions& options, std::string* error) {
  switch (backend) {
  case Backend::D3D12:
#ifdef _WIN32
    return create_d3d12_device(options, error);
#else
    *error = "D3D12 exists only on Windows";
    return nullptr;
#endif
  case Backend::Vulkan:
    return create_vulkan_device(options, error);
  }
  return nullptr;
}

} // namespace aurora::rhi
