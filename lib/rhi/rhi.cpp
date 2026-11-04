#include "rhi.hpp"

namespace aurora::rhi {

std::unique_ptr<Device> create_d3d12_device(std::string* error);
std::unique_ptr<Device> create_vulkan_device(std::string* error);

std::unique_ptr<Device> create_device(Backend backend, std::string* error) {
  switch (backend) {
  case Backend::D3D12:
#ifdef _WIN32
    return create_d3d12_device(error);
#else
    *error = "D3D12 exists only on Windows";
    return nullptr;
#endif
  case Backend::Vulkan:
    return create_vulkan_device(error);
  }
  return nullptr;
}

} // namespace aurora::rhi
