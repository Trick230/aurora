// Runs the same work on every backend and compares the pixels, headless: both
// must give the expected bytes, and the same ones.
#include "rhi.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace aurora::rhi;

namespace {

constexpr uint32_t Width = 64, Height = 64;
constexpr uint32_t BytesPerRow = CopyRowAlignment;

bool render_clear(Device& device, std::vector<uint8_t>& pixels) {
  auto texture = device.create_texture({Width, Height, Format::RGBA8Unorm});
  auto buffer = device.create_buffer({uint64_t{BytesPerRow} * Height, MemoryKind::Readback});
  auto list = device.create_command_list();
  if (!texture || !buffer || !list || !buffer->mapped())
    return false;
  list->begin();
  list->begin_render_pass(*texture, {0.25f, 0.5f, 0.75f, 1.0f});
  list->end_render_pass();
  list->copy_texture_to_buffer(*texture, *buffer, BytesPerRow);
  list->end();
  device.submit(*list);
  device.wait_idle();
  pixels.resize(size_t{Width} * 4 * Height);
  const auto* mapped = static_cast<const uint8_t*>(buffer->mapped());
  for (uint32_t y = 0; y < Height; ++y)
    std::memcpy(pixels.data() + size_t{y} * Width * 4, mapped + size_t{y} * BytesPerRow, Width * 4);
  return true;
}

} // namespace

int main() {
  int failures = 0;
  std::vector<uint8_t> results[2];
  const struct {
    Backend backend;
    const char* name;
  } backends[] = {{Backend::D3D12, "D3D12"}, {Backend::Vulkan, "Vulkan"}};
  for (int i = 0; i < 2; ++i) {
    std::string error;
    auto device = create_device(backends[i].backend, &error);
    if (!device) {
      std::printf("%s: no device: %s\n", backends[i].name, error.c_str());
      ++failures;
      continue;
    }
    std::printf("%s: %s\n", backends[i].name, device->adapter_name().c_str());
    if (!render_clear(*device, results[i])) {
      std::printf("%s: could not render\n", backends[i].name);
      ++failures;
      continue;
    }
    const uint8_t* p = results[i].data();
    std::printf("%s: first pixel %u %u %u %u\n", backends[i].name, p[0], p[1], p[2], p[3]);
    for (size_t at = 0; at < results[i].size(); at += 4) {
      // 0.25, 0.5 and 0.75 of 255 fall close to a half; a driver may round them either way.
      if (p[at] < 63 || p[at] > 64 || p[at + 1] < 127 || p[at + 1] > 128 || p[at + 2] < 191 || p[at + 2] > 192 ||
          p[at + 3] != 255) {
        std::printf("%s: wrong pixel at byte %zu\n", backends[i].name, at);
        ++failures;
        break;
      }
    }
  }
  if (!results[0].empty() && results[0] == results[1])
    std::printf("backends match\n");
  else {
    std::printf("backends differ\n");
    ++failures;
  }
  return failures ? 1 : 0;
}
