// Runs the same work on every backend and compares the pixels, headless: both
// must give the expected bytes, and the same ones.
#include "rhi.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

const char* const TriangleShader = R"(
struct Varyings {
  float4 position : SV_Position;
  float3 color : COLOR0;
};
Varyings vs_main(uint vertex : SV_VertexID) {
  static const float2 positions[3] = {float2(-0.8, -0.8), float2(0.8, -0.8), float2(0.0, 0.8)};
  static const float3 colors[3] = {float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1)};
  Varyings result;
  result.position = float4(positions[vertex], 0.5, 1.0);
  result.color = colors[vertex];
  return result;
}
float4 fs_main(Varyings input) : SV_Target {
  return float4(input.color, 1.0);
}
)";

bool render_triangle(Device& device, std::vector<uint8_t>& pixels) {
  ShaderCode vertex, fragment;
  std::string error;
  if (!compile_shader(device.backend(), ShaderStage::Vertex, TriangleShader, "vs_main", &vertex, &error) ||
      !compile_shader(device.backend(), ShaderStage::Fragment, TriangleShader, "fs_main", &fragment, &error)) {
    std::printf("shader: %s\n", error.c_str());
    return false;
  }
  RenderPipelineDesc desc;
  desc.vertex = &vertex;
  desc.vertex_entry = "vs_main";
  desc.fragment = &fragment;
  desc.fragment_entry = "fs_main";
  auto pipeline = device.create_render_pipeline(desc);
  auto texture = device.create_texture({Width, Height, Format::RGBA8Unorm});
  auto buffer = device.create_buffer({uint64_t{BytesPerRow} * Height, MemoryKind::Readback});
  auto list = device.create_command_list();
  if (!pipeline || !texture || !buffer || !list)
    return false;
  list->begin();
  list->begin_render_pass(*texture, {0.0f, 0.0f, 0.0f, 1.0f});
  list->set_pipeline(*pipeline);
  list->draw(3);
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
  std::vector<uint8_t> results[2], triangles[2];
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
    if (!render_triangle(*device, triangles[i])) {
      std::printf("%s: could not draw\n", backends[i].name);
      ++failures;
      continue;
    }
    // The apex points up on both: the bottom row is wide, the top one empty.
    const auto pixel = [&](uint32_t x, uint32_t y) { return &triangles[i][(size_t{y} * Width + x) * 4]; };
    if (pixel(32, 56)[0] + pixel(32, 56)[1] == 0 || pixel(32, 2)[0] + pixel(32, 2)[1] + pixel(32, 2)[2] != 0) {
      std::printf("%s: the triangle is upside down\n", backends[i].name);
      ++failures;
    }
  }
  if (!triangles[0].empty() && triangles[0].size() == triangles[1].size()) {
    size_t different = 0;
    int worst = 0;
    for (size_t at = 0; at < triangles[0].size(); ++at) {
      const int delta = std::abs(triangles[0][at] - triangles[1][at]);
      different += delta != 0;
      worst = std::max(worst, delta);
    }
    std::printf("triangle: %zu bytes differ, worst by %d\n", different, worst);
    if (worst > 1)
      ++failures;
  } else {
    ++failures;
  }
  if (!results[0].empty() && results[0] == results[1])
    std::printf("backends match\n");
  else {
    std::printf("backends differ\n");
    ++failures;
  }
  return failures ? 1 : 0;
}
