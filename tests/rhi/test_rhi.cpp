// Runs the same work on every backend and compares the pixels, headless: the same
// bytes on both, or close to it where the APIs leave rounding to the driver.
#include "rhi.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace aurora::rhi;

namespace {

constexpr uint32_t Width = 128, Height = 128;
constexpr uint32_t BytesPerRow = CopyRowAlignment * 3; // 128 * 4 rounded up, with room to spare
constexpr uint32_t UniformStride = 256;

using Pixels = std::vector<uint8_t>;

template <typename T>
std::unique_ptr<T> must(std::unique_ptr<T> object, const char* what) {
  if (!object) {
    std::printf("could not create %s\n", what);
    std::exit(1);
  }
  return object;
}

// Everything the renderer needs in one frame: vertices pulled from a storage
// buffer, indices, uniforms at dynamic offsets, mipped and compressed textures,
// swizzles, samplers, depth, blending with a constant, dual source blending,
// alpha to coverage, multisampling with a resolve, a multisampled depth read, and
// copies between textures and buffers.
const char* const SceneShader = R"(
ByteAddressBuffer vertices : register(t0, space0);

cbuffer Uniforms : register(b0, space1) {
  float2 offset;
  float scale;
  float depth;
  float4 tint;
};

Texture2D color_map : register(t0, space2);
SamplerState color_sampler : register(s1, space2);
Texture2D compressed_map : register(t2, space2);
SamplerState compressed_sampler : register(s3, space2);

struct Varyings {
  float4 position : SV_Position;
  float2 uv : TEXCOORD0;
};

Varyings vs_main(uint vertex : SV_VertexID) {
  const float2 position = asfloat(vertices.Load2(vertex * 16));
  Varyings result;
  result.position = float4(position * scale + offset, depth, 1.0);
  result.uv = asfloat(vertices.Load2(vertex * 16 + 8));
  return result;
}

float4 fs_textured(Varyings input) : SV_Target {
  return color_map.Sample(color_sampler, input.uv) * tint +
         compressed_map.Sample(compressed_sampler, input.uv) * 0.25;
}

struct DualOutput {
  [[vk::location(0)]] [[vk::index(0)]] float4 color : SV_Target0;
  [[vk::location(0)]] [[vk::index(1)]] float4 factor : SV_Target1;
};

DualOutput fs_dual(Varyings input) {
  DualOutput output;
  output.color = tint;
  output.factor = float4(0.0, 0.0, 0.0, input.uv.x);
  return output;
}

float4 fs_cutout(Varyings input) : SV_Target {
  return float4(tint.rgb, input.uv.x);
}
)";

// Reads sample 0 of a multisampled depth texture, as the EFB's depth copies do.
const char* const DepthReadShader = R"(
Texture2DMS<float> depth_map : register(t0, space0);
float4 vs_main(uint vertex : SV_VertexID) : SV_Position {
  return float4(float2((vertex << 1) & 2, vertex & 2) * 2.0 - 1.0, 0.0, 1.0);
}
float4 fs_main(float4 position : SV_Position) : SV_Target {
  const float depth = depth_map.Load(int2(position.xy), 0);
  return float4(depth, depth, depth, 1.0);
}
)";

ShaderCode compile(Device& device, ShaderStage stage, const char* source, const char* entry) {
  ShaderCode code;
  std::string error;
  if (!compile_shader(device.backend(), stage, source, entry, &code, &error)) {
    std::printf("shader %s: %s\n", entry, error.c_str());
    std::exit(1);
  }
  return code;
}

struct Scene {
  std::unique_ptr<Buffer> upload, storage, uniforms, indices, readback;
  std::unique_ptr<Texture> color_map, compressed_map;
  std::unique_ptr<TextureView> color_view, compressed_view;
  std::unique_ptr<Sampler> linear, aniso;
};

Pixels read_pixels(Buffer& readback, uint64_t offset) {
  Pixels pixels(size_t{Width} * 4 * Height);
  const auto* mapped = static_cast<const uint8_t*>(readback.mapped()) + offset;
  for (uint32_t y = 0; y < Height; ++y)
    std::memcpy(pixels.data() + size_t{y} * Width * 4, mapped + size_t{y} * BytesPerRow, Width * 4);
  return pixels;
}

// Three results: the scene with 4 samples, with 8, and the depth read back.
std::vector<Pixels> render(Device& device) {
  std::vector<Pixels> results;
  Scene scene;

  static const float vertices[] = {
      -1.0f, -1.0f, 0.0f, 2.0f, 1.0f, -1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 0.0f, -1.0f, 1.0f, 0.0f, 0.0f,
  };
  static const uint16_t indices[] = {0, 1, 2, 0, 2, 3};
  struct Uniform {
    float offset[2];
    float scale;
    float depth;
    float tint[4];
    uint8_t pad[UniformStride - 32];
  };
  static const Uniform uniforms[] = {
      {{0.0f, 0.0f}, 1.0f, 0.75f, {1.0f, 1.0f, 1.0f, 1.0f}, {}},
      {{0.2f, 0.1f}, 0.5f, 0.25f, {0.8f, 0.3f, 0.2f, 0.5f}, {}},
      {{-0.45f, -0.4f}, 0.4f, 0.5f, {0.1f, 0.7f, 0.3f, 1.0f}, {}}, // behind the second, in front of the first
      {{-0.3f, 0.45f}, 0.35f, 0.2f, {0.2f, 0.4f, 0.9f, 1.0f}, {}},
      {{0.5f, -0.5f}, 0.35f, 0.2f, {0.9f, 0.8f, 0.1f, 1.0f}, {}},
  };

  scene.upload = must(device.create_buffer({1u << 20, MemoryKind::Upload}), "upload buffer");
  scene.storage = must(device.create_buffer({sizeof(vertices)}), "storage buffer");
  scene.uniforms = must(device.create_buffer({sizeof(uniforms)}), "uniform buffer");
  scene.indices = must(device.create_buffer({sizeof(indices)}), "index buffer");
  scene.readback = must(device.create_buffer({uint64_t{BytesPerRow} * Height * 3, MemoryKind::Readback}), "readback");

  auto* upload = static_cast<uint8_t*>(scene.upload->mapped());
  std::memcpy(upload, vertices, sizeof(vertices));
  std::memcpy(upload + 1024, uniforms, sizeof(uniforms));
  std::memcpy(upload + 4096, indices, sizeof(indices));

  // A checkerboard with a second mip of another color, in R and G only: the view's
  // swizzle fills the rest.
  constexpr uint64_t Mip0 = 8192, Mip1 = 16384, Block = 32768;
  for (uint32_t y = 0; y < 16; ++y) {
    for (uint32_t x = 0; x < 16; ++x) {
      uint8_t* pixel = upload + Mip0 + y * CopyRowAlignment + x * 2;
      pixel[0] = ((x ^ y) & 1) ? 255 : 32;
      pixel[1] = uint8_t(x * 16);
    }
  }
  for (uint32_t y = 0; y < 8; ++y) {
    for (uint32_t x = 0; x < 8; ++x) {
      uint8_t* pixel = upload + Mip1 + y * CopyRowAlignment + x * 2;
      pixel[0] = 128;
      pixel[1] = 200;
    }
  }
  // A BC1 texture of 8x8 in four blocks of two colors.
  for (uint32_t y = 0; y < 2; ++y) {
    for (uint32_t x = 0; x < 2; ++x) {
      static const uint8_t blocks[4][8] = {{0x00, 0xF8, 0x00, 0x00, 0x55, 0x55, 0x55, 0x55},
                                           {0xE0, 0x07, 0x00, 0x00, 0xAA, 0xAA, 0xAA, 0xAA},
                                           {0x1F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
                                           {0xFF, 0xFF, 0x00, 0x00, 0xE4, 0xE4, 0xE4, 0xE4}};
      std::memcpy(upload + Block + y * CopyRowAlignment + x * 8, blocks[y * 2 + x], 8);
    }
  }

  scene.color_map = must(device.create_texture({16, 16, 2, 1, Format::RG8Unorm, TextureSampled}), "color map");
  scene.compressed_map = must(device.create_texture({8, 8, 1, 1, Format::BC1, TextureSampled}), "compressed map");
  TextureViewDesc swizzle;
  swizzle.b = Swizzle::R;
  swizzle.a = Swizzle::One;
  scene.color_view = must(scene.color_map->create_view(swizzle), "color view");
  scene.compressed_view = must(scene.compressed_map->create_view({}), "compressed view");
  SamplerDesc linear;
  scene.linear = must(device.create_sampler(linear), "sampler");
  SamplerDesc aniso;
  aniso.anisotropy = 16;
  aniso.u = AddressMode::MirrorRepeat;
  aniso.v = AddressMode::ClampToEdge;
  scene.aniso = must(device.create_sampler(aniso), "anisotropic sampler");

  const BindingLayout static_entries[] = {{0, BindingKind::StorageRead}};
  BindingLayout uniform_entry{0, BindingKind::Uniform};
  uniform_entry.dynamic_offset = true;
  const BindingLayout texture_entries[] = {{0, BindingKind::Texture},
                                           {1, BindingKind::Sampler},
                                           {2, BindingKind::Texture},
                                           {3, BindingKind::Sampler}};
  auto static_layout = must(device.create_bind_group_layout({static_entries, 1}), "static layout");
  auto uniform_layout = must(device.create_bind_group_layout({&uniform_entry, 1}), "uniform layout");
  auto texture_layout = must(device.create_bind_group_layout({texture_entries, 4}), "texture layout");
  BindGroupLayout* groups[] = {static_layout.get(), uniform_layout.get(), texture_layout.get()};
  auto layout = must(device.create_pipeline_layout({groups, 3}), "pipeline layout");

  BindingValue static_value{};
  static_value.buffer = scene.storage.get();
  auto static_group = must(device.create_bind_group({static_layout.get(), &static_value, 1}), "static group");
  BindingValue uniform_value{};
  uniform_value.buffer = scene.uniforms.get();
  uniform_value.size = 32;
  auto uniform_group = must(device.create_bind_group({uniform_layout.get(), &uniform_value, 1}), "uniform group");
  BindingValue texture_values[4]{};
  texture_values[0].binding = 0;
  texture_values[0].view = scene.color_view.get();
  texture_values[1].binding = 1;
  texture_values[1].sampler = scene.aniso.get();
  texture_values[2].binding = 2;
  texture_values[2].view = scene.compressed_view.get();
  texture_values[3].binding = 3;
  texture_values[3].sampler = scene.linear.get();
  auto texture_group = must(device.create_bind_group({texture_layout.get(), texture_values, 4}), "texture group");

  const ShaderCode vertex = compile(device, ShaderStage::Vertex, SceneShader, "vs_main");
  const ShaderCode textured = compile(device, ShaderStage::Fragment, SceneShader, "fs_textured");
  const ShaderCode dual = compile(device, ShaderStage::Fragment, SceneShader, "fs_dual");
  const ShaderCode cutout = compile(device, ShaderStage::Fragment, SceneShader, "fs_cutout");

  auto list = must(device.create_command_list(), "command list");
  auto setup = must(device.create_command_list(), "setup list");
  setup->begin();
  setup->copy_buffer(*scene.upload, 0, *scene.storage, 0, sizeof(vertices));
  setup->copy_buffer(*scene.upload, 1024, *scene.uniforms, 0, sizeof(uniforms));
  setup->copy_buffer(*scene.upload, 4096, *scene.indices, 0, sizeof(indices));
  setup->copy_buffer_to_texture(*scene.upload, Mip0, CopyRowAlignment, *scene.color_map, {0});
  setup->copy_buffer_to_texture(*scene.upload, Mip1, CopyRowAlignment, *scene.color_map, {1});
  setup->copy_buffer_to_texture(*scene.upload, Block, CopyRowAlignment, *scene.compressed_map, {0});
  setup->end();
  device.submit(*setup);

  uint64_t readback_offset = 0;
  const auto render_scene = [&](uint32_t samples) {
    auto target = must(device.create_texture({Width, Height, 1, samples, Format::RGBA8Unorm, TextureColorTarget}), "target");
    auto resolve =
        must(device.create_texture({Width, Height, 1, 1, Format::RGBA8Unorm, TextureSampled | TextureColorTarget}),
             "resolve");
    auto depth = must(
        device.create_texture({Width, Height, 1, samples, Format::Depth32Float, TextureSampled | TextureDepthTarget}),
        "depth");

    const auto base_desc = [&] {
      RenderPipelineDesc desc;
      desc.layout = layout.get();
      desc.vertex = &vertex;
      desc.vertex_entry = "vs_main";
      desc.fragment_entry = "fs_textured";
      desc.fragment = &textured;
      desc.samples = samples;
      desc.depth_format = Format::Depth32Float;
      desc.depth_compare = Compare::Less;
      desc.depth_write = true;
      desc.cull = CullMode::Back;
      return desc;
    };
    auto opaque = must(device.create_render_pipeline(base_desc()), "opaque pipeline");
    RenderPipelineDesc blend_desc = base_desc();
    blend_desc.depth_write = false;
    blend_desc.blend = true;
    blend_desc.blend_color = {BlendFactor::SrcAlpha, BlendFactor::OneMinusSrcAlpha, BlendOp::Add};
    blend_desc.blend_alpha = {BlendFactor::Constant, BlendFactor::Zero, BlendOp::Add};
    auto blended = must(device.create_render_pipeline(blend_desc), "blended pipeline");
    RenderPipelineDesc dual_desc = base_desc();
    dual_desc.fragment = &dual;
    dual_desc.fragment_entry = "fs_dual";
    dual_desc.blend = true;
    dual_desc.blend_color = {BlendFactor::One, BlendFactor::OneMinusSrc1Alpha, BlendOp::Add};
    dual_desc.blend_alpha = {BlendFactor::One, BlendFactor::Zero, BlendOp::Add};
    auto dual_blend = must(device.create_render_pipeline(dual_desc), "dual source pipeline");
    RenderPipelineDesc cutout_desc = base_desc();
    cutout_desc.fragment = &cutout;
    cutout_desc.fragment_entry = "fs_cutout";
    cutout_desc.alpha_to_coverage = true;
    auto covered = must(device.create_render_pipeline(cutout_desc), "alpha to coverage pipeline");

    list->begin();
    RenderPassDesc pass;
    pass.color.target = target.get();
    pass.color.resolve = resolve.get();
    pass.color.clear = true;
    pass.color.clear_color = {0.05f, 0.1f, 0.15f, 1.0f};
    pass.depth.target = depth.get();
    pass.depth.clear = true;
    list->begin_render_pass(pass);
    list->set_pipeline(*opaque);
    list->set_bind_group(0, *static_group, nullptr, 0);
    list->set_bind_group(2, *texture_group, nullptr, 0);
    list->set_index_buffer_u16(*scene.indices, 0);
    list->set_blend_constant(0.5f, 0.5f, 0.5f, 0.5f);
    const auto draw = [&](uint32_t index) {
      const uint32_t offset = index * UniformStride;
      list->set_bind_group(1, *uniform_group, &offset, 1);
      list->draw_indexed(6, 0, 0);
    };
    draw(0);
    list->set_pipeline(*blended);
    list->set_bind_group(0, *static_group, nullptr, 0);
    list->set_bind_group(2, *texture_group, nullptr, 0);
    draw(1);
    list->set_pipeline(*opaque);
    list->set_scissor(10, 10, 100, 100);
    list->set_viewport(0.0f, 0.0f, 120.0f, 120.0f, 0.0f, 1.0f);
    draw(2);
    list->set_scissor(0, 0, Width, Height);
    list->set_viewport(0.0f, 0.0f, Width, Height, 0.0f, 1.0f);
    list->set_pipeline(*dual_blend);
    draw(3);
    list->set_pipeline(*covered);
    draw(4);
    list->end_render_pass();
    list->copy_texture_to_buffer(*resolve, {0}, *scene.readback, readback_offset, BytesPerRow);
    list->end();
    device.submit(*list);
    device.wait_idle();
    results.push_back(read_pixels(*scene.readback, readback_offset));
    readback_offset += uint64_t{BytesPerRow} * Height;
    if (samples != 4)
      return;

    // The depth of that pass, sample 0, through a shader, then through a copy.
    const BindingLayout depth_entry{0, BindingKind::Texture};
    auto depth_layout = must(device.create_bind_group_layout({&depth_entry, 1}), "depth layout");
    BindGroupLayout* depth_groups[] = {depth_layout.get()};
    auto depth_pipeline_layout = must(device.create_pipeline_layout({depth_groups, 1}), "depth pipeline layout");
    const ShaderCode depth_vertex = compile(device, ShaderStage::Vertex, DepthReadShader, "vs_main");
    const ShaderCode depth_fragment = compile(device, ShaderStage::Fragment, DepthReadShader, "fs_main");
    RenderPipelineDesc depth_desc;
    depth_desc.layout = depth_pipeline_layout.get();
    depth_desc.vertex = &depth_vertex;
    depth_desc.vertex_entry = "vs_main";
    depth_desc.fragment = &depth_fragment;
    depth_desc.fragment_entry = "fs_main";
    auto depth_read = must(device.create_render_pipeline(depth_desc), "depth read pipeline");
    auto view = must(depth->create_view({}), "depth view");
    BindingValue depth_value{};
    depth_value.view = view.get();
    auto depth_group = must(device.create_bind_group({depth_layout.get(), &depth_value, 1}), "depth group");
    auto output =
        must(device.create_texture({Width, Height, 1, 1, Format::RGBA8Unorm, TextureSampled | TextureColorTarget}), "output");
    auto copy =
        must(device.create_texture({Width, Height, 1, 1, Format::RGBA8Unorm, TextureSampled}), "copy");
    list->begin();
    RenderPassDesc read_pass;
    read_pass.color.target = output.get();
    read_pass.color.clear = true;
    list->begin_render_pass(read_pass);
    list->set_pipeline(*depth_read);
    list->set_bind_group(0, *depth_group, nullptr, 0);
    list->draw(3, 0);
    list->end_render_pass();
    list->copy_texture(*output, {0}, *copy, 0, 0);
    list->copy_texture_to_buffer(*copy, {0}, *scene.readback, readback_offset, BytesPerRow);
    list->end();
    device.submit(*list);
    device.wait_idle();
    results.push_back(read_pixels(*scene.readback, readback_offset));
    readback_offset += uint64_t{BytesPerRow} * Height;
  };
  render_scene(4);
  render_scene(8);
  return results;
}

} // namespace

int main() {
  int failures = 0;
  std::vector<Pixels> results[2];
  const struct {
    Backend backend;
    const char* name;
  } backends[] = {{Backend::D3D12, "D3D12"}, {Backend::Vulkan, "Vulkan"}};
  DeviceOptions options;
  options.validation = std::getenv("RHI_VALIDATION") != nullptr;
  for (int i = 0; i < 2; ++i) {
    std::string error;
    auto device = create_device(backends[i].backend, options, &error);
    if (!device) {
      std::printf("%s: no device: %s\n", backends[i].name, error.c_str());
      ++failures;
      continue;
    }
    std::printf("%s: %s, up to %u samples\n", backends[i].name, device->adapter_name().c_str(),
                device->features().max_samples);
    if (device->features().max_samples < 8) {
      std::printf("%s: 8 samples missing\n", backends[i].name);
      ++failures;
    }
    results[i] = render(*device);
    device->wait_idle();
  }
  if (results[0].size() != results[1].size() || results[0].empty()) {
    std::printf("no results to compare\n");
    return 1;
  }
  const char* const names[] = {"scene 4x", "depth read", "scene 8x"};
  for (size_t r = 0; r < results[0].size(); ++r) {
    const Pixels& a = results[0][r];
    const Pixels& b = results[1][r];
    size_t different = 0;
    int worst = 0;
    uint64_t sum = 0;
    // Which samples alpha to coverage picks for a given alpha is left to the
    // driver, and it differs between the APIs on the same GPU: that quad is held
    // to its mean only.
    int64_t quad_a = 0, quad_b = 0;
    uint32_t quad_bytes = 0;
    for (size_t at = 0; at < a.size(); ++at) {
      const uint32_t x = uint32_t(at / 4 % Width), y = uint32_t(at / 4 / Width);
      if (x >= 73 && x <= 119 && y >= 73 && y <= 119) {
        quad_a += a[at], quad_b += b[at], ++quad_bytes;
        continue;
      }
      const int delta = std::abs(a[at] - b[at]);
      different += delta != 0;
      worst = std::max(worst, delta);
      sum += a[at];
    }
    const double quad_delta = std::abs(double(quad_a - quad_b)) / quad_bytes;
    std::printf("%s: %zu bytes differ, worst by %d, mean %.1f; coverage quad's means differ by %.2f\n", names[r],
                different, worst, double(sum) / double(a.size()), quad_delta);
    // Nothing drawn at all would match too.
    if (worst > 1 || sum / a.size() < 8 || quad_delta > 2.0)
      ++failures;
  }
  const int64_t messages = validation_errors();
  std::printf("validation: %s, %lld messages\n", messages < 0 ? "not active" : "active",
              static_cast<long long>(messages < 0 ? 0 : messages));
  if (messages > 0)
    ++failures;
  return failures ? 1 : 0;
}
