// Runs the same work on every backend and compares the pixels, headless: the same
// bytes on both, or close to it where the APIs leave rounding to the driver.
#include "rhi.hpp"
#include "../../lib/gx/draw_packet.hpp" // compiles with no graphics API in sight

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

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

// Quads placed by the draw constant and the instance: what a GX draw's vertex offset
// and its expanded lines ask for.
const char* const DrawConstantShader = R"(
struct DrawConstants {
  uint value;
};
[[vk::push_constant]] ConstantBuffer<DrawConstants> draw : register(b0, space0);
struct Varyings {
  float4 position : SV_Position;
  float3 color : TEXCOORD0;
};
Varyings vs_main(uint vertex : SV_VertexID, uint instance : SV_InstanceID) {
  static const float2 corners[6] = {float2(0, 0), float2(1, 0), float2(1, 1), float2(0, 0), float2(1, 1), float2(0, 1)};
  const float top = draw.value < 100u ? 0.0 : -1.0;
  Varyings output;
  output.position = float4(-1.0 + (float(instance) + corners[vertex].x) * 0.5, top + corners[vertex].y, 0.0, 1.0);
  output.color = float3(float(draw.value) / 255.0, float((instance + 1u) * 40u) / 255.0, 0.0);
  return output;
}
float4 fs_main(Varyings input) : SV_Target {
  return float4(input.color, 1.0);
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

// Two instanced draws with their own constants, one of them indexed. Returns the
// image, or nothing if a quad isn't where and what it should be.
Pixels render_draw_constants(Device& device, int& failures) {
  static const uint16_t indices[] = {0, 1, 2, 3, 4, 5};
  auto upload = must(device.create_buffer({sizeof(indices), MemoryKind::Upload}), "upload");
  auto index_buffer = must(device.create_buffer({sizeof(indices)}), "indices");
  auto readback = must(device.create_buffer({uint64_t{BytesPerRow} * Height, MemoryKind::Readback}), "readback");
  std::memcpy(upload->mapped(), indices, sizeof(indices));
  auto layout = must(device.create_pipeline_layout({nullptr, 0, true}), "draw constant layout");
  const ShaderCode vertex = compile(device, ShaderStage::Vertex, DrawConstantShader, "vs_main");
  const ShaderCode fragment = compile(device, ShaderStage::Fragment, DrawConstantShader, "fs_main");
  RenderPipelineDesc desc;
  desc.layout = layout.get();
  desc.vertex = &vertex;
  desc.vertex_entry = "vs_main";
  desc.fragment = &fragment;
  desc.fragment_entry = "fs_main";
  auto pipeline = must(device.create_render_pipeline(desc), "draw constant pipeline");
  auto target =
      must(device.create_texture({Width, Height, 1, 1, Format::RGBA8Unorm, TextureSampled | TextureColorTarget}), "target");
  auto list = must(device.create_command_list(), "command list");
  list->begin();
  list->copy_buffer(*upload, 0, *index_buffer, 0, sizeof(indices));
  RenderPassDesc pass;
  pass.color.target = target.get();
  pass.color.clear = true;
  list->begin_render_pass(pass);
  list->set_pipeline(*pipeline);
  list->set_index_buffer_u16(*index_buffer, 0);
  list->set_draw_constant(10);
  list->draw(6, 0, 2);
  list->set_draw_constant(200);
  list->draw_indexed(6, 0, 0, 3);
  list->end_render_pass();
  list->copy_texture_to_buffer(*target, {0}, *readback, 0, BytesPerRow);
  list->end();
  const uint64_t ticket = device.submit(*list);
  device.wait(ticket);
  if (!device.is_done(ticket)) {
    std::printf("the ticket isn't done after waiting for it\n");
    ++failures;
  }
  Pixels pixels = read_pixels(*readback, 0);
  const struct {
    uint32_t x, y;
    uint8_t red, green;
  } expected[] = {
      {Width / 8, Height / 4, 10, 40},        {Width * 3 / 8, Height / 4, 10, 80},
      {Width * 5 / 8, Height / 4, 0, 0},      {Width / 8, Height * 3 / 4, 200, 40},
      {Width * 5 / 8, Height * 3 / 4, 200, 120}, {Width * 7 / 8, Height * 3 / 4, 0, 0},
  };
  for (const auto& at : expected) {
    const uint8_t* pixel = pixels.data() + (size_t{at.y} * Width + at.x) * 4;
    if (pixel[0] != at.red || pixel[1] != at.green) {
      std::printf("draw constant at %u,%u: got %u %u, wanted %u %u\n", at.x, at.y, pixel[0], pixel[1], at.red, at.green);
      ++failures;
    }
  }
  return pixels;
}

#ifdef _WIN32
// Frames through a real swapchain, in a window never shown, in every present mode
// and across a resize: each frame's color has to reach the image that was acquired.
// Each mip of a BC1 texture down to 1x1 written whole blocks at a time and read
// back, smaller than a block ones too.
int test_compressed_mips(Device& device) {
  constexpr uint32_t Mips = 5;
  constexpr uint64_t Slot = 512;
  auto texture = must(device.create_texture({16, 16, Mips, 1, Format::BC1, TextureSampled}), "bc1");
  auto upload = must(device.create_buffer({Slot * Mips, MemoryKind::Upload}), "upload");
  auto readback = must(device.create_buffer({Slot * Mips, MemoryKind::Readback}), "readback");
  auto* const in = static_cast<uint8_t*>(upload->mapped());
  for (uint64_t at = 0; at < Slot * Mips; ++at) {
    in[at] = static_cast<uint8_t>(at * 31 + at / Slot);
  }
  auto list = must(device.create_command_list(), "list");
  list->begin();
  for (uint32_t mip = 0; mip < Mips; ++mip) {
    const uint32_t blocks = std::max(((16u >> mip) + 3) / 4, 1u);
    TextureRegion region;
    region.mip = mip;
    region.width = blocks * 4;
    region.height = blocks * 4;
    list->copy_buffer_to_texture(*upload, Slot * mip, 256, *texture, region);
  }
  for (uint32_t mip = 0; mip < Mips; ++mip) {
    TextureRegion region;
    region.mip = mip;
    list->copy_texture_to_buffer(*texture, region, *readback, Slot * mip, 256);
  }
  list->end();
  device.wait(device.submit(*list));
  const auto* const back = static_cast<const uint8_t*>(readback->mapped());
  int failures = 0;
  for (uint32_t mip = 0; mip < Mips; ++mip) {
    const uint32_t blocks = std::max(((16u >> mip) + 3) / 4, 1u);
    for (uint32_t row = 0; row < blocks; ++row) {
      if (std::memcmp(in + Slot * mip + row * 256, back + Slot * mip + row * 256, blocks * 8) != 0) {
        ++failures;
      }
    }
  }
  return failures;
}

// A buffer destroyed while a list that copies from it is still open, with
// another list submitted and finished meanwhile: it has to outlive the first.
// Only validation tells.
void test_retire_while_open(Device& device) {
  auto source = must(device.create_buffer({256, MemoryKind::Upload}), "source");
  auto target = must(device.create_buffer({256, MemoryKind::Readback}), "target");
  auto open = must(device.create_command_list(), "open list");
  auto other = must(device.create_command_list(), "other list");
  open->begin();
  open->copy_buffer(*source, 0, *target, 0, 256);
  source.reset();
  other->begin();
  other->end();
  device.wait(device.submit(*other));
  other->begin();
  other->end();
  device.wait(device.submit(*other));
  open->end();
  device.wait(device.submit(*open));
}

int test_swapchain(Device& device) {
  WNDCLASSA window_class{};
  window_class.lpfnWndProc = DefWindowProcA;
  window_class.hInstance = GetModuleHandleA(nullptr);
  window_class.lpszClassName = "aurora_rhi_test";
  RegisterClassA(&window_class);
  const HWND window = CreateWindowExA(0, window_class.lpszClassName, "rhi", WS_POPUP, 0, 0, 64, 48, nullptr,
                                      nullptr, window_class.hInstance, nullptr);
  int failures = 0;
  auto readback = must(device.create_buffer({uint64_t{BytesPerRow} * 256, MemoryKind::Readback}), "readback");
  auto list = must(device.create_command_list(), "command list");
  uint32_t frame = 0;
  for (PresentMode mode : {PresentMode::Fifo, PresentMode::Mailbox, PresentMode::Immediate}) {
    SetWindowPos(window, nullptr, 0, 0, 64, 48, SWP_NOMOVE | SWP_NOZORDER);
    SwapchainDesc desc;
    desc.window = {window_class.hInstance, window};
    desc.width = 64;
    desc.height = 48;
    desc.mode = mode;
    auto swapchain = device.create_swapchain(desc);
    if (!swapchain) {
      std::printf("no swapchain\n");
      ++failures;
      continue;
    }
    for (uint32_t i = 0; i < 16; ++i) {
      if (i == 8) {
        SetWindowPos(window, nullptr, 0, 0, 96, 80, SWP_NOMOVE | SWP_NOZORDER);
        swapchain->resize(96, 80);
      }
      Texture* image = swapchain->acquire();
      if (!image) {
        std::printf("acquire failed at frame %u\n", i);
        ++failures;
        break;
      }
      const float red = float(++frame % 200 + 20);
      list->begin();
      RenderPassDesc pass;
      pass.color.target = image;
      pass.color.clear = true;
      pass.color.clear_color = {red / 255.0f, 0.25f, 0.4f, 1.0f};
      list->begin_render_pass(pass);
      list->end_render_pass();
      list->copy_texture_to_buffer(*image, {}, *readback, 0, BytesPerRow);
      list->end();
      device.submit(*list);
      swapchain->present();
      device.wait_idle();
      // BGRA, in the last pixel of the frame's size.
      const uint32_t width = i < 8 ? 64 : 96, height = i < 8 ? 48 : 80;
      const auto* pixel = static_cast<const uint8_t*>(readback->mapped()) + size_t{height - 1} * BytesPerRow + (width - 1) * 4;
      if (pixel[2] != uint8_t(red) || pixel[1] != 64 || pixel[0] != 102) {
        std::printf("frame %u: got %u %u %u, wanted %u 64 102\n", i, pixel[2], pixel[1], pixel[0], uint8_t(red));
        ++failures;
      }
    }
  }
  DestroyWindow(window);
  UnregisterClassA(window_class.lpszClassName, window_class.hInstance);
  return failures;
}
#endif

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
  int64_t provoked = 0;
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
    results[i].push_back(render_draw_constants(*device, failures));
    device->wait_idle();
    const int bc_failures = test_compressed_mips(*device);
    std::printf("%s: compressed mips %s\n", backends[i].name, bc_failures ? "failed" : "ok");
    failures += bc_failures;
    test_retire_while_open(*device);
#ifdef _WIN32
    const int swapchain_failures = test_swapchain(*device);
    std::printf("%s: swapchain %s\n", backends[i].name, swapchain_failures ? "failed" : "ok");
    failures += swapchain_failures;
#endif
    if (options.validation) {
      // Zero messages mean something only if this one is caught: a draw with no
      // pipeline, recorded and never submitted.
      const int64_t before = validation_errors();
      auto target = must(device->create_texture({4, 4, 1, 1, Format::RGBA8Unorm, TextureColorTarget}), "target");
      auto list = must(device->create_command_list(), "command list");
      list->begin();
      RenderPassDesc pass;
      pass.color.target = target.get();
      list->begin_render_pass(pass);
      list->draw(3, 0);
      list->end_render_pass();
      list->end();
      device->wait_idle();
      const int64_t caught = validation_errors() - before;
      std::printf("%s: the layer caught the bad draw with %lld messages\n", backends[i].name,
                  static_cast<long long>(caught));
      if (before < 0 || caught <= 0)
        ++failures;
      provoked += caught;
    }
  }
  if (results[0].size() != results[1].size() || results[0].empty()) {
    std::printf("no results to compare\n");
    return 1;
  }
  const char* const names[] = {"scene 4x", "depth read", "scene 8x", "draw constants"};
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
  const int64_t messages = validation_errors() - provoked;
  std::printf("validation: %s, %lld messages\n", messages < 0 ? "not active" : "active",
              static_cast<long long>(messages < 0 ? 0 : messages));
  if (messages > 0)
    ++failures;
  return failures ? 1 : 0;
}
