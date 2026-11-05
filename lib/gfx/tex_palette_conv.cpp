#include "tex_palette_conv.hpp"

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "texture.hpp"

#include <array>
#include <string>

namespace aurora::gfx::tex_palette_conv {
static Module Log("aurora::gfx::tex_palette_conv");

using webgpu::g_device;

static constexpr std::string_view ShaderPreambleVtx = R"(
struct VertexOutput {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};
static const float2 positions[3] = {float2(-1.0, 1.0), float2(-1.0, -3.0), float2(3.0, 1.0)};
static const float2 uvs[3] = {float2(0.0, 0.0), float2(0.0, 2.0), float2(2.0, 0.0)};
VertexOutput vs_main(uint vi : SV_VertexID) {
    VertexOutput output;
    output.pos = float4(positions[vi], 0.0, 1.0);
    output.uv = uvs[vi];
    return output;
}
)"sv;

// Direct: R16Sint index texture + TLUT -> RGBA8
static constexpr std::string_view ShaderDirect = R"(
Texture2D<int> src : register(t0, space0);
Texture2D tlut : register(t1, space0);
float4 fs_main(VertexOutput input) : SV_Target {
    uint width, height;
    src.GetDimensions(width, height);
    int2 coord = int2(floor(input.uv * float2(width, height)));
    int idx = src.Load(int3(coord, 0)).r;
    return tlut.Load(int3(idx, 0, 0));
}
)"sv;

// FromFloat8: f32 texture (R8Unorm) -> 8-bit index -> TLUT -> RGBA8
static constexpr std::string_view ShaderFromFloat8 = R"(
Texture2D src : register(t0, space0);
Texture2D tlut : register(t1, space0);
float4 fs_main(VertexOutput input) : SV_Target {
    uint width, height;
    src.GetDimensions(width, height);
    int2 coord = int2(floor(input.uv * float2(width, height)));
    float r = src.Load(int3(coord, 0)).r;
    return tlut.Load(int3(int(r * 255.0), 0, 0));
}
)"sv;

// FromFloat4: f32 texture (R8Unorm) -> 4-bit index -> TLUT -> RGBA8
static constexpr std::string_view ShaderFromFloat4 = R"(
Texture2D src : register(t0, space0);
Texture2D tlut : register(t1, space0);
float4 fs_main(VertexOutput input) : SV_Target {
    uint width, height;
    src.GetDimensions(width, height);
    int2 coord = int2(floor(input.uv * float2(width, height)));
    float r = src.Load(int3(coord, 0)).r;
    return tlut.Load(int3(int(r * 15.0), 0, 0));
}
)"sv;

static std::unique_ptr<rhi::BindGroupLayout> g_bindGroupLayout;
static std::unique_ptr<rhi::PipelineLayout> g_pipelineLayout;
static std::unique_ptr<rhi::RenderPipeline> g_directPipeline;
static std::unique_ptr<rhi::RenderPipeline> g_fromFloat8Pipeline;
static std::unique_ptr<rhi::RenderPipeline> g_fromFloat4Pipeline;

static std::unique_ptr<rhi::RenderPipeline> create_pipeline(std::string_view fragment) {
  std::string source{ShaderPreambleVtx};
  source += fragment;
  const auto vs = webgpu::compile_shader(rhi::ShaderStage::Vertex, source, "vs_main");
  const auto fs = webgpu::compile_shader(rhi::ShaderStage::Fragment, source, "fs_main");
  rhi::RenderPipelineDesc desc;
  desc.layout = g_pipelineLayout.get();
  desc.vertex = &vs;
  desc.vertex_entry = "vs_main";
  desc.fragment = &fs;
  desc.fragment_entry = "fs_main";
  desc.color_format = rhi::Format::RGBA8Unorm;
  return g_device->create_render_pipeline(desc);
}

static rhi::RenderPipeline& pipeline_for_variant(Variant variant) {
  switch (variant) {
  case Variant::Direct:
    return *g_directPipeline;
  case Variant::FromFloat8:
    return *g_fromFloat8Pipeline;
  case Variant::FromFloat4:
    return *g_fromFloat4Pipeline;
  }
  FATAL("invalid palette conv variant {}", static_cast<int>(variant));
}

void initialize() {
  const std::array entries{
      rhi::BindingLayout{0, rhi::BindingKind::Texture},
      rhi::BindingLayout{1, rhi::BindingKind::Texture},
  };
  g_bindGroupLayout = g_device->create_bind_group_layout({entries.data(), entries.size()});
  rhi::BindGroupLayout* groups[] = {g_bindGroupLayout.get()};
  g_pipelineLayout = g_device->create_pipeline_layout({groups, 1});
  g_directPipeline = create_pipeline(ShaderDirect);
  g_fromFloat8Pipeline = create_pipeline(ShaderFromFloat8);
  g_fromFloat4Pipeline = create_pipeline(ShaderFromFloat4);
}

void shutdown() {
  g_directPipeline.reset();
  g_fromFloat8Pipeline.reset();
  g_fromFloat4Pipeline.reset();
  g_pipelineLayout.reset();
  g_bindGroupLayout.reset();
}

void run(rhi::CommandList& list, const ConvRequest& req) {
  std::array<rhi::BindingValue, 2> values{};
  values[0].binding = 0;
  values[0].view = req.src->sampleTextureView.get();
  values[1].binding = 1;
  values[1].view = req.tlut->sampleTextureView.get();
  const auto bindGroup = g_device->create_bind_group({g_bindGroupLayout.get(), values.data(), values.size()});

  rhi::RenderPassDesc pass;
  pass.color.target = req.dst->texture.get();
  pass.color.clear = true;
  list.begin_render_pass(pass);
  list.set_pipeline(pipeline_for_variant(req.variant));
  list.set_bind_group(0, *bindGroup, nullptr, 0);
  list.draw(3, 0);
  list.end_render_pass();
}

} // namespace aurora::gfx::tex_palette_conv
