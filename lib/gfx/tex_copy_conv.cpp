#include "tex_copy_conv.hpp"

#include "../internal.hpp"
#include "../gx/gx.hpp"
#include "../webgpu/gpu.hpp"
#include "texture.hpp"
#include "../gx/gx_fmt.hpp"

#include <absl/container/flat_hash_map.h>

#include "texture_convert.hpp"

using namespace std::string_literals;

namespace aurora::gfx::tex_copy_conv {
static Module Log("aurora::gfx::tex_copy_conv");

using webgpu::g_device;

static constexpr std::string_view VertexShader = R"(
struct VertexOutput {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};
static const float2 positions[3] = {float2(-1.0, 1.0), float2(-1.0, -3.0), float2(3.0, 1.0)};
static const float2 uvs[3] = {float2(0.0, 0.0), float2(0.0, 2.0), float2(2.0, 0.0)};
VertexOutput vs_main(uint vi : SV_VertexID) {
    VertexOutput output;
    output.pos = float4(positions[vi], 0.0, 1.0);
    output.uv = uvs[vi] * uv_xf.scale + uv_xf.offset;
    return output;
}
)"sv;

static const std::string ShaderPreamble = std::string{R"(
SamplerState src_samp : register(s0, space0);
Texture2D src : register(t1, space0);
struct UVTransform {
    float2 offset;
    float2 scale;
    uint opaque_alpha; // the EFB has no alpha: it reads as 1
};
ConstantBuffer<UVTransform> uv_xf : register(b2, space0);
)"} + std::string{VertexShader} + R"(
float4 sample_efb(float2 uv) {
    float4 c = src.Sample(src_samp, uv);
    return float4(c.rgb, uv_xf.opaque_alpha != 0u ? 1.0 : c.a);
}
float intensity(float3 rgb) {
    // ITU-R BT.601 luma coefficients
    return dot(rgb, float3(0.257, 0.504, 0.098)) + 16.0 / 255.0;
}
float quantize4(float v) {
    return floor(v * 16.0) / 15.0;
}
)";

static std::string depth_preamble(bool multisampled) {
  return std::string{multisampled ? "Texture2DMS<float> src : register(t0, space0);\n"
                                  : "Texture2D<float> src : register(t0, space0);\n"} +
         R"(
struct UVTransform {
    float2 offset;
    float2 scale;
};
ConstantBuffer<UVTransform> uv_xf : register(b1, space0);
)" + std::string{VertexShader} +
         R"(
uint gx_z24(float2 uv) {
    uint width, height;
)" + (multisampled ? "    uint samples;\n    src.GetDimensions(width, height, samples);\n"
                   : "    src.GetDimensions(width, height);\n") +
         R"(    int2 size = int2(width, height);
    int2 coord = clamp(int2(floor(uv * float2(size))), int2(0, 0), size - int2(1, 1));
)" + (multisampled ? "    float depth = src.Load(coord, 0);\n" : "    float depth = src.Load(int3(coord, 0));\n") +
         (gx::UseReversedZ ? "    return min(uint(clamp(1.0 - depth, 0.0, 1.0) * 16777215.0 + 0.5), 0x00ffffffu);\n"
                           : "    return min(uint(clamp(depth, 0.0, 1.0) * 16777215.0 + 0.5), 0x00ffffffu);\n") +
         "}\n";
}

// Passthrough blit (for scaling)
static constexpr std::string_view FragPassthrough = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    return sample_efb(input.uv);
}
)"sv;

// GX_TF_I4: 4-bit intensity -> R8Unorm (quantized)
static constexpr std::string_view FragI4 = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    float3 rgb = sample_efb(input.uv).rgb;
    float i = quantize4(intensity(rgb));
    return float4(i, i, i, i);
}
)"sv;

// GX_TF_I8: 8-bit intensity -> R8Unorm
static constexpr std::string_view FragI8 = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    float3 rgb = sample_efb(input.uv).rgb;
    float i = intensity(rgb);
    return float4(i, i, i, i);
}
)"sv;

// GX_TF_IA4: 4-bit intensity + 4-bit alpha -> RG8Unorm
static constexpr std::string_view FragIA4 = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    float4 c = sample_efb(input.uv);
    float i = quantize4(intensity(c.rgb));
    float a = quantize4(c.a);
    return float4(i, i, i, a);
}
)"sv;

// GX_TF_IA8: 8-bit intensity + 8-bit alpha -> RG8Unorm
static constexpr std::string_view FragIA8 = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    float4 c = sample_efb(input.uv);
    float i = intensity(c.rgb);
    return float4(i, i, i, c.a);
}
)"sv;

// GX_TF_RGB565: Blit alpha to 1.0
static constexpr std::string_view FragRGB565 = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    float4 c = sample_efb(input.uv);
    return float4(c.rgb, 1.0);
}
)"sv;

// GX_CTF_R4: 4-bit red -> R8Unorm
static constexpr std::string_view FragR4 = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    float r = quantize4(sample_efb(input.uv).r);
    return float4(r, r, r, r);
}
)"sv;

// GX_CTF_RA4: 4-bit red + 4-bit alpha -> RG8Unorm
static constexpr std::string_view FragRA4 = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    float4 c = sample_efb(input.uv);
    float r = quantize4(c.r);
    return float4(r, r, r, quantize4(c.a));
}
)"sv;

// GX_CTF_RA8: 8-bit red + 8-bit alpha -> RG8Unorm
static constexpr std::string_view FragRA8 = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    float4 c = sample_efb(input.uv);
    return float4(c.r, c.r, c.r, c.a);
}
)"sv;

// GX_CTF_A8: 8-bit alpha -> R8Unorm
static constexpr std::string_view FragA8 = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    float a = sample_efb(input.uv).a;
    return float4(a, a, a, a);
}
)"sv;

// GX_CTF_R8: 8-bit red -> R8Unorm
static constexpr std::string_view FragR8 = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    float r = sample_efb(input.uv).r;
    return float4(r, r, r, r);
}
)"sv;

// GX_CTF_G8: 8-bit green -> R8Unorm
static constexpr std::string_view FragG8 = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    float g = sample_efb(input.uv).g;
    return float4(g, g, g, g);
}
)"sv;

// GX_CTF_B8: 8-bit blue -> R8Unorm
static constexpr std::string_view FragB8 = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    float b = sample_efb(input.uv).b;
    return float4(b, b, b, b);
}
)"sv;

// GX_CTF_RG8: 8-bit red + 8-bit green -> RG8Unorm
static constexpr std::string_view FragRG8 = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    float4 c = sample_efb(input.uv);
    return float4(c.r, c.r, c.r, c.g);
}
)"sv;

// GX_CTF_GB8: 8-bit green + 8-bit blue -> RG8Unorm
static constexpr std::string_view FragGB8 = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    float4 c = sample_efb(input.uv);
    return float4(c.g, c.g, c.g, c.b);
}
)"sv;

// GX_TF_Z16: Upper 16-bits depth -> IA8
static constexpr std::string_view FragZ16 = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    uint z16 = gx_z24(input.uv) >> 8u;
    float i = float((z16 >> 8u) & 0xFFu) / 255.0;
    float a = float(z16 & 0xFFu) / 255.0;
    return float4(i, i, i, a);
}
)"sv;

// Depth channel selection mirrors Dolphin TextureConverterShaderGen: the 24-bit
// depth expands to bytes (hi, mid, lo) and the copy format picks channels.

// GX_CTF_Z4: Upper 4 bits depth -> R (quantized; the depth preamble has no
// quantize4 helper, so quantize inline: floor(v*16)/15)
static constexpr std::string_view FragZ4 = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    float v = float((gx_z24(input.uv) >> 16u) & 0xFFu) / 255.0;
    float r = floor(min(v, 0.999) * 16.0) / 15.0;
    return float4(r, r, r, r);
}
)"sv;

// GX_TF_Z8: Upper 8 bits depth -> R
static constexpr std::string_view FragZ8 = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    float r = float((gx_z24(input.uv) >> 16u) & 0xFFu) / 255.0;
    return float4(r, r, r, r);
}
)"sv;

// GX_CTF_Z8M: Middle 8 bits depth -> R
static constexpr std::string_view FragZ8M = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    float r = float((gx_z24(input.uv) >> 8u) & 0xFFu) / 255.0;
    return float4(r, r, r, r);
}
)"sv;

// GX_CTF_Z8L: Low 8 bits depth -> R
static constexpr std::string_view FragZ8L = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    float r = float(gx_z24(input.uv) & 0xFFu) / 255.0;
    return float4(r, r, r, r);
}
)"sv;

// GX_CTF_Z16L: Low 16 bits depth -> IA8 (mid, low)
static constexpr std::string_view FragZ16L = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    uint z24 = gx_z24(input.uv);
    float i = float((z24 >> 8u) & 0xFFu) / 255.0;
    float a = float(z24 & 0xFFu) / 255.0;
    return float4(i, i, i, a);
}
)"sv;

// GX_TF_Z24X8: Full 24-bit depth -> RGB (hi, mid, lo), alpha 1
static constexpr std::string_view FragZ24X8 = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    uint z24 = gx_z24(input.uv);
    return float4(float((z24 >> 16u) & 0xFFu) / 255.0,
                 float((z24 >> 8u) & 0xFFu) / 255.0,
                 float(z24 & 0xFFu) / 255.0, 1.0);
}
)"sv;

struct ConvPipeline {
  GXTexFmt fmt;
  std::string_view fragShader;
  const char* label;
};

static constexpr std::array ConvPipelines{
    ConvPipeline{GX_TF_I4, FragI4, "TexCopyConv I4"},
    ConvPipeline{GX_TF_I8, FragI8, "TexCopyConv I8"},
    ConvPipeline{GX_TF_IA4, FragIA4, "TexCopyConv IA4"},
    ConvPipeline{GX_TF_IA8, FragIA8, "TexCopyConv IA8"},
    ConvPipeline{GX_TF_RGB565, FragRGB565, "TexCopyConv RGB565"},
    ConvPipeline{GX_CTF_R4, FragR4, "TexCopyConv R4"},
    ConvPipeline{GX_CTF_RA4, FragRA4, "TexCopyConv RA4"},
    ConvPipeline{GX_CTF_RA8, FragRA8, "TexCopyConv RA8"},
    ConvPipeline{GX_CTF_A8, FragA8, "TexCopyConv A8"},
    ConvPipeline{GX_CTF_R8, FragR8, "TexCopyConv R8"},
    ConvPipeline{GX_CTF_G8, FragG8, "TexCopyConv G8"},
    ConvPipeline{GX_CTF_B8, FragB8, "TexCopyConv B8"},
    ConvPipeline{GX_CTF_RG8, FragRG8, "TexCopyConv RG8"},
    ConvPipeline{GX_CTF_GB8, FragGB8, "TexCopyConv GB8"},
};

static constexpr std::array DepthConvPipelines{
    ConvPipeline{GX_TF_Z16, FragZ16, "TexCopyConv Z16"},
    ConvPipeline{GX_CTF_Z4, FragZ4, "TexCopyConv Z4"},
    ConvPipeline{GX_TF_Z8, FragZ8, "TexCopyConv Z8"},
    ConvPipeline{GX_CTF_Z8M, FragZ8M, "TexCopyConv Z8M"},
    ConvPipeline{GX_CTF_Z8L, FragZ8L, "TexCopyConv Z8L"},
    ConvPipeline{GX_CTF_Z16L, FragZ16L, "TexCopyConv Z16L"},
    ConvPipeline{GX_TF_Z24X8, FragZ24X8, "TexCopyConv Z24X8"},
};

using Pipeline = std::unique_ptr<rhi::RenderPipeline>;

static std::unique_ptr<rhi::BindGroupLayout> g_bindGroupLayout;
static std::unique_ptr<rhi::BindGroupLayout> g_depthBindGroupLayout;
static std::unique_ptr<rhi::PipelineLayout> g_pipelineLayout;
static std::unique_ptr<rhi::PipelineLayout> g_depthPipelineLayout;
// Depth conversions reading a multisampled EFB (MSAA), by format.
static absl::flat_hash_map<GXTexFmt, Pipeline> g_msPipelines;
static std::unique_ptr<rhi::Sampler> g_nearestSampler;
static std::unique_ptr<rhi::Sampler> g_linearSampler;
static absl::flat_hash_map<GXTexFmt, Pipeline> g_pipelines;
static Pipeline g_blitPipeline;

static Pipeline create_pipeline(std::string_view fragShader, rhi::Format format, const std::string& preamble,
                                rhi::PipelineLayout* layout) {
  std::string source = preamble;
  source += fragShader;
  const auto vs = webgpu::compile_shader(rhi::ShaderStage::Vertex, source, "vs_main");
  const auto fs = webgpu::compile_shader(rhi::ShaderStage::Fragment, source, "fs_main");
  rhi::RenderPipelineDesc desc;
  desc.layout = layout;
  desc.vertex = &vs;
  desc.vertex_entry = "vs_main";
  desc.fragment = &fs;
  desc.fragment_entry = "fs_main";
  desc.color_format = format;
  return g_device->create_render_pipeline(desc);
}

bool needs_conversion(const GXTexFmt fmt) { return g_pipelines.contains(fmt); }

void initialize() {
  const std::array entries{
      rhi::BindingLayout{0, rhi::BindingKind::Sampler},
      rhi::BindingLayout{1, rhi::BindingKind::Texture},
      rhi::BindingLayout{2, rhi::BindingKind::Uniform},
  };
  g_bindGroupLayout = g_device->create_bind_group_layout({entries.data(), entries.size()});
  rhi::BindGroupLayout* groups[] = {g_bindGroupLayout.get()};
  g_pipelineLayout = g_device->create_pipeline_layout({groups, 1});

  const std::array depthEntries{
      rhi::BindingLayout{0, rhi::BindingKind::Texture},
      rhi::BindingLayout{1, rhi::BindingKind::Uniform},
  };
  g_depthBindGroupLayout = g_device->create_bind_group_layout({depthEntries.data(), depthEntries.size()});
  rhi::BindGroupLayout* depthGroups[] = {g_depthBindGroupLayout.get()};
  g_depthPipelineLayout = g_device->create_pipeline_layout({depthGroups, 1});

  g_blitPipeline =
      create_pipeline(FragPassthrough, webgpu::g_graphicsConfig.surfaceFormat, ShaderPreamble, g_pipelineLayout.get());
  for (const auto& conv : ConvPipelines) {
    g_pipelines[conv.fmt] = create_pipeline(conv.fragShader, to_rhi(conv.fmt), ShaderPreamble, g_pipelineLayout.get());
  }
  const std::string depthPreamble = depth_preamble(false);
  const std::string msPreamble = depth_preamble(true);
  for (const auto& conv : DepthConvPipelines) {
    g_msPipelines[conv.fmt] = create_pipeline(conv.fragShader, to_rhi(conv.fmt), msPreamble, g_depthPipelineLayout.get());
    g_pipelines[conv.fmt] = create_pipeline(conv.fragShader, to_rhi(conv.fmt), depthPreamble, g_depthPipelineLayout.get());
  }

  rhi::SamplerDesc sampler;
  sampler.u = rhi::AddressMode::ClampToEdge;
  sampler.v = rhi::AddressMode::ClampToEdge;
  sampler.mip = rhi::Filter::Nearest;
  sampler.min = rhi::Filter::Nearest;
  sampler.mag = rhi::Filter::Nearest;
  g_nearestSampler = g_device->create_sampler(sampler);
  sampler.min = rhi::Filter::Linear;
  sampler.mag = rhi::Filter::Linear;
  g_linearSampler = g_device->create_sampler(sampler);
}

void shutdown() {
  g_pipelines.clear();
  g_msPipelines.clear();
  g_blitPipeline.reset();
  g_pipelineLayout.reset();
  g_depthPipelineLayout.reset();
  g_bindGroupLayout.reset();
  g_depthBindGroupLayout.reset();
  g_nearestSampler.reset();
  g_linearSampler.reset();
}

static void execute(rhi::CommandList& list, const ConvRequest& req, rhi::RenderPipeline& pipeline) {
  std::unique_ptr<rhi::BindGroup> bindGroup;
  if (gx::is_depth_format(req.fmt)) {
    std::array<rhi::BindingValue, 2> values{};
    values[0].binding = 0;
    values[0].view = req.srcView.get();
    values[1].binding = 1;
    values[1].buffer = g_uniformBuffer;
    values[1].offset = req.uniformRange.offset;
    values[1].size = req.uniformRange.size;
    bindGroup = g_device->create_bind_group({g_depthBindGroupLayout.get(), values.data(), values.size()});
  } else {
    std::array<rhi::BindingValue, 3> values{};
    values[0].binding = 0;
    values[0].sampler = req.sampleFilter == SampleFilter::Linear ? g_linearSampler.get() : g_nearestSampler.get();
    values[1].binding = 1;
    values[1].view = req.srcView.get();
    values[2].binding = 2;
    values[2].buffer = g_uniformBuffer;
    values[2].offset = req.uniformRange.offset;
    values[2].size = req.uniformRange.size;
    bindGroup = g_device->create_bind_group({g_bindGroupLayout.get(), values.data(), values.size()});
  }
  rhi::RenderPassDesc pass;
  pass.color.target = req.dst->texture.get();
  pass.color.clear = true;
  list.begin_render_pass(pass);
  list.set_pipeline(pipeline);
  list.set_bind_group(0, *bindGroup, nullptr, 0);
  list.draw(3, 0);
  list.end_render_pass();
}

void run(rhi::CommandList& list, const ConvRequest& req) {
  const auto& pipelines = req.multisampled ? g_msPipelines : g_pipelines;
  const auto it = pipelines.find(req.fmt);
  if (it == pipelines.end()) {
    Log.fatal("No copy conversion pipeline for format {}", static_cast<int>(req.fmt));
  }
  execute(list, req, *it->second);
}

void blit(rhi::CommandList& list, const ConvRequest& req) { execute(list, req, *g_blitPipeline); }

} // namespace aurora::gfx::tex_copy_conv
