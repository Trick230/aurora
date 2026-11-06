#include "ambient_occlusion.hpp"

#include "../gpu/gpu.hpp"
#include "../internal.hpp"

#include <array>
#include <cmath>

// GTAO (Jimenez et al., "Practical Real-Time Strategies for Accurate Indirect
// Occlusion", 2016), after Intel's XeGTAO: per pixel, a few slices around the
// view vector, the horizon on each side found by marching the depth buffer,
// and the visible arc integrated against the normal. A 4x4 noise pattern
// rotates the slices and staggers the steps; the pass that applies the result
// averages each 4x4 block, weighted by depth, which cancels the pattern
// without blurring across edges.
//
// The radius is a share of the viewport's height, so the effect looks alike
// in any game's units; it fades out toward the far plane.

namespace aurora::gfx::ao {
namespace {
using gpu::g_device;

// Above this EFB height the occlusion is computed at half resolution.
constexpr uint32_t FullResolutionMaxHeight = 1600;
constexpr float Radius = 0.035f;    // of the viewport's height
constexpr float Strength = 0.85f;   // 0 leaves the image, 1 applies all of it
constexpr float FadeStart = 0.25f;  // of the distance to the far plane
constexpr float FadeEnd = 0.6f;

struct Params {
  float viewport[4]; // left, top, width, height in EFB pixels
  float proj[4];     // GX's p0..p3: x = p0 x + p1 z, y = p2 y + p3 z
  float depth[4];    // p4, p5 (z = p4 z + p5), window depth at depth 0, its span to depth 1
  float size[4];     // EFB width, height, occlusion scale, unused
  float tune[4];     // radius, strength, fade start, fade end
};

constexpr std::string_view ParamsStruct = R"(
struct AoParams {
    float4 viewport;
    float4 proj;
    float4 depth;
    float4 size;
    float4 tune;
};
)";

constexpr std::string_view Common = R"(
struct VertexOutput {
    float4 pos : SV_Position;
};
static const float2 positions[3] = {float2(-1.0, 1.0), float2(-1.0, -3.0), float2(3.0, 1.0)};
VertexOutput vs_main(uint vi : SV_VertexID) {
    VertexOutput output;
    output.pos = float4(positions[vi], 0.0, 1.0);
    return output;
}
// Reversed Z: 1 at the near plane, 0 at the far one.
float ndc_depth(float d) {
    return (d - params.depth.z) / params.depth.w;
}
bool is_geometry(float d) {
    float n = ndc_depth(d);
    return n > 1e-6 && n < 1.0;
}
// View space: x right, y up, looking down -z.
float view_z(float d) {
    return params.depth.y / (ndc_depth(d) - params.depth.x);
}
float3 view_pos(float2 px, float d) {
    float z = view_z(d);
    float2 ndc = float2((px.x - params.viewport.x) / params.viewport.z * 2.0 - 1.0,
                        1.0 - (px.y - params.viewport.y) / params.viewport.w * 2.0);
    return float3(-z * (ndc.x + params.proj.y) / params.proj.x, -z * (ndc.y + params.proj.w) / params.proj.z, z);
}
// The viewport's pixels that are in the EFB.
float2 bounds_min() { return max(params.viewport.xy, float2(0.0, 0.0)); }
float2 bounds_max() { return min(params.viewport.xy + params.viewport.zw, params.size.xy); }
int2 clamp_px(int2 p) {
    return clamp(p, int2(bounds_min()), int2(bounds_max()) - 1);
}
)";

constexpr std::string_view OcclusionShader = R"(
static const float Pi = 3.14159265;
static const uint Slices = 3;
static const uint Steps = 4;
static const float Bayer[16] = {0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5};

float fast_acos(float x) {
    float a = abs(x);
    float r = (-0.156583 * a + Pi * 0.5) * sqrt(1.0 - a);
    return x >= 0.0 ? r : Pi - r;
}

float fs_main(VertexOutput input) : SV_Target {
    int2 aoPx = int2(input.pos.xy);
    int scale = int(params.size.z);
    int2 c = aoPx * scale + scale / 2;
    float2 vpMin = bounds_min();
    float2 vpMax = bounds_max();
    if (any(float2(c) < vpMin) || any(float2(c) >= vpMax))
        return 1.0;
    float d0 = load_depth(c);
    if (!is_geometry(d0))
        return 1.0;
    float2 center = float2(c) + 0.5;
    float3 P = view_pos(center, d0);

    // The normal from the depth: on each axis the side nearer in depth, so
    // edges don't bend it.
    float3 pl = view_pos(center - float2(1, 0), load_depth(clamp_px(c - int2(1, 0))));
    float3 pr = view_pos(center + float2(1, 0), load_depth(clamp_px(c + int2(1, 0))));
    float3 pu = view_pos(center - float2(0, 1), load_depth(clamp_px(c - int2(0, 1))));
    float3 pd = view_pos(center + float2(0, 1), load_depth(clamp_px(c + int2(0, 1))));
    float3 dx = abs(pr.z - P.z) < abs(P.z - pl.z) ? pr - P : P - pl;
    float3 dy = abs(pd.z - P.z) < abs(P.z - pu.z) ? pd - P : P - pu;
    float3 N = cross(dx, dy);
    float nLen = length(N);
    if (!(nLen > 0.0))
        return 1.0;
    N /= nLen;
    float3 V = normalize(-P);
    if (dot(N, V) < 0.0)
        N = -N;

    float radiusPx = params.tune.x * params.viewport.w;
    // View-space length of one EFB pixel at this depth.
    float pxToView = -P.z * 2.0 / (abs(params.proj.z) * params.viewport.w);
    float radius = radiusPx * pxToView;
    float falloffRange = 0.615 * radius;
    float falloffMul = -1.0 / falloffRange;
    float falloffAdd = (radius - falloffRange) / falloffRange + 1.0;

    uint2 q = uint2(aoPx) & 3u;
    float sliceNoise = Bayer[q.y * 4u + q.x] / 16.0;
    float stepNoise = frac(sliceNoise * 5.0 + 0.5);

    float visibility = 0.0;
    [unroll]
    for (uint s = 0; s < Slices; ++s) {
        float phi = (float(s) + sliceNoise) * Pi / float(Slices);
        float3 dir = float3(cos(phi), sin(phi), 0.0);
        float2 omega = float2(dir.x, -dir.y); // screen y runs down
        float3 orthoDir = dir - dot(dir, V) * V;
        float3 axis = normalize(cross(orthoDir, V));
        float3 projN = N - axis * dot(N, axis);
        float projNLen = length(projN);
        if (!(projNLen > 1e-6))
            continue;
        float cosN = saturate(dot(projN, V) / projNLen);
        float n = (dot(orthoDir, projN) < 0.0 ? -1.0 : 1.0) * fast_acos(cosN);
        float horizon0 = cos(n + Pi * 0.5);
        float horizon1 = cos(n - Pi * 0.5);
        float low0 = horizon0, low1 = horizon1;
        [unroll]
        for (uint j = 0; j < Steps; ++j) {
            float t = (float(j) + stepNoise) / float(Steps);
            float offset = max(t * t * radiusPx, float(j) + 1.0);
            [unroll]
            for (uint side = 0; side < 2; ++side) {
                float2 spx = center + (side == 0 ? omega : -omega) * offset;
                if (any(spx < vpMin) || any(spx >= vpMax))
                    continue;
                float sd = load_depth(int2(spx));
                if (!is_geometry(sd))
                    continue;
                float3 delta = view_pos(floor(spx) + 0.5, sd) - P;
                float dist = length(delta);
                float cosH = dot(delta / max(dist, 1e-6), V);
                float weight = saturate(dist * falloffMul + falloffAdd);
                if (side == 0)
                    horizon0 = max(horizon0, lerp(low0, cosH, weight));
                else
                    horizon1 = max(horizon1, lerp(low1, cosH, weight));
            }
        }
        float h0 = -fast_acos(horizon1);
        float h1 = fast_acos(horizon0);
        h0 = n + clamp(h0 - n, -Pi * 0.5, Pi * 0.5);
        h1 = n + clamp(h1 - n, -Pi * 0.5, Pi * 0.5);
        float sinN = sin(n);
        float arc0 = (cosN + 2.0 * h0 * sinN - cos(2.0 * h0 - n)) * 0.25;
        float arc1 = (cosN + 2.0 * h1 * sinN - cos(2.0 * h1 - n)) * 0.25;
        visibility += projNLen * (arc0 + arc1);
    }
    visibility = saturate(visibility / float(Slices));

    // Toward the far plane, where the radius spans a lot of the world, it fades.
    float farZ = -params.depth.y / params.depth.x;
    float fade = 1.0 - smoothstep(params.tune.z, params.tune.w, P.z / farZ);
    return lerp(1.0, visibility, params.tune.y * fade);
}
)";

constexpr std::string_view ApplyShader = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    int2 c = int2(input.pos.xy);
    float d0 = load_depth(c);
    if (!is_geometry(d0))
        return float4(1.0, 1.0, 1.0, 1.0);
    float z0 = view_z(d0);
    int scale = int(params.size.z);
    uint aoWidth, aoHeight;
    occlusion.GetDimensions(aoWidth, aoHeight);
    int2 aoMax = int2(aoWidth, aoHeight) - 1;
    // Every 4x4 block holds the noise pattern once; weighting by depth keeps
    // the average on this pixel's surface.
    int2 base = int2(floor((input.pos.xy) / float(scale) - 0.5)) - 1;
    float sum = 0.0, weights = 0.0;
    [unroll]
    for (int y = 0; y < 4; ++y) {
        [unroll]
        for (int x = 0; x < 4; ++x) {
            int2 t = clamp(base + int2(x, y), int2(0, 0), aoMax);
            float td = load_depth(clamp_px(t * scale + scale / 2));
            float w = is_geometry(td) ? exp2(-abs(view_z(td) - z0) / (abs(z0) * 0.02)) : 0.0;
            sum += occlusion.Load(int3(t, 0)) * w;
            weights += w;
        }
    }
    float ao = weights > 1e-4 ? sum / weights : 1.0;
    return float4(ao, ao, ao, 1.0);
}
)";

std::string depth_access(bool multisampled) {
  return multisampled ? "Texture2DMS<float> depth_tex : register(t0, space0);\n"
                        "float load_depth(int2 p) { return depth_tex.Load(p, 0); }\n"
                      : "Texture2D<float> depth_tex : register(t0, space0);\n"
                        "float load_depth(int2 p) { return depth_tex.Load(int3(p, 0)); }\n";
}

std::string occlusion_source(bool multisampled) {
  return std::string{ParamsStruct} + depth_access(multisampled) +
         "ConstantBuffer<AoParams> params : register(b1, space0);\n" + std::string{Common} +
         std::string{OcclusionShader};
}

std::string apply_source(bool multisampled) {
  return std::string{ParamsStruct} + depth_access(multisampled) +
         "Texture2D<float> occlusion : register(t1, space0);\n" +
         "ConstantBuffer<AoParams> params : register(b2, space0);\n" + std::string{Common} + std::string{ApplyShader};
}

using Pipeline = std::unique_ptr<rhi::RenderPipeline>;

bool g_enabled = false;
std::unique_ptr<rhi::BindGroupLayout> g_occlusionGroupLayout;
std::unique_ptr<rhi::BindGroupLayout> g_applyGroupLayout;
std::unique_ptr<rhi::PipelineLayout> g_occlusionLayout;
std::unique_ptr<rhi::PipelineLayout> g_applyLayout;
std::array<Pipeline, 2> g_occlusionPipelines; // by whether the depth is multisampled
std::array<Pipeline, 8> g_applyPipelines;     // by log2 of the color's samples, then multisampled depth
std::shared_ptr<rhi::Texture> g_texture;      // the occlusion, R8
std::unique_ptr<rhi::TextureView> g_view;
uint32_t g_textureWidth = 0, g_textureHeight = 0;

uint32_t occlusion_scale(const Extent3D& target) { return target.height > FullResolutionMaxHeight ? 2 : 1; }

uint32_t sample_index(uint32_t samples) {
  uint32_t index = 0;
  while ((2u << index) <= samples && index < 3) {
    ++index;
  }
  return index;
}

Pipeline create(rhi::PipelineLayout* layout, const std::string& source, rhi::Format format, uint32_t samples,
                bool multiply) {
  const auto vs = gpu::compile_shader(rhi::ShaderStage::Vertex, source, "vs_main");
  const auto fs = gpu::compile_shader(rhi::ShaderStage::Fragment, source, "fs_main");
  rhi::RenderPipelineDesc desc;
  desc.layout = layout;
  desc.vertex = &vs;
  desc.vertex_entry = "vs_main";
  desc.fragment = &fs;
  desc.fragment_entry = "fs_main";
  desc.color_format = format;
  desc.samples = samples;
  if (multiply) {
    // color *= occlusion; the EFB's alpha stays.
    desc.blend = true;
    desc.blend_color = {rhi::BlendFactor::Dst, rhi::BlendFactor::Zero, rhi::BlendOp::Add};
    desc.blend_alpha = {rhi::BlendFactor::Zero, rhi::BlendFactor::One, rhi::BlendOp::Add};
    desc.color_write = rhi::WriteR | rhi::WriteG | rhi::WriteB;
  }
  return g_device->create_render_pipeline(desc);
}

rhi::RenderPipeline& apply_pipeline(uint32_t samples, bool multisampledDepth) {
  auto& pipeline = g_applyPipelines[sample_index(samples) * 2 + multisampledDepth];
  if (!pipeline) {
    pipeline = create(g_applyLayout.get(), apply_source(multisampledDepth), gpu::g_graphicsConfig.surfaceFormat,
                      samples, true);
  }
  return *pipeline;
}

void ensure_texture(uint32_t width, uint32_t height) {
  if (g_texture && g_textureWidth == width && g_textureHeight == height) {
    return;
  }
  g_view.reset();
  g_texture = g_device->create_texture({
      .width = width,
      .height = height,
      .format = rhi::Format::R8Unorm,
      .usage = rhi::TextureSampled | rhi::TextureColorTarget,
  });
  g_view = g_texture->create_view({});
  g_textureWidth = width;
  g_textureHeight = height;
}
} // namespace

void initialize() {
  g_enabled = g_config.ambientOcclusion;
  if (!g_enabled) {
    return;
  }
  const std::array occlusionEntries{
      rhi::BindingLayout{0, rhi::BindingKind::Texture},
      rhi::BindingLayout{1, rhi::BindingKind::Uniform},
  };
  g_occlusionGroupLayout = g_device->create_bind_group_layout({occlusionEntries.data(), occlusionEntries.size()});
  rhi::BindGroupLayout* occlusionGroups[] = {g_occlusionGroupLayout.get()};
  g_occlusionLayout = g_device->create_pipeline_layout({occlusionGroups, 1});
  const std::array applyEntries{
      rhi::BindingLayout{0, rhi::BindingKind::Texture},
      rhi::BindingLayout{1, rhi::BindingKind::Texture},
      rhi::BindingLayout{2, rhi::BindingKind::Uniform},
  };
  g_applyGroupLayout = g_device->create_bind_group_layout({applyEntries.data(), applyEntries.size()});
  rhi::BindGroupLayout* applyGroups[] = {g_applyGroupLayout.get()};
  g_applyLayout = g_device->create_pipeline_layout({applyGroups, 1});
  const uint32_t samples = gpu::g_graphicsConfig.msaaSamples;
  g_occlusionPipelines[samples > 1] =
      create(g_occlusionLayout.get(), occlusion_source(samples > 1), rhi::Format::R8Unorm, 1, false);
  apply_pipeline(samples, samples > 1);
  if (samples > 1) {
    apply_pipeline(1, true);
  }
}

void shutdown() {
  g_view.reset();
  g_texture.reset();
  g_textureWidth = g_textureHeight = 0;
  for (auto& pipeline : g_occlusionPipelines) {
    pipeline.reset();
  }
  for (auto& pipeline : g_applyPipelines) {
    pipeline.reset();
  }
  g_occlusionLayout.reset();
  g_applyLayout.reset();
  g_occlusionGroupLayout.reset();
  g_applyGroupLayout.reset();
  g_enabled = false;
}

bool enabled() noexcept { return g_enabled; }

Range push_params(const Mat4x4<float>& proj, const Viewport& viewport, const Extent3D& target) {
  // As the viewport command maps depth (reversed Z): window depth runs from
  // 1 - zfar at depth 0 to 1 - znear at depth 1.
  const float minDepth = 1.f - viewport.zfar;
  const float maxDepth = 1.f - viewport.znear;
  const Params params{
      .viewport = {viewport.left, viewport.top, viewport.width, viewport.height},
      .proj = {proj.m0[0], proj.m0[2], proj.m1[1], proj.m1[2]},
      .depth = {proj.m2[2], proj.m2[3], minDepth, maxDepth - minDepth},
      .size = {static_cast<float>(target.width), static_cast<float>(target.height),
               static_cast<float>(occlusion_scale(target)), 0.f},
      .tune = {Radius, Strength, FadeStart, FadeEnd},
  };
  return push_uniform(params);
}

void apply(rhi::CommandList& list, const Target& target) {
  const uint32_t scale = occlusion_scale(target.size);
  ensure_texture((target.size.width + scale - 1) / scale, (target.size.height + scale - 1) / scale);
  const bool multisampled = target.depthSamples > 1;
  auto& occlusionPipeline = g_occlusionPipelines[multisampled];
  if (!occlusionPipeline) {
    occlusionPipeline =
        create(g_occlusionLayout.get(), occlusion_source(multisampled), rhi::Format::R8Unorm, 1, false);
  }

  {
    std::array<rhi::BindingValue, 2> values{};
    values[0].binding = 0;
    values[0].view = target.depth;
    values[1].binding = 1;
    values[1].buffer = g_uniformBuffer;
    values[1].offset = target.params.offset;
    values[1].size = target.params.size;
    const auto group = g_device->create_bind_group({g_occlusionGroupLayout.get(), values.data(), values.size()});
    rhi::RenderPassDesc pass;
    pass.color.target = g_texture.get();
    list.begin_render_pass(pass);
    list.set_pipeline(*occlusionPipeline);
    list.set_bind_group(0, *group, nullptr, 0);
    list.draw(3, 0);
    list.end_render_pass();
  }
  {
    std::array<rhi::BindingValue, 3> values{};
    values[0].binding = 0;
    values[0].view = target.depth;
    values[1].binding = 1;
    values[1].view = g_view.get();
    values[2].binding = 2;
    values[2].buffer = g_uniformBuffer;
    values[2].offset = target.params.offset;
    values[2].size = target.params.size;
    const auto group = g_device->create_bind_group({g_applyGroupLayout.get(), values.data(), values.size()});
    rhi::RenderPassDesc pass;
    pass.color.target = target.color;
    pass.color.resolve = target.resolve;
    list.begin_render_pass(pass);
    list.set_pipeline(apply_pipeline(target.samples, multisampled));
    list.set_bind_group(0, *group, nullptr, 0);
    list.set_scissor(static_cast<uint32_t>(target.rect.x), static_cast<uint32_t>(target.rect.y),
                     static_cast<uint32_t>(target.rect.width), static_cast<uint32_t>(target.rect.height));
    list.draw(3, 0);
    list.end_render_pass();
  }
}

std::vector<std::string> shader_sources() {
  return {occlusion_source(false), occlusion_source(true), apply_source(false), apply_source(true)};
}
} // namespace aurora::gfx::ao
