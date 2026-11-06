#include "ambient_occlusion.hpp"

#include "../gpu/gpu.hpp"
#include "../internal.hpp"

#include <array>
#include <cmath>

// GTAO (Jimenez et al., "Practical Real-Time Strategies for Accurate Indirect
// Occlusion", 2016), after Intel's XeGTAO: per pixel, a few slices around the
// view vector, the horizon on each side found by marching the depth buffer,
// and the visible arc integrated against the normal. A 4x4 noise pattern
// rotates the slices and staggers the steps; a depth-weighted average of each
// 4x4 block cancels it without blurring across edges.
//
// Four passes, the first three at the occlusion's resolution: the view depth,
// read once from the (multisampled) EFB depth; the occlusion; the average,
// packed into the depth; and over the EFB, a depth-weighted 2x2 upsample that
// multiplies the color and resolves it.
//
// The radius is a share of the viewport's height, so the effect looks alike
// in any game's units. It fades out toward the far plane and into the game's
// own fog, which covers what it hides. Thin occluders (hair, poles) count less
// the further in front they are, as XeGTAO's thin occluder compensation; and
// where few neighbours share a pixel's depth (foliage, edges) the occlusion
// gives way rather than leave the noise pattern.

namespace aurora::gfx::ao {
namespace {
using gpu::g_device;

// Above this EFB height the occlusion is computed at half resolution.
constexpr uint32_t FullResolutionMaxHeight = 720;
constexpr float Radius = 0.035f;    // of the viewport's height
constexpr float Strength = 1.0f;    // 0 leaves the image, 1 applies all of it
constexpr float FadeStart = 0.25f;  // of the distance to the far plane
constexpr float FadeEnd = 0.6f;
constexpr float ThinOccluders = 0.7f; // how much depth in front of a pixel stretches a sample's distance
constexpr float Power = 1.5f;          // deepens the occlusion where there is some, as XeGTAO's final power

struct Params {
  float viewport[4]; // left, top, width, height in EFB pixels
  float proj[4];     // GX's p0..p3: x = p0 x + p1 z, y = p2 y + p3 z
  float depth[4];    // p4, p5 (z = p4 z + p5), window depth at depth 0, its span to depth 1
  float size[4];     // EFB width, height, occlusion scale, unused
  float tune[4];     // radius, strength, fade start, fade end
  float fog[4];      // GX fog type (0: none), a, b, c
  float thin[4];     // thin occluder compensation, power, unused
};

constexpr std::string_view ParamsStruct = R"(
struct AoParams {
    float4 viewport;
    float4 proj;
    float4 depth;
    float4 size;
    float4 tune;
    float4 fog;
    float4 thin;
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
float window_depth(float z) {
    return params.depth.z + (params.depth.x + params.depth.y / z) * params.depth.w;
}
float3 view_pos_z(float2 px, float z) {
    float2 ndc = float2((px.x - params.viewport.x) / params.viewport.z * 2.0 - 1.0,
                        1.0 - (px.y - params.viewport.y) / params.viewport.w * 2.0);
    return float3(-z * (ndc.x + params.proj.y) / params.proj.x, -z * (ndc.y + params.proj.w) / params.proj.z, z);
}
float3 view_pos(float2 px, float d) {
    return view_pos_z(px, view_z(d));
}
// How much of the GX fog covers a pixel at this depth, as the game's shaders
// mix it (range adjustment left out).
float fog_amount(float d) {
    uint type = uint(params.fog.x);
    if (type == 0u)
        return 0.0;
    float z = 1.0 - d;
    float base = (type & 8u) != 0u ? params.fog.y * z : params.fog.y / (params.fog.z - z);
    float f = saturate(base - params.fog.w);
    switch (type & 7u) {
    case 2u:
        return f;
    case 4u:
        return 1.0 - exp2(-8.0 * f);
    case 5u:
        return 1.0 - exp2(-8.0 * f * f);
    case 6u:
        return exp2(-8.0 * (1.0 - f));
    case 7u:
        return exp2(-8.0 * (1.0 - f) * (1.0 - f));
    }
    return 0.0;
}
// The viewport's pixels that are in the EFB.
float2 bounds_min() { return max(params.viewport.xy, float2(0.0, 0.0)); }
float2 bounds_max() { return min(params.viewport.xy + params.viewport.zw, params.size.xy); }
int2 clamp_px(int2 p) {
    return clamp(p, int2(bounds_min()), int2(bounds_max()) - 1);
}
)";

// The view depth of each occlusion pixel, read once from the EFB's depth (for
// a multisampled one, its first sample); 0 where there is no geometry.
constexpr std::string_view LinearizeShader = R"(
float fs_main(VertexOutput input) : SV_Target {
    int scale = int(params.size.z);
    int2 c = int2(input.pos.xy) * scale + scale / 2;
    if (any(float2(c) < bounds_min()) || any(float2(c) >= bounds_max()))
        return 0.0;
    float d = load_depth(c);
    return is_geometry(d) ? view_z(d) : 0.0;
}
)";

constexpr std::string_view OcclusionShader = R"(
static const float Pi = 3.14159265;
static const uint Slices = 2;
static const uint Steps = 3;
static const float Bayer[16] = {0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5};

float fast_acos(float x) {
    float a = abs(x);
    float r = (-0.156583 * a + Pi * 0.5) * sqrt(1.0 - a);
    return x >= 0.0 ? r : Pi - r;
}

// The view position at an occlusion pixel, from its linear depth.
float3 tap(int2 aoPx, float z) {
    int scale = int(params.size.z);
    return view_pos_z(float2(aoPx * scale + scale / 2) + 0.5, z);
}

float fs_main(VertexOutput input) : SV_Target {
    int2 aoPx = int2(input.pos.xy);
    int scale = int(params.size.z);
    int2 c = aoPx * scale + scale / 2;
    float2 vpMin = bounds_min();
    float2 vpMax = bounds_max();
    float z0 = linear_z.Load(int3(aoPx, 0));
    if (!(z0 < 0.0))
        return 1.0;
    float2 center = float2(c) + 0.5;
    float3 P = view_pos_z(center, z0);
    // The occlusion pixels that cover the viewport.
    int2 aoMin = int2(ceil((vpMin - float(scale / 2)) / float(scale)));
    int2 aoMax = int2(ceil((vpMax - float(scale / 2)) / float(scale))) - 1;

    // The normal from the depth: on each axis the side nearer in depth, so
    // edges don't bend it.
    int2 il = clamp(aoPx - int2(1, 0), aoMin, aoMax), ir = clamp(aoPx + int2(1, 0), aoMin, aoMax);
    int2 iu = clamp(aoPx - int2(0, 1), aoMin, aoMax), id = clamp(aoPx + int2(0, 1), aoMin, aoMax);
    float3 pl = tap(il, linear_z.Load(int3(il, 0)));
    float3 pr = tap(ir, linear_z.Load(int3(ir, 0)));
    float3 pu = tap(iu, linear_z.Load(int3(iu, 0)));
    float3 pd = tap(id, linear_z.Load(int3(id, 0)));
    // A side on the sky doesn't count.
    bool vl = pl.z < 0.0, vr = pr.z < 0.0, vu = pu.z < 0.0, vd = pd.z < 0.0;
    float3 dx = vr && (!vl || abs(pr.z - P.z) < abs(P.z - pl.z)) ? pr - P : (vl ? P - pl : float3(0, 0, 0));
    float3 dy = vd && (!vu || abs(pd.z - P.z) < abs(P.z - pu.z)) ? pd - P : (vu ? P - pu : float3(0, 0, 0));
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
            // At least a texel a step: one of its own would give no direction.
            float offset = max(t * t * radiusPx, (float(j) + 1.0) * float(scale));
            [unroll]
            for (uint side = 0; side < 2; ++side) {
                float2 spx = center + (side == 0 ? omega : -omega) * offset;
                if (any(spx < vpMin) || any(spx >= vpMax))
                    continue;
                int2 si = int2(spx) / scale;
                if (all(si == aoPx))
                    continue;
                float sz = linear_z.Load(int3(si, 0));
                if (!(sz < 0.0))
                    continue;
                float3 delta = tap(si, sz) - P;
                float dist = length(delta);
                float cosH = dot(delta / max(dist, 1e-6), V);
                // Depth toward the camera stretches the distance: a thin
                // strand in front shadows less than a wall would.
                float stretched = length(float3(delta.xy, delta.z * (1.0 + params.thin.x)));
                float weight = saturate(stretched * falloffMul + falloffAdd);
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
    visibility = pow(visibility, params.thin.y);

    // Toward the far plane, where the radius spans a lot of the world, it fades.
    float farZ = -params.depth.y / params.depth.x;
    float fade = 1.0 - smoothstep(params.tune.z, params.tune.w, P.z / farZ);
    fade *= 1.0 - fog_amount(window_depth(z0));
    return lerp(1.0, visibility, params.tune.y * fade);
}
)";

// At the occlusion's resolution: the 4x4 average that removes the noise
// pattern, weighted by depth to stay on each pixel's surface. It packs the
// result in the low byte of the view depth, which keeps 15 bits of mantissa.
constexpr std::string_view BlurShader = R"(
float fs_main(VertexOutput input) : SV_Target {
    int2 p = int2(input.pos.xy);
    float z0 = linear_z.Load(int3(p, 0));
    if (!(z0 < 0.0))
        return 0.0;
    uint width, height;
    linear_z.GetDimensions(width, height);
    int2 last = int2(width, height) - 1;
    float sum = 0.0, weights = 0.0;
    [unroll]
    for (int y = -1; y < 3; ++y) {
        [unroll]
        for (int x = -1; x < 3; ++x) {
            int2 t = clamp(p + int2(x, y), int2(0, 0), last);
            float tz = linear_z.Load(int3(t, 0));
            float w = tz < 0.0 ? exp2(-abs(tz - z0) / (abs(z0) * 0.02)) : 0.0;
            sum += occlusion.Load(int3(t, 0)) * w;
            weights += w;
        }
    }
    // Fewer than half the taps on this surface: trust it that much less.
    float ao = lerp(1.0, sum / weights, saturate(weights / 8.0));
    return asfloat((asuint(z0) & 0xFFFFFF00u) | uint(round(saturate(ao) * 255.0)));
}
)";

// At the EFB's resolution: the nearest 2x2 of the blurred occlusion, weighted
// bilinearly and by depth.
constexpr std::string_view ApplyShader = R"(
float4 fs_main(VertexOutput input) : SV_Target {
    int2 c = int2(input.pos.xy);
    float d0 = load_depth(c);
    if (!is_geometry(d0))
        return float4(1.0, 1.0, 1.0, 1.0);
    float z0 = view_z(d0);
    int scale = int(params.size.z);
    uint width, height;
    blurred.GetDimensions(width, height);
    int2 last = int2(width, height) - 1;
    float2 f = (input.pos.xy - (float(scale / 2) + 0.5)) / float(scale);
    float2 cell = floor(f);
    float2 frac2 = f - cell;
    float sum = 0.0, weights = 0.0, nearestAo = 1.0, nearestDz = 3.4e38;
    [unroll]
    for (int y = 0; y < 2; ++y) {
        [unroll]
        for (int x = 0; x < 2; ++x) {
            int2 t = clamp(int2(cell) + int2(x, y), int2(0, 0), last);
            uint packed = asuint(blurred.Load(int3(t, 0)));
            float tz = asfloat(packed & 0xFFFFFF00u);
            if (!(tz < 0.0))
                continue;
            float ao = float(packed & 0xFFu) / 255.0;
            float dz = abs(tz - z0);
            float2 b = lerp(1.0 - frac2, frac2, float2(x, y));
            float w = b.x * b.y * exp2(-dz / (abs(z0) * 0.02));
            sum += ao * w;
            weights += w;
            if (dz < nearestDz) {
                nearestDz = dz;
                nearestAo = ao;
            }
        }
    }
    // No texel on this surface (a thin edge): the nearest in depth.
    float ao = weights > 1e-3 ? sum / weights : nearestAo;
    return float4(ao, ao, ao, 1.0);
}
)";

std::string depth_access(bool multisampled) {
  return multisampled ? "Texture2DMS<float> depth_tex : register(t0, space0);\n"
                        "float load_depth(int2 p) { return depth_tex.Load(p, 0); }\n"
                      : "Texture2D<float> depth_tex : register(t0, space0);\n"
                        "float load_depth(int2 p) { return depth_tex.Load(int3(p, 0)); }\n";
}

std::string linearize_source(bool multisampled) {
  return std::string{ParamsStruct} + depth_access(multisampled) +
         "ConstantBuffer<AoParams> params : register(b1, space0);\n" + std::string{Common} +
         std::string{LinearizeShader};
}

std::string occlusion_source() {
  return std::string{ParamsStruct} + "Texture2D<float> linear_z : register(t0, space0);\n" +
         "ConstantBuffer<AoParams> params : register(b1, space0);\n" + std::string{Common} +
         std::string{OcclusionShader};
}

std::string blur_source() {
  return std::string{ParamsStruct} + "Texture2D<float> linear_z : register(t0, space0);\n" +
         "Texture2D<float> occlusion : register(t1, space0);\n" +
         "ConstantBuffer<AoParams> params : register(b2, space0);\n" + std::string{Common} + std::string{BlurShader};
}

std::string apply_source(bool multisampled) {
  return std::string{ParamsStruct} + depth_access(multisampled) +
         "Texture2D<float> blurred : register(t1, space0);\n" +
         "ConstantBuffer<AoParams> params : register(b2, space0);\n" + std::string{Common} + std::string{ApplyShader};
}

using Pipeline = std::unique_ptr<rhi::RenderPipeline>;

bool g_enabled = false;
std::unique_ptr<rhi::BindGroupLayout> g_occlusionGroupLayout;
std::unique_ptr<rhi::BindGroupLayout> g_applyGroupLayout;
std::unique_ptr<rhi::PipelineLayout> g_occlusionLayout;
std::unique_ptr<rhi::PipelineLayout> g_applyLayout;
std::array<Pipeline, 2> g_linearizePipelines; // by whether the depth is multisampled
Pipeline g_occlusionPipeline;
Pipeline g_blurPipeline;
std::array<Pipeline, 8> g_applyPipelines; // by log2 of the color's samples, then multisampled depth
std::shared_ptr<rhi::Texture> g_texture;  // the occlusion, R8
std::unique_ptr<rhi::TextureView> g_view;
std::shared_ptr<rhi::Texture> g_linear; // the view depth, R32Float
std::unique_ptr<rhi::TextureView> g_linearView;
std::shared_ptr<rhi::Texture> g_blurred; // the blurred occlusion in the view depth, R32Float
std::unique_ptr<rhi::TextureView> g_blurredView;
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
  g_linearView.reset();
  g_blurredView.reset();
  g_texture = g_device->create_texture({
      .width = width,
      .height = height,
      .format = rhi::Format::R8Unorm,
      .usage = rhi::TextureSampled | rhi::TextureColorTarget,
  });
  g_view = g_texture->create_view({});
  g_linear = g_device->create_texture({
      .width = width,
      .height = height,
      .format = rhi::Format::R32Float,
      .usage = rhi::TextureSampled | rhi::TextureColorTarget,
  });
  g_linearView = g_linear->create_view({});
  g_blurred = g_device->create_texture({
      .width = width,
      .height = height,
      .format = rhi::Format::R32Float,
      .usage = rhi::TextureSampled | rhi::TextureColorTarget,
  });
  g_blurredView = g_blurred->create_view({});
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
  g_linearizePipelines[samples > 1] =
      create(g_occlusionLayout.get(), linearize_source(samples > 1), rhi::Format::R32Float, 1, false);
  g_occlusionPipeline = create(g_occlusionLayout.get(), occlusion_source(), rhi::Format::R8Unorm, 1, false);
  g_blurPipeline = create(g_applyLayout.get(), blur_source(), rhi::Format::R32Float, 1, false);
  apply_pipeline(samples, samples > 1);
  if (samples > 1) {
    apply_pipeline(1, true);
  }
}

void shutdown() {
  g_view.reset();
  g_texture.reset();
  g_linearView.reset();
  g_linear.reset();
  g_blurredView.reset();
  g_blurred.reset();
  g_textureWidth = g_textureHeight = 0;
  for (auto& pipeline : g_linearizePipelines) {
    pipeline.reset();
  }
  g_occlusionPipeline.reset();
  g_blurPipeline.reset();
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

Range push_params(const Mat4x4<float>& proj, const Viewport& viewport, const Extent3D& target, const Fog& fog) {
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
      .fog = {static_cast<float>(fog.type), fog.a, fog.b, fog.c},
      .thin = {ThinOccluders, Power, 0.f, 0.f},
  };
  return push_uniform(params);
}

void apply(rhi::CommandList& list, const Target& target) {
  const uint32_t scale = occlusion_scale(target.size);
  ensure_texture((target.size.width + scale - 1) / scale, (target.size.height + scale - 1) / scale);
  const bool multisampled = target.depthSamples > 1;
  auto& linearizePipeline = g_linearizePipelines[multisampled];
  if (!linearizePipeline) {
    linearizePipeline =
        create(g_occlusionLayout.get(), linearize_source(multisampled), rhi::Format::R32Float, 1, false);
  }

  // A full-screen pass reading one or two textures.
  const auto pass = [&](rhi::RenderPipeline& pipeline, rhi::TextureView* first, rhi::TextureView* second,
                        rhi::Texture* output, rhi::Texture* resolve) {
    std::array<rhi::BindingValue, 3> values{};
    values[0].binding = 0;
    values[0].view = first;
    uint32_t count = 1;
    if (second != nullptr) {
      values[count].binding = 1;
      values[count++].view = second;
    }
    values[count].binding = second != nullptr ? 2 : 1;
    values[count].buffer = g_uniformBuffer;
    values[count].offset = target.params.offset;
    values[count++].size = target.params.size;
    const auto group = g_device->create_bind_group(
        {second != nullptr ? g_applyGroupLayout.get() : g_occlusionGroupLayout.get(), values.data(), count});
    rhi::RenderPassDesc desc;
    desc.color.target = output;
    desc.color.resolve = resolve;
    list.begin_render_pass(desc);
    list.set_pipeline(pipeline);
    list.set_bind_group(0, *group, nullptr, 0);
    if (output == target.color) {
      list.set_scissor(static_cast<uint32_t>(target.rect.x), static_cast<uint32_t>(target.rect.y),
                       static_cast<uint32_t>(target.rect.width), static_cast<uint32_t>(target.rect.height));
    }
    list.draw(3, 0);
    list.end_render_pass();
  };
  pass(*linearizePipeline, target.depth, nullptr, g_linear.get(), nullptr);
  pass(*g_occlusionPipeline, g_linearView.get(), nullptr, g_texture.get(), nullptr);
  pass(*g_blurPipeline, g_linearView.get(), g_view.get(), g_blurred.get(), nullptr);
  pass(apply_pipeline(target.samples, multisampled), target.depth, g_blurredView.get(), target.color,
       target.resolve);
}

std::vector<std::string> shader_sources() {
  return {linearize_source(false), linearize_source(true), occlusion_source(), blur_source(), apply_source(false),
          apply_source(true)};
}
} // namespace aurora::gfx::ao
