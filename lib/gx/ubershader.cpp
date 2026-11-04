// One shader that draws any GX state, for draws whose own pipeline is still
// compiling: the same vertex fetch, texgen, lighting, TEV, fog and alpha compare
// as build_shader_source generates, each reading what it has to do from a table
// of the draw's configuration in the storage buffer. It must draw the very
// pixels the generated shader does; a change to either goes in both.
//
// A pipeline still depends on the fixed-function state (blend, depth, cull, the
// samples, alpha to coverage), so there is one for each state a draw has used,
// sharing a shader for each way destination alpha reaches the blend.

#include "pipeline.hpp"

#include "../webgpu/gpu.hpp"
#include "shader_info.hpp"

#include <absl/container/flat_hash_map.h>
#include <array>
#include <cstddef>
#include <mutex>
#include <string>

namespace aurora::gx {
namespace {
struct UberStage {
  u32 colorArgs; // a, b, c, d: a byte each
  u32 alphaArgs;
  u32 colorOp; // op, bias, scale, output register: a byte each
  u32 alphaOp;
  u32 konst;   // color selection, alpha selection, rasterized source, alpha bump mask
  u32 tex;     // texcoord, texture map, swizzle of the rasterized color, of the texture
  u32 indirect; // see build_stage
  u32 flags;
};
enum UberStageFlag : u32 {
  ColorClamp = 1u << 0,
  AlphaClamp = 1u << 1,
  ColorOverflowA = 1u << 2,
  ColorOverflowB = 1u << 3,
  ColorOverflowC = 1u << 4,
  AlphaOverflowA = 1u << 5,
  AlphaOverflowB = 1u << 6,
  AlphaOverflowC = 1u << 7,
  NeedsTevCoord = 1u << 8,
  NeedsSample = 1u << 9,
  HasBaseTexture = 1u << 10,
  HasBaseCoord = 1u << 11,
  SimpleCoords = 1u << 12,
  AddPrevious = 1u << 13,
  WritesCoord = 1u << 14,
  AlphaComparesColor = 1u << 15,
};
// What a stage's rasterized input is.
enum UberRasterized : u32 { RasZero, RasChannel0, RasChannel1, RasBump, RasBumpNormalized };

struct UberConfig {
  u32 tevStages;
  u32 indStages;
  u32 vtxStride;
  u32 lineMode;
  u32 fogType;
  u32 fogRange;
  u32 loadsTevReg;
  u32 sampledTexCoords;
  u32 sampledChannels;
  u32 usedIndStages;
  u32 alphaCompare;
  u32 alphaComp0;
  u32 alphaRef0;
  u32 alphaOp;
  u32 alphaComp1;
  u32 alphaRef1;
  u32 alphaToCoverage;
  f32 coverageThreshold;
  u32 prevOverflow; // 1 color, 2 alpha
  u32 lastOutputs;  // the last stage's color and alpha registers: a byte each
  f32 lineWidth;
  f32 lineTexOffset;
  u32 lineTexMask;
  u32 pad;
  std::array<u32, MaxColorChannels> channels; // material, ambient, diffuse, attenuation, lit: a nibble each
  std::array<std::array<u32, 2>, MaxTexCoord> tcgs;
  std::array<std::array<u32, 2>, MaxIndStages> ind;
  std::array<std::array<u32, 2>, MaxVtxAttr> attrs;
  std::array<UberStage, MaxTevStages> stages;
};
static_assert(sizeof(UberConfig) % 4 == 0);

u32 swizzle(const TevSwap& swap) {
  return static_cast<u32>(swap.red) | static_cast<u32>(swap.green) << 2 | static_cast<u32>(swap.blue) << 4 |
         static_cast<u32>(swap.alpha) << 6;
}

u32 bump_mask(GXIndTexFormat format) {
  switch (format) {
  case GX_ITF_5:
    return 0xE0;
  case GX_ITF_4:
    return 0xF0;
  default:
    return 0xF8;
  }
}

UberStage build_stage(const ShaderConfig& config, const TevStage& stage, const ShaderInfo& info,
                      std::array<bool, MaxTevRegs>& colorNormalized, std::array<bool, MaxTevRegs>& alphaNormalized) {
  const auto color_overflow = [&](GXTevColorArg arg, u32 flag) {
    return tev_color_arg_is_normalized(arg, colorNormalized, alphaNormalized) ? 0u : flag;
  };
  const auto alpha_overflow = [&](GXTevAlphaArg arg, u32 flag) {
    return tev_alpha_arg_is_normalized(arg, alphaNormalized) ? 0u : flag;
  };

  const bool needsIndirectCoord = stage.indTexMtxId != GX_ITM_OFF;
  const bool hasIndirectStage = stage.indTexStage < config.numIndStages;
  const bool needsTevCoord =
      needsIndirectCoord || stage.indTexWrapS != GX_ITW_OFF || stage.indTexWrapT != GX_ITW_OFF || stage.indTexAddPrev;
  const bool hasBaseTexture = stage.texMapId != GX_TEXMAP_NULL;
  const bool hasBaseCoord = stage.texCoordId != GX_TEXCOORD_NULL && hasBaseTexture;

  // 0 no indirect offset, 1 from a static matrix, 2 and 3 scaled by S and T.
  u32 matrixKind = 0;
  u32 matrixIndex = 0;
  if (needsIndirectCoord && hasIndirectStage) {
    if (stage.indTexMtxId >= GX_ITM_0 && stage.indTexMtxId <= GX_ITM_2) {
      matrixKind = 1;
      matrixIndex = stage.indTexMtxId - GX_ITM_0;
    } else if (stage.indTexMtxId >= GX_ITM_S0 && stage.indTexMtxId <= GX_ITM_S2 && hasBaseCoord) {
      matrixKind = 2;
      matrixIndex = stage.indTexMtxId - GX_ITM_S0;
    } else if (stage.indTexMtxId >= GX_ITM_T0 && stage.indTexMtxId <= GX_ITM_T2 && hasBaseCoord) {
      matrixKind = 3;
      matrixIndex = stage.indTexMtxId - GX_ITM_T0;
    }
  }

  UberRasterized rasterized = RasZero;
  if (stage.channelId == GX_ALPHA_BUMP || stage.channelId == GX_ALPHA_BUMPN) {
    if (hasIndirectStage && stage.indTexAlphaSel != GX_ITBA_OFF) {
      rasterized = stage.channelId == GX_ALPHA_BUMP ? RasBump : RasBumpNormalized;
    }
  } else if (stage.channelId != GX_COLOR_ZERO && stage.channelId != GX_COLOR_NULL) {
    rasterized = color_channel(stage.channelId) == 0 ? RasChannel0 : RasChannel1;
  }

  u32 flags = 0;
  flags |= stage.colorOp.clamp ? ColorClamp : 0u;
  flags |= stage.alphaOp.clamp ? AlphaClamp : 0u;
  flags |= color_overflow(stage.colorPass.a, ColorOverflowA);
  flags |= color_overflow(stage.colorPass.b, ColorOverflowB);
  flags |= color_overflow(stage.colorPass.c, ColorOverflowC);
  flags |= alpha_overflow(stage.alphaPass.a, AlphaOverflowA);
  flags |= alpha_overflow(stage.alphaPass.b, AlphaOverflowB);
  flags |= alpha_overflow(stage.alphaPass.c, AlphaOverflowC);
  flags |= needsTevCoord ? NeedsTevCoord : 0u;
  flags |= uses_texture_sample(stage) ? NeedsSample : 0u;
  flags |= hasBaseTexture ? HasBaseTexture : 0u;
  flags |= hasBaseCoord ? HasBaseCoord : 0u;
  flags |= stage.indTexMtxId == GX_ITM_OFF && !stage.indTexAddPrev ? SimpleCoords : 0u;
  flags |= stage.indTexAddPrev ? AddPrevious : 0u;
  flags |= info.usedIndStages.any() && (hasBaseCoord || matrixKind != 0) ? WritesCoord : 0u;
  flags |= tev_alpha_compares_color(stage.alphaOp.op) ? AlphaComparesColor : 0u;

  const auto bytes = [](u32 a, u32 b, u32 c, u32 d) { return (a & 0xFF) | (b & 0xFF) << 8 | (c & 0xFF) << 16 | d << 24; };
  const UberStage result{
      .colorArgs = bytes(stage.colorPass.a, stage.colorPass.b, stage.colorPass.c, stage.colorPass.d),
      .alphaArgs = bytes(stage.alphaPass.a, stage.alphaPass.b, stage.alphaPass.c, stage.alphaPass.d),
      .colorOp = bytes(stage.colorOp.op, stage.colorOp.bias, stage.colorOp.scale, stage.colorOp.outReg),
      .alphaOp = bytes(stage.alphaOp.op, stage.alphaOp.bias, stage.alphaOp.scale, stage.alphaOp.outReg),
      .konst = bytes(stage.kcSel, stage.kaSel, rasterized, bump_mask(stage.indTexFormat)),
      .tex = bytes(stage.texCoordId, stage.texMapId, swizzle(config.tevSwapTable[stage.tevSwapRas]),
                   swizzle(config.tevSwapTable[stage.tevSwapTex])),
      // A nibble each: stage, format, bias, alpha selection, matrix kind, matrix, wrap S, wrap T.
      .indirect = static_cast<u32>(stage.indTexStage & 0xF) | static_cast<u32>(stage.indTexFormat) << 4 |
                  static_cast<u32>(stage.indTexBiasSel) << 8 | static_cast<u32>(stage.indTexAlphaSel) << 12 |
                  matrixKind << 16 | matrixIndex << 20 | static_cast<u32>(stage.indTexWrapS) << 24 |
                  static_cast<u32>(stage.indTexWrapT) << 28,
      .flags = flags,
  };
  colorNormalized[stage.colorOp.outReg] = stage.colorOp.clamp;
  alphaNormalized[stage.alphaOp.outReg] = stage.alphaOp.clamp;
  return result;
}

UberConfig build_config(const ShaderConfig& config, const ShaderInfo& info) {
  UberConfig result{};
  result.tevStages = config.tevStageCount;
  result.indStages = config.numIndStages;
  result.vtxStride = config.vtxStride;
  result.lineMode = config.lineMode;
  result.fogType = config.fogType;
  result.fogRange = config.fogRangeEnabled;
  result.loadsTevReg = static_cast<u32>(info.loadsTevReg.to_ulong());
  result.sampledTexCoords = static_cast<u32>(info.sampledTexCoords.to_ulong());
  result.sampledChannels = static_cast<u32>(info.sampledColorChannels.to_ulong());
  result.usedIndStages = static_cast<u32>(info.usedIndStages.to_ulong());
  result.alphaCompare = static_cast<bool>(config.alphaCompare);
  result.alphaComp0 = config.alphaCompare.comp0;
  result.alphaRef0 = config.alphaCompare.ref0;
  result.alphaOp = config.alphaCompare.op;
  result.alphaComp1 = config.alphaCompare.comp1;
  result.alphaRef1 = config.alphaCompare.ref1;
  result.alphaToCoverage = config.alphaToCoverage;
  result.coverageThreshold = config.alphaToCoverage ? alpha_cutout_threshold(config.alphaCompare) : 0.f;

  std::array<bool, MaxTevRegs> colorNormalized{};
  std::array<bool, MaxTevRegs> alphaNormalized{};
  for (u32 i = 0; i < config.tevStageCount; ++i) {
    result.stages[i] = build_stage(config, config.tevStages[i], info, colorNormalized, alphaNormalized);
  }
  // No stages, which a stream with an incomplete state can reach, reads stage 0.
  const auto& last = config.tevStages[config.tevStageCount != 0 ? config.tevStageCount - 1 : 0];
  const bool prevColorNormalized = config.tevStageCount != 0 && colorNormalized[last.colorOp.outReg];
  const bool prevAlphaNormalized = config.tevStageCount != 0 && alphaNormalized[last.alphaOp.outReg];
  result.prevOverflow = (prevColorNormalized ? 0u : 1u) | (prevAlphaNormalized ? 0u : 2u);
  result.lastOutputs = static_cast<u32>(last.colorOp.outReg) | static_cast<u32>(last.alphaOp.outReg) << 8;

  for (u32 i = 0; i < MaxColorChannels; ++i) {
    const auto& cc = config.colorChannels[i];
    result.channels[i] = static_cast<u32>(cc.matSrc) | static_cast<u32>(cc.ambSrc) << 4 |
                         static_cast<u32>(cc.diffFn) << 8 | static_cast<u32>(cc.attnFn) << 12 |
                         static_cast<u32>(cc.lightingEnabled) << 16;
  }
  for (u32 i = 0; i < MaxTexCoord; ++i) {
    const auto& tcg = config.tcgs[i];
    const u32 matrix = tcg.mtx == GX_IDENTITY ? 0xFF : tcg.mtx / 3;
    const u32 post = tcg.postMtx == GX_PTIDENTITY ? 0xFF : (tcg.postMtx - GX_PTTEXMTX0) / 3;
    result.tcgs[i] = {
        static_cast<u32>(tcg.type) | static_cast<u32>(tcg.src) << 8 | static_cast<u32>(tcg.embossSrc) << 16 |
            static_cast<u32>(tcg.normalize) << 24 | static_cast<u32>(info.indexAttr.test(GX_VA_TEX0MTXIDX + i)) << 25,
        matrix | post << 8,
    };
  }
  for (u32 i = 0; i < config.numIndStages; ++i) {
    const auto& stage = config.indStages[i];
    result.ind[i] = {
        static_cast<u32>(stage.texCoordId) | static_cast<u32>(stage.texMapId) << 8,
        static_cast<u32>(stage.scaleS) | static_cast<u32>(stage.scaleT) << 8,
    };
  }
  for (u32 i = 0; i < MaxVtxAttr; ++i) {
    const auto& attr = config.attrs[i];
    result.attrs[i] = {
        static_cast<u32>(attr.attrType) | static_cast<u32>(attr.cnt) << 8 | static_cast<u32>(attr.compType) << 16 |
            static_cast<u32>(attr.offset) << 24,
        static_cast<u32>(attr.stride) | static_cast<u32>(attr.frac) << 8 | static_cast<u32>(attr.le) << 16 |
            static_cast<u32>(attr.nbt3) << 24,
    };
  }
  return result;
}

constexpr std::string_view UberSource = R"""(
struct Light {
    pos: vec3f,
    dir: vec3f,
    color: vec4f,
    cos_att: vec3f,
    dist_att: vec3f,
};
struct Fog {
    color: vec4f,
    a: f32,
    b: f32,
    c: f32,
    pad: f32,
};
struct Uniform {
    vtx_start: u32,
    current_pnmtx: u32,
    render_viewport_size: vec2f,
    logical_viewport_size: vec2f,
    fog_range_base: u32,
    config_base: u32,
    array_start: array<u32, 12>,
    proj: mat4x4f,
    postex_mtx: array<mat3x4f, 20>,
    nrm_mtx: array<mat3x4f, 10>,
    tev_regs: array<vec4f, 4>,
    lights: array<Light, 8>,
    light_state: vec4u,
    cc_amb: array<vec4f, 4>,
    cc_mat: array<vec4f, 4>,
    kcolors: array<vec4f, 4>,
    postmtx: array<mat3x4f, 20>,
    fog: Fog,
    texcoord_scale: array<vec4f, 8>,
    ind_mtx: array<mat2x4f, 3>,
    tex_size_bias: array<vec4f, 8>,
};
@group(0) @binding(0)
var<storage, read> vbuf: array<u32>;
@group(0) @binding(1)
var<storage, read> abuf: array<u32>;
@group(1) @binding(0)
var<uniform> ubuf: Uniform;
@group(2) @binding(0) var tex0: texture_2d<f32>;
@group(2) @binding(1) var tex0_samp: sampler;
@group(2) @binding(2) var tex1: texture_2d<f32>;
@group(2) @binding(3) var tex1_samp: sampler;
@group(2) @binding(4) var tex2: texture_2d<f32>;
@group(2) @binding(5) var tex2_samp: sampler;
@group(2) @binding(6) var tex3: texture_2d<f32>;
@group(2) @binding(7) var tex3_samp: sampler;
@group(2) @binding(8) var tex4: texture_2d<f32>;
@group(2) @binding(9) var tex4_samp: sampler;
@group(2) @binding(10) var tex5: texture_2d<f32>;
@group(2) @binding(11) var tex5_samp: sampler;
@group(2) @binding(12) var tex6: texture_2d<f32>;
@group(2) @binding(13) var tex6_samp: sampler;
@group(2) @binding(14) var tex7: texture_2d<f32>;
@group(2) @binding(15) var tex7_samp: sampler;

fn cfg(word: u32) -> u32 {
  return abuf[ubuf.config_base + word];
}
fn cfg_f32(word: u32) -> f32 {
  return bitcast<f32>(cfg(word));
}
fn byte_of(value: u32, index: u32) -> u32 {
  return (value >> (index * 8u)) & 0xFFu;
}
fn nibble_of(value: u32, index: u32) -> u32 {
  return (value >> (index * 4u)) & 0xFu;
}

// ---------------------------------------------------------------- vertices

const VA_PNMTXIDX = 0u;
const VA_TEX0MTXIDX = 1u;
const VA_POS = 9u;
const VA_NRM = 10u;
const VA_CLR0 = 11u;
const VA_TEX0 = 13u;
const AT_NONE = 0u;
const AT_INDEX8 = 2u;
const AT_INDEX16 = 3u;

var<private> vtx_start_of_draw: u32;

struct AttrAddress {
  offs: u32,
  in_arrays: bool,
  le: bool,
};

fn attr_type(attr: u32) -> u32 {
  return cfg(CFG_ATTRS + attr * 2u) & 0xFFu;
}

fn attr_address(attr: u32, vidx: u32, dl_extra: u32, within: u32) -> AttrAddress {
  let w0 = cfg(CFG_ATTRS + attr * 2u);
  let w1 = cfg(CFG_ATTRS + attr * 2u + 1u);
  let type_ = byte_of(w0, 0u);
  let entry = vtx_start_of_draw + vidx * cfg(CFG_VTX_STRIDE) + byte_of(w0, 3u) + dl_extra;
  let stride = byte_of(w1, 0u);
  let le = byte_of(w1, 2u) != 0u;
  if (type_ == AT_INDEX8) {
    return AttrAddress(ubuf.array_start[attr - VA_POS] + raw_fetch_u8_1(&vbuf, entry) * stride + within, true, le);
  }
  if (type_ == AT_INDEX16) {
    return AttrAddress(ubuf.array_start[attr - VA_POS] + raw_fetch_u16_1(&vbuf, entry, false) * stride + within, true, le);
  }
  return AttrAddress(entry + within, false, false);
}

fn fetch_component(p: ptr<storage, array<u32>>, comp: u32, offs: u32, frac: u32, le: bool) -> f32 {
  switch comp {
    case 0u: { return fetch_u8_1(p, offs, frac, le); }
    case 1u: { return fetch_s8_1(p, offs, frac, le); }
    case 2u: { return fetch_u16_1(p, offs, frac, le); }
    case 3u: { return fetch_s16_1(p, offs, frac, le); }
    default: { return fetch_f32_1(p, offs, le); }
  }
}

fn component_size(comp: u32) -> u32 {
  if (comp <= 1u) { return 1u; }
  if (comp <= 3u) { return 2u; }
  return 4u;
}

fn fetch_vector(p: ptr<storage, array<u32>>, comp: u32, count: u32, offs: u32, frac: u32, le: bool) -> vec3f {
  var v = vec3f(0.0);
  let size = component_size(comp);
  for (var i = 0u; i < min(count, 3u); i++) {
    v[i] = fetch_component(p, comp, offs + i * size, frac, le);
  }
  return v;
}

fn load_vector(attr: u32, count: u32, vidx: u32, dl_extra: u32, within: u32) -> vec3f {
  let w0 = cfg(CFG_ATTRS + attr * 2u);
  let frac = byte_of(cfg(CFG_ATTRS + attr * 2u + 1u), 1u);
  let comp = byte_of(w0, 2u);
  let a = attr_address(attr, vidx, dl_extra, within);
  if (a.in_arrays) {
    return fetch_vector(&abuf, comp, count, a.offs, frac, a.le);
  }
  return fetch_vector(&vbuf, comp, count, a.offs, frac, a.le);
}

fn load_attr(attr: u32, vidx: u32) -> vec3f {
  return load_vector(attr, byte_of(cfg(CFG_ATTRS + attr * 2u), 1u), vidx, 0u, 0u);
}

fn load_pos(vidx: u32) -> vec3f {
  return load_attr(VA_POS, vidx);
}

fn load_nrm(vidx: u32) -> vec3f {
  if (attr_type(VA_NRM) == AT_NONE) {
    return vec3f(1.0, 0.0, 0.0);
  }
  return load_attr(VA_NRM, vidx);
}

// The binormal (1) or tangent (2) of a normal with all three.
fn load_nbt_slice(slice: u32, vidx: u32) -> vec3f {
  let w0 = cfg(CFG_ATTRS + VA_NRM * 2u);
  let w1 = cfg(CFG_ATTRS + VA_NRM * 2u + 1u);
  let type_ = byte_of(w0, 0u);
  var dl_extra = 0u;
  if (byte_of(w1, 3u) != 0u) {
    if (type_ == AT_INDEX8) {
      dl_extra = slice;
    } else if (type_ == AT_INDEX16) {
      dl_extra = slice * 2u;
    }
  }
  return load_vector(VA_NRM, 3u, vidx, dl_extra, slice * 3u * component_size(byte_of(w0, 2u)));
}

fn load_index_byte(attr: u32, vidx: u32) -> u32 {
  let a = attr_address(attr, vidx, 0u, 0u);
  if (a.in_arrays) {
    return raw_fetch_u8_1(&abuf, a.offs);
  }
  return raw_fetch_u8_1(&vbuf, a.offs);
}

fn load_pnmtx(vidx: u32) -> u32 {
  if (attr_type(VA_PNMTXIDX) == AT_NONE) {
    return ubuf.current_pnmtx;
  }
  return load_index_byte(VA_PNMTXIDX, vidx) / 3u;
}

fn fetch_color(p: ptr<storage, array<u32>>, comp: u32, offs: u32, le: bool) -> vec4f {
  switch comp {
    case 0u: { return fetch_rgb565(p, offs, le); }
    case 1u: { return fetch_rgb8(p, offs, le); }
    case 2u: { return fetch_rgbx8(p, offs, le); }
    case 3u: { return fetch_rgba4(p, offs, le); }
    case 4u: { return fetch_rgba6(p, offs, le); }
    default: { return fetch_rgba8(p, offs, le); }
  }
}

fn load_color(channel: u32, vidx: u32) -> vec4f {
  let attr = VA_CLR0 + channel;
  if (attr_type(attr) == AT_NONE) {
    return vec4f(0.0);
  }
  let comp = byte_of(cfg(CFG_ATTRS + attr * 2u), 2u);
  let a = attr_address(attr, vidx, 0u, 0u);
  if (a.in_arrays) {
    return fetch_color(&abuf, comp, a.offs, a.le);
  }
  return fetch_color(&vbuf, comp, a.offs, a.le);
}

// ---------------------------------------------------------------- lighting

fn light_channel(index: u32, vertex_color: vec4f, mv_pos: vec3f, mv_nrm: vec3f) -> vec4f {
  let config = cfg(CFG_CHANNELS + index);
  let material = select(ubuf.cc_mat[index], vertex_color, nibble_of(config, 0u) == 1u);
  if (nibble_of(config, 4u) == 0u) {
    return material;
  }
  let diff_fn = nibble_of(config, 2u);
  let attn_fn = nibble_of(config, 3u);
  var lighting = select(ubuf.cc_amb[index], vertex_color, nibble_of(config, 1u) == 1u);
  for (var i = 0u; i < 8u; i++) {
    if ((ubuf.light_state[index] & (1u << i)) == 0u) { continue; }
    let light = ubuf.lights[i];
    var ldir = light.pos - mv_pos;
    let dist2 = dot(ldir, ldir);
    let dist = sqrt(dist2);
    ldir = ldir / dist;
    var attn = 1.0;
    if (attn_fn == 1u) { // spot
      let cosine = max(0.0, dot(ldir, light.dir));
      let cos_attn = dot(light.cos_att, vec3f(1.0, cosine, cosine * cosine));
      let dist_attn = dot(light.dist_att, vec3f(1.0, dist, dist2));
      attn = max(0.0, cos_attn / dist_attn);
    } else if (attn_fn == 0u) { // specular
      attn = select(0.0, max(0.0, dot(mv_nrm, light.dir)), dot(mv_nrm, ldir) >= 0.0);
      let cos_attn = dot(light.cos_att, vec3f(1.0, attn, attn * attn));
      var dist_attn: f32;
      if (diff_fn != 0u) {
        dist_attn = max(0.0, dot(normalize(light.dist_att), vec3f(1.0, attn, attn * attn)));
      } else {
        dist_attn = max(0.0, dot(light.dist_att, vec3f(1.0, attn, attn * attn)));
      }
      attn = max(0.0, cos_attn / dist_attn);
    }
    var diff = 1.0;
    if (diff_fn == 1u) {
      diff = dot(ldir, mv_nrm);
    } else if (diff_fn == 2u) {
      diff = max(0.0, dot(ldir, mv_nrm));
    }
    lighting = lighting + (attn * diff * light.color);
  }
  return material * clamp(lighting, vec4f(0.0), vec4f(1.0));
}

// ---------------------------------------------------------------- vertex stage

const TG_MTX3X4 = 0u;
const TG_MTX2X4 = 1u;
const TG_BUMP0 = 2u;
const TG_BUMP7 = 9u;
const TG_SRTG = 10u;
const TG_POS = 0u;
const TG_NRM = 1u;
const TG_BINRM = 2u;
const TG_TANGENT = 3u;
const TG_TEX0 = 4u;
const TG_TEX7 = 11u;
const TG_COLOR0 = 19u;

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) cc0: vec4f,
    @location(1) cc1: vec4f,
    @location(2) tex0: vec3f,
    @location(3) tex1: vec3f,
    @location(4) tex2: vec3f,
    @location(5) tex3: vec3f,
    @location(6) tex4: vec3f,
    @location(7) tex5: vec3f,
    @location(8) tex6: vec3f,
    @location(9) tex7: vec3f,
};

@vertex
fn vs_main(@builtin(vertex_index) vidx: u32, @builtin(instance_index) iidx: u32) -> VertexOutput {
    var out: VertexOutput;
    let line_mode = cfg(CFG_LINE_MODE);
    // Single-instance draws carry their vertex byte offset as firstInstance.
    vtx_start_of_draw = select(iidx, ubuf.vtx_start, line_mode != 0u);
    var in_vidx = vidx;
    var in_pos: vec3f;
    var in_pnmtxidx: u32;
    var mv_pos: vec3f;
    if (line_mode == 0u) {
        in_pos = load_pos(vidx);
        in_pnmtxidx = load_pnmtx(vidx);
        mv_pos = vec4f(in_pos, 1.0) * ubuf.postex_mtx[in_pnmtxidx];
        out.pos = vec4f(mv_pos, 1.0) * ubuf.proj;
    } else if (line_mode == 3u) {
        // GX_POINTS: expand a vertex to an axis-aligned screen-space square.
        in_vidx = iidx;
        in_pos = load_pos(in_vidx);
        in_pnmtxidx = load_pnmtx(in_vidx);
        mv_pos = vec4f(in_pos, 1.0) * ubuf.postex_mtx[in_pnmtxidx];
        let clip = vec4f(mv_pos, 1.0) * ubuf.proj;
        let viewport_scale = ubuf.render_viewport_size / max(ubuf.logical_viewport_size, vec2f(1.0));
        let point_size = cfg_f32(CFG_LINE_WIDTH) * min(viewport_scale.x, viewport_scale.y);
        let x_sign = select(-1.0, 1.0, (vidx & 1u) != 0u);
        let y_sign = select(-1.0, 1.0, vidx >= 2u);
        let offset_px = vec2f(x_sign, y_sign) * (point_size / 2.0);
        let offset_ndc = (offset_px * 2.0) / ubuf.render_viewport_size;
        out.pos = vec4f(clip.xy + offset_ndc * clip.w, clip.zw);
    } else {
        // GX_LINES, GX_LINESTRIP: expand a segment perpendicular to its direction.
        let use_b = vidx >= 2u;
        let vidx_a = iidx * select(1u, 2u, line_mode == 1u);
        let vidx_b = vidx_a + 1u;
        in_vidx = select(vidx_a, vidx_b, use_b);
        let pos_a = load_pos(vidx_a);
        let pos_b = load_pos(vidx_b);
        in_pos = select(pos_a, pos_b, use_b);
        let pnmtxidx_a = load_pnmtx(vidx_a);
        let pnmtxidx_b = load_pnmtx(vidx_b);
        in_pnmtxidx = select(pnmtxidx_a, pnmtxidx_b, use_b);
        let mv_pos_a = vec4f(pos_a, 1.0) * ubuf.postex_mtx[pnmtxidx_a];
        let mv_pos_b = vec4f(pos_b, 1.0) * ubuf.postex_mtx[pnmtxidx_b];
        mv_pos = select(mv_pos_a, mv_pos_b, use_b);
        let clip_a = vec4f(mv_pos_a, 1.0) * ubuf.proj;
        let clip_b = vec4f(mv_pos_b, 1.0) * ubuf.proj;
        let ndc_a = clip_a.xy / clip_a.w;
        let ndc_b = clip_b.xy / clip_b.w;
        let viewport_scale = ubuf.render_viewport_size / max(ubuf.logical_viewport_size, vec2f(1.0));
        let delta_px = (ndc_b - ndc_a) / 2.0 * ubuf.render_viewport_size;
        let dir_px = select(vec2f(1.0, 0.0), normalize(delta_px), dot(delta_px, delta_px) > 1e-10);
        let perp_px = vec2f(-dir_px.y, dir_px.x);
        let line_width = cfg_f32(CFG_LINE_WIDTH) * min(viewport_scale.x, viewport_scale.y);
        let offset_px = perp_px * (line_width / 2.0) * select(-1.0, 1.0, (vidx & 1u) != 0u);
        let offset_ndc = (offset_px * 2.0) / ubuf.render_viewport_size;
        let clip_base = select(clip_a, clip_b, use_b);
        out.pos = vec4f(clip_base.xy + offset_ndc * clip_base.w, clip_base.zw);
    }
    out.pos.z = -out.pos.z * (1.0 - 1e-7);

    let in_nrm = load_nrm(in_vidx);
    let nrm_tmp = vec4f(in_nrm, 0.0) * ubuf.nrm_mtx[in_pnmtxidx];
    let mv_nrm = select(nrm_tmp, normalize(nrm_tmp), dot(nrm_tmp, nrm_tmp) > 1e-10);

    let sampled_channels = cfg(CFG_SAMPLED_CHANNELS);
    if ((sampled_channels & 1u) != 0u) {
        let color = load_color(0u, in_vidx);
        out.cc0 = light_channel(0u, color, mv_pos, mv_nrm);
        out.cc0.a = light_channel(2u, color, mv_pos, mv_nrm).a;
    }
    if ((sampled_channels & 2u) != 0u) {
        let color = load_color(1u, in_vidx);
        out.cc1 = light_channel(1u, color, mv_pos, mv_nrm);
        out.cc1.a = light_channel(3u, color, mv_pos, mv_nrm).a;
    }

    var tc_out: array<vec3f, 8>;
    let sampled_coords = cfg(CFG_SAMPLED_COORDS);
    for (var i = 0u; i < 8u; i++) {
        if ((sampled_coords & (1u << i)) == 0u) { continue; }
        let w0 = cfg(CFG_TCGS + i * 2u);
        let w1 = cfg(CFG_TCGS + i * 2u + 1u);
        let type_ = byte_of(w0, 0u);
        let src = byte_of(w0, 1u);
        if (type_ >= TG_BUMP0 && type_ <= TG_BUMP7) {
            // Emboss: the source texcoord, moved by the light across the surface.
            let ldir = normalize(ubuf.lights[type_ - TG_BUMP0].pos - mv_pos);
            let tangent = vec4f(load_nbt_slice(2u, in_vidx), 0.0) * ubuf.nrm_mtx[in_pnmtxidx];
            let binormal = vec4f(load_nbt_slice(1u, in_vidx), 0.0) * ubuf.nrm_mtx[in_pnmtxidx];
            let uv = tc_out[byte_of(w0, 2u)].xy + vec2f(dot(ldir, tangent), dot(ldir, binormal));
            tc_out[i] = vec3f(uv, 1.0);
            continue;
        }
        var tc: vec4f;
        if (src >= TG_TEX0 && src <= TG_TEX7) {
            tc = vec4f(load_attr(VA_TEX0 + (src - TG_TEX0), in_vidx).xy, 1.0, 1.0);
        } else if (src == TG_POS) {
            tc = vec4f(in_pos, 1.0);
        } else if (src == TG_NRM) {
            tc = vec4f(in_nrm, 1.0);
        } else if (src >= TG_COLOR0) {
            // XF texgen sees the channel after lighting, not the vertex color.
            tc = select(out.cc0, out.cc1, src != TG_COLOR0);
        } else if (src == TG_BINRM) {
            tc = vec4f(load_nbt_slice(1u, in_vidx), 1.0);
        } else {
            tc = vec4f(load_nbt_slice(2u, in_vidx), 1.0);
        }
        var tmp: vec3f;
        if (type_ == TG_SRTG) {
            tmp = vec3f(tc.xy, 1.0);
        } else {
            if (((w0 >> 25u) & 1u) != 0u) {
                tmp = tc * ubuf.postex_mtx[load_index_byte(VA_TEX0MTXIDX + i, in_vidx) / 3u];
            } else if (byte_of(w1, 0u) == 0xFFu) {
                tmp = tc.xyz;
            } else {
                tmp = tc * ubuf.postex_mtx[byte_of(w1, 0u)];
            }
            if (type_ == TG_MTX2X4) {
                tmp.z = 1.0;
            }
        }
        if (((w0 >> 24u) & 1u) != 0u) {
            tmp = normalize(tmp);
        }
        var proj = tmp;
        if (byte_of(w1, 1u) != 0xFFu) {
            proj = vec4f(tmp, 1.0) * ubuf.postmtx[byte_of(w1, 1u)];
        }
        if (line_mode != 0u && (cfg(CFG_LINE_TEX_MASK) & (1u << i)) != 0u) {
            let tex_offset = cfg_f32(CFG_LINE_TEX_OFFSET);
            if (line_mode == 3u) {
                if ((vidx & 1u) != 0u) { proj.x += tex_offset; }
                if (vidx >= 2u) { proj.y += tex_offset; }
            } else if ((vidx & 1u) != 0u) {
                proj.y += tex_offset;
            }
        }
        tc_out[i] = proj;
    }
    out.tex0 = tc_out[0];
    out.tex1 = tc_out[1];
    out.tex2 = tc_out[2];
    out.tex3 = tc_out[3];
    out.tex4 = tc_out[4];
    out.tex5 = tc_out[5];
    out.tex6 = tc_out[6];
    out.tex7 = tc_out[7];
    return out;
}

// ---------------------------------------------------------------- fragment stage

const FLAG_COLOR_CLAMP = 1u << 0u;
const FLAG_ALPHA_CLAMP = 1u << 1u;
const FLAG_COLOR_OVERFLOW_A = 1u << 2u;
const FLAG_ALPHA_OVERFLOW_A = 1u << 5u;
const FLAG_NEEDS_TEV_COORD = 1u << 8u;
const FLAG_NEEDS_SAMPLE = 1u << 9u;
const FLAG_HAS_BASE_TEXTURE = 1u << 10u;
const FLAG_HAS_BASE_COORD = 1u << 11u;
const FLAG_SIMPLE_COORDS = 1u << 12u;
const FLAG_ADD_PREVIOUS = 1u << 13u;
const FLAG_WRITES_COORD = 1u << 14u;
const FLAG_ALPHA_COMPARES_COLOR = 1u << 15u;

fn sample_map(map: u32, uv: vec2f, bias: f32) -> vec4f {
  switch map {
    case 0u: { return textureSampleBias(tex0, tex0_samp, uv, bias); }
    case 1u: { return textureSampleBias(tex1, tex1_samp, uv, bias); }
    case 2u: { return textureSampleBias(tex2, tex2_samp, uv, bias); }
    case 3u: { return textureSampleBias(tex3, tex3_samp, uv, bias); }
    case 4u: { return textureSampleBias(tex4, tex4_samp, uv, bias); }
    case 5u: { return textureSampleBias(tex5, tex5_samp, uv, bias); }
    case 6u: { return textureSampleBias(tex6, tex6_samp, uv, bias); }
    default: { return textureSampleBias(tex7, tex7_samp, uv, bias); }
  }
}

fn wrap_component(wrap: u32, coord: f32) -> f32 {
  if (wrap == 0u) { return coord; }
  if (wrap >= 6u) { return 0.0; }
  return coord % f32(512u >> wrap);
}

var<private> regs: array<vec4f, 4>;
var<private> rast: array<vec4f, 2>;
var<private> ind_coord: array<vec3f, 4>;
var<private> stage_sample: vec4f;
var<private> stage_konst: u32;
var<private> stage_tex: u32;
var<private> stage_ind: u32;

fn swizzled(value: vec4f, swizzle: u32) -> vec4f {
  return vec4f(value[swizzle & 3u], value[(swizzle >> 2u) & 3u], value[(swizzle >> 4u) & 3u], value[(swizzle >> 6u) & 3u]);
}

fn stage_texture() -> vec4f {
  return swizzled(stage_sample, byte_of(stage_tex, 3u));
}

// What the stage rasterizes; an alpha bump fills all four components.
fn stage_rasterized() -> vec4f {
  let kind = byte_of(stage_konst, 2u);
  if (kind == 0u) {
    return vec4f(0.0);
  }
  if (kind <= 2u) {
    return swizzled(rast[kind - 1u], byte_of(stage_tex, 2u));
  }
  let coord = ind_coord[nibble_of(stage_ind, 0u)][nibble_of(stage_ind, 3u) - 1u];
  var alpha = f32(u32(coord) & byte_of(stage_konst, 3u)) / 255.0;
  if (kind == 4u) {
    alpha = alpha * (255.0 / 248.0);
  }
  return vec4f(alpha);
}

fn konst_fraction(selection: u32) -> f32 {
  return f32(8u - selection) / 8.0;
}

fn konst_component(selection: u32) -> f32 {
  let from_k0 = selection - 0x10u;
  return ubuf.kcolors[from_k0 & 3u][from_k0 >> 2u];
}

fn color_arg(arg: u32) -> vec3f {
  switch arg {
    case 0u: { return regs[0].rgb; }
    case 1u: { return vec3f(regs[0].a); }
    case 2u: { return regs[1].rgb; }
    case 3u: { return vec3f(regs[1].a); }
    case 4u: { return regs[2].rgb; }
    case 5u: { return vec3f(regs[2].a); }
    case 6u: { return regs[3].rgb; }
    case 7u: { return vec3f(regs[3].a); }
    case 8u: { return stage_texture().rgb; }
    case 9u: { return vec3f(stage_texture().a); }
    case 10u: { return stage_rasterized().rgb; }
    case 11u: { return vec3f(stage_rasterized().a); }
    case 12u: { return vec3f(1.0); }
    case 13u: { return vec3f(0.5); }
    case 14u: {
      let selection = byte_of(stage_konst, 0u);
      if (selection < 8u) { return vec3f(konst_fraction(selection)); }
      if (selection < 0x10u) { return ubuf.kcolors[selection - 0xCu].rgb; }
      return vec3f(konst_component(selection));
    }
    default: { return vec3f(0.0); }
  }
}

fn alpha_arg(arg: u32) -> f32 {
  switch arg {
    case 0u: { return regs[0].a; }
    case 1u: { return regs[1].a; }
    case 2u: { return regs[2].a; }
    case 3u: { return regs[3].a; }
    case 4u: { return stage_texture().a; }
    case 5u: { return stage_rasterized().a; }
    case 6u: {
      let selection = byte_of(stage_konst, 1u);
      if (selection < 8u) { return konst_fraction(selection); }
      return konst_component(selection);
    }
    default: { return 0.0; }
  }
}

fn tev_bias(bias: u32) -> f32 {
  if (bias == 1u) { return 0.5; }
  if (bias == 2u) { return -0.5; }
  return 0.0;
}

fn tev_scale(scale: u32) -> f32 {
  switch scale {
    case 1u: { return 2.0; }
    case 2u: { return 4.0; }
    case 3u: { return 0.5; }
    default: { return 1.0; }
  }
}

fn rg16(v: vec3f) -> f32 {
  return round(dot(v.rg * 255.0, vec2(1.0, 256.0)));
}
fn bgr24(v: vec3f) -> f32 {
  return round(dot(v.rgb * 255.0, vec3(1.0, 256.0, 65536.0)));
}
fn r8(v: vec3f) -> f32 {
  return round(v.r * 255.0);
}

// The packed comparisons: whether a's value passes against b's.
fn packed_compare(op: u32, a: vec3f, b: vec3f) -> bool {
  switch op {
    case 8u: { return r8(a) > r8(b); }
    case 9u: { return r8(a) == r8(b); }
    case 10u: { return rg16(a) > rg16(b); }
    case 11u: { return rg16(a) == rg16(b); }
    case 12u: { return bgr24(a) > bgr24(b); }
    default: { return bgr24(a) == bgr24(b); }
  }
}

fn tev_color(op_word: u32, clamped: bool, a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f {
  let op = byte_of(op_word, 0u);
  var result: vec3f;
  if (op <= 1u) {
    let mixed = select(mix(a, b, c), -mix(a, b, c), op == 1u);
    result = (mixed + d + tev_bias(byte_of(op_word, 1u))) * tev_scale(byte_of(op_word, 2u));
  } else if (op <= 13u) {
    result = select(vec3(0.0), c, packed_compare(op, a, b)) + d;
  } else if (op == 14u) {
    result = select(vec3(0.0), c, round(a * 255.0) > round(b * 255.0)) + d;
  } else {
    result = select(vec3(0.0), c, round(a * 255.0) == round(b * 255.0)) + d;
  }
  if (clamped) {
    return clamp(result, vec3f(0.0), vec3f(1.0));
  }
  return clamp(result, vec3f(-4.0), vec3f(4.0));
}

fn tev_alpha(op_word: u32, clamped: bool, a: f32, b: f32, c: f32, d: f32, color_a: vec3f, color_b: vec3f) -> f32 {
  let op = byte_of(op_word, 0u);
  var result: f32;
  if (op <= 1u) {
    let mixed = select(mix(a, b, c), -mix(a, b, c), op == 1u);
    result = (mixed + d + tev_bias(byte_of(op_word, 1u))) * tev_scale(byte_of(op_word, 2u));
  } else if (op <= 13u) {
    result = select(0.0, c, packed_compare(op, color_a, color_b)) + d;
  } else if (op == 14u) {
    result = select(0.0, c, round(a * 255.0) > round(b * 255.0)) + d;
  } else {
    result = select(0.0, c, round(a * 255.0) == round(b * 255.0)) + d;
  }
  if (clamped) {
    return clamp(result, 0.0, 1.0);
  }
  return clamp(result, -4.0, 4.0);
}

fn compares(comp: u32, value: u32, reference: u32) -> bool {
  switch comp {
    case 0u: { return false; }
    case 1u: { return value < reference; }
    case 2u: { return value == reference; }
    case 3u: { return value <= reference; }
    case 4u: { return value > reference; }
    case 5u: { return value != reference; }
    case 6u: { return value >= reference; }
    default: { return true; }
  }
}

fn tex_coord(in: VertexOutput, i: u32) -> vec3f {
  switch i {
    case 0u: { return in.tex0; }
    case 1u: { return in.tex1; }
    case 2u: { return in.tex2; }
    case 3u: { return in.tex3; }
    case 4u: { return in.tex4; }
    case 5u: { return in.tex5; }
    case 6u: { return in.tex6; }
    default: { return in.tex7; }
  }
}

fn tev_shade(in: VertexOutput) -> vec4f {
    let loads = cfg(CFG_LOADS_TEV_REG);
    for (var i = 0u; i < 4u; i++) {
        if ((loads & (1u << i)) != 0u) {
            regs[i] = ubuf.tev_regs[i];
        }
    }
    rast[0] = in.cc0;
    rast[1] = in.cc1;

    var tex_uv: array<vec2f, 8>;
    let sampled_coords = cfg(CFG_SAMPLED_COORDS);
    for (var i = 0u; i < 8u; i++) {
        if ((sampled_coords & (1u << i)) == 0u) { continue; }
        let coord = tex_coord(in, i);
        if (byte_of(cfg(CFG_TCGS + i * 2u), 0u) == TG_MTX3X4) {
            tex_uv[i] = coord.xy / coord.z;
        } else {
            tex_uv[i] = coord.xy;
        }
    }

    let used_ind = cfg(CFG_USED_IND_STAGES);
    for (var i = 0u; i < 4u; i++) {
        if ((used_ind & (1u << i)) == 0u) { continue; }
        let w0 = cfg(CFG_IND + i * 2u);
        let w1 = cfg(CFG_IND + i * 2u + 1u);
        let coord = byte_of(w0, 0u);
        let map = byte_of(w0, 1u);
        let scale = vec2f(1.0 / f32(1u << byte_of(w1, 0u)), 1.0 / f32(1u << byte_of(w1, 1u)));
        let uv = tex_uv[coord] * ubuf.texcoord_scale[coord].xy * scale / ubuf.tex_size_bias[map].xy;
        ind_coord[i] = 255.0 * sample_map(map, uv, ubuf.tex_size_bias[map].z).abg;
    }

    // Each stage's texture, with the texcoords the indirect stages carry on.
    var samples: array<vec4f, 16>;
    var t_tex_coord = vec2f(0.0);
    let stages = cfg(CFG_TEV_STAGES);
    for (var i = 0u; i < stages; i++) {
        let base = CFG_STAGES + i * CFG_STAGE_WORDS;
        let flags = cfg(base + 7u);
        if ((flags & (FLAG_NEEDS_TEV_COORD | FLAG_NEEDS_SAMPLE)) == 0u) { continue; }
        let tex = cfg(base + 5u);
        let coord = byte_of(tex, 0u);
        let map = byte_of(tex, 1u);
        var uv_set = false;
        var uv: vec2f;
        if ((flags & FLAG_NEEDS_TEV_COORD) != 0u) {
            let ind = cfg(base + 6u);
            let matrix_kind = nibble_of(ind, 4u);
            var offset = vec2f(0.0);
            if (matrix_kind != 0u) {
                let format = nibble_of(ind, 1u);
                var ind_value = ind_coord[nibble_of(ind, 0u)];
                if (format != 0u) {
                    ind_value = floor(ind_value / f32(4u << format));
                }
                let bias_sel = nibble_of(ind, 2u);
                if (bias_sel != 0u) {
                    let bias = select(1.0, -128.0, format == 0u);
                    ind_value = ind_value + vec3f(f32(bias_sel & 1u), f32((bias_sel >> 1u) & 1u), f32((bias_sel >> 2u) & 1u)) * bias;
                }
                let matrix_index = nibble_of(ind, 5u);
                let c0 = ubuf.ind_mtx[matrix_index][0];
                let c1 = ubuf.ind_mtx[matrix_index][1];
                if (matrix_kind == 1u) {
                    offset = vec2f(dot(vec3f(c0.xz, c1.x), ind_value), dot(vec3f(c0.yw, c1.y), ind_value)) * c1.z;
                } else if (matrix_kind == 2u) {
                    offset = tex_uv[coord] * ubuf.texcoord_scale[coord].xy * ind_value.x * c1.z / 256.0;
                } else {
                    offset = tex_uv[coord] * ubuf.texcoord_scale[coord].xy * ind_value.y * c1.z / 256.0;
                }
            }
            if ((flags & FLAG_WRITES_COORD) != 0u) {
                var final_coord = offset;
                if ((flags & FLAG_HAS_BASE_COORD) != 0u) {
                    var base_coord = tex_uv[coord];
                    if ((flags & FLAG_SIMPLE_COORDS) == 0u) {
                        base_coord = base_coord * ubuf.texcoord_scale[coord].xy;
                    }
                    let wrapped = vec2f(wrap_component(nibble_of(ind, 6u), base_coord.x), wrap_component(nibble_of(ind, 7u), base_coord.y));
                    final_coord = wrapped + offset;
                }
                if ((flags & FLAG_ADD_PREVIOUS) != 0u) {
                    t_tex_coord += final_coord;
                } else {
                    t_tex_coord = final_coord;
                }
                if ((flags & FLAG_NEEDS_SAMPLE) != 0u && (flags & FLAG_HAS_BASE_TEXTURE) != 0u) {
                    uv = t_tex_coord;
                    if ((flags & FLAG_SIMPLE_COORDS) == 0u) {
                        uv = uv / ubuf.tex_size_bias[map].xy;
                    }
                    uv_set = true;
                }
            }
        }
        if ((flags & FLAG_NEEDS_SAMPLE) == 0u) { continue; }
        if ((flags & FLAG_HAS_BASE_TEXTURE) == 0u) {
            // Tex order disabled: the hardware feeds TEV white.
            samples[i] = vec4f(1.0);
            continue;
        }
        if (!uv_set) {
            uv = tex_uv[coord];
        }
        samples[i] = sample_map(map, uv, ubuf.tex_size_bias[map].z);
    }

    for (var i = 0u; i < stages; i++) {
        let base = CFG_STAGES + i * CFG_STAGE_WORDS;
        let color_args = cfg(base);
        let alpha_args = cfg(base + 1u);
        let color_op = cfg(base + 2u);
        let alpha_op = cfg(base + 3u);
        let flags = cfg(base + 7u);
        stage_konst = cfg(base + 4u);
        stage_tex = cfg(base + 5u);
        stage_ind = cfg(base + 6u);
        stage_sample = samples[i];

        var color_a = color_arg(byte_of(color_args, 0u));
        var color_b = color_arg(byte_of(color_args, 1u));
        var color_c = color_arg(byte_of(color_args, 2u));
        if ((flags & FLAG_COLOR_OVERFLOW_A) != 0u) { color_a = tev_overflow_vec3f(color_a); }
        if ((flags & (FLAG_COLOR_OVERFLOW_A << 1u)) != 0u) { color_b = tev_overflow_vec3f(color_b); }
        if ((flags & (FLAG_COLOR_OVERFLOW_A << 2u)) != 0u) { color_c = tev_overflow_vec3f(color_c); }
        let color_result = tev_color(color_op, (flags & FLAG_COLOR_CLAMP) != 0u, color_a, color_b, color_c, color_arg(byte_of(color_args, 3u)));
        let color_out = byte_of(color_op, 3u);
        regs[color_out] = vec4f(color_result, regs[color_out].a);

        var alpha_a = alpha_arg(byte_of(alpha_args, 0u));
        var alpha_b = alpha_arg(byte_of(alpha_args, 1u));
        var alpha_c = alpha_arg(byte_of(alpha_args, 2u));
        if ((flags & FLAG_ALPHA_OVERFLOW_A) != 0u) { alpha_a = tev_overflow_f32(alpha_a); }
        if ((flags & (FLAG_ALPHA_OVERFLOW_A << 1u)) != 0u) { alpha_b = tev_overflow_f32(alpha_b); }
        if ((flags & (FLAG_ALPHA_OVERFLOW_A << 2u)) != 0u) { alpha_c = tev_overflow_f32(alpha_c); }
        let alpha_out = byte_of(alpha_op, 3u);
        regs[alpha_out].a = tev_alpha(alpha_op, (flags & FLAG_ALPHA_CLAMP) != 0u, alpha_a, alpha_b, alpha_c, alpha_arg(byte_of(alpha_args, 3u)), color_a, color_b);
    }

    var prev = regs[0];
    let last_outputs = cfg(CFG_LAST_OUTPUTS);
    if (byte_of(last_outputs, 0u) != 0u) {
        prev = vec4f(regs[byte_of(last_outputs, 0u)].rgb, prev.a);
    }
    if (byte_of(last_outputs, 1u) != 0u) {
        prev.a = regs[byte_of(last_outputs, 1u)].a;
    }

    let fog_type = cfg(CFG_FOG_TYPE);
    if (fog_type != 0u) {
        let fog_depth = 1.0 - in.pos.z;
        var fog_base: f32;
        if ((fog_type & 8u) != 0u) {
            fog_base = ubuf.fog.a * fog_depth;
        } else {
            fog_base = ubuf.fog.a / (ubuf.fog.b - fog_depth);
        }
        if (cfg(CFG_FOG_RANGE) != 0u) {
            fog_base *= bitcast<f32>(abuf[ubuf.fog_range_base + u32(in.pos.x)]);
        }
        var fog_f = clamp(fog_base - ubuf.fog.c, 0.0, 1.0);
        var fog_z: f32;
        switch fog_type & 7u {
            case 2u: { fog_z = fog_f; }
            case 4u: { fog_z = 1.0 - exp2(-8.0 * fog_f); }
            case 5u: { fog_z = 1.0 - exp2(-8.0 * fog_f * fog_f); }
            case 6u: { fog_z = exp2(-8.0 * (1.0 - fog_f)); }
            default: {
                fog_f = 1.0 - fog_f;
                fog_z = exp2(-8.0 * fog_f * fog_f);
            }
        }
        prev = vec4f(mix(prev.rgb, ubuf.fog.color.rgb, clamp(fog_z, 0.0, 1.0)), prev.a);
    }

    let prev_overflow = cfg(CFG_PREV_OVERFLOW);
    if ((prev_overflow & 1u) != 0u) {
        prev = vec4f(tev_overflow_vec3f(prev.rgb), prev.a);
    }
    if ((prev_overflow & 2u) != 0u) {
        prev.a = tev_overflow_f32(prev.a);
    }

    let alpha_width = max(fwidth(prev.a), 0.0001);
    if (cfg(CFG_ALPHA_COMPARE) != 0u) {
        if (cfg(CFG_ALPHA_TO_COVERAGE) != 0u) {
            // Coverage from how far alpha is past the edge, in pixels.
            prev.a = clamp((prev.a - cfg_f32(CFG_COVERAGE_THRESHOLD)) / alpha_width + 0.5, 0.0, 1.0);
        } else {
            let alpha = u32(round(clamp(prev.a, 0.0, 1.0) * 255.0));
            let pass0 = compares(cfg(CFG_ALPHA_COMP0), alpha, cfg(CFG_ALPHA_REF0));
            let pass1 = compares(cfg(CFG_ALPHA_COMP1), alpha, cfg(CFG_ALPHA_REF1));
            var passes: bool;
            switch cfg(CFG_ALPHA_OP) {
                case 0u: { passes = pass0 && pass1; }
                case 1u: { passes = pass0 || pass1; }
                case 2u: { passes = pass0 != pass1; }
                default: { passes = pass0 == pass1; }
            }
            if (!passes) { discard; }
        }
    }
    return prev;
}
)""";

std::string uber_source(DstAlphaMode mode) {
  std::string source;
  if (mode == DstAlphaMode::DualSource) {
    source += "enable dual_source_blending;\n";
  }
  // Each texture is chosen at run time, inside the loop over the stages.
  source += "diagnostic(off, derivative_uniformity);\n";
  source += shader_helpers();
  const auto constant = [&source](std::string_view name, size_t offset) {
    source += fmt::format("const {} = {}u;\n", name, offset / sizeof(u32));
  };
  constant("CFG_TEV_STAGES", offsetof(UberConfig, tevStages));
  constant("CFG_VTX_STRIDE", offsetof(UberConfig, vtxStride));
  constant("CFG_LINE_MODE", offsetof(UberConfig, lineMode));
  constant("CFG_FOG_TYPE", offsetof(UberConfig, fogType));
  constant("CFG_FOG_RANGE", offsetof(UberConfig, fogRange));
  constant("CFG_LOADS_TEV_REG", offsetof(UberConfig, loadsTevReg));
  constant("CFG_SAMPLED_COORDS", offsetof(UberConfig, sampledTexCoords));
  constant("CFG_SAMPLED_CHANNELS", offsetof(UberConfig, sampledChannels));
  constant("CFG_USED_IND_STAGES", offsetof(UberConfig, usedIndStages));
  constant("CFG_ALPHA_COMPARE", offsetof(UberConfig, alphaCompare));
  constant("CFG_ALPHA_COMP0", offsetof(UberConfig, alphaComp0));
  constant("CFG_ALPHA_REF0", offsetof(UberConfig, alphaRef0));
  constant("CFG_ALPHA_OP", offsetof(UberConfig, alphaOp));
  constant("CFG_ALPHA_COMP1", offsetof(UberConfig, alphaComp1));
  constant("CFG_ALPHA_REF1", offsetof(UberConfig, alphaRef1));
  constant("CFG_ALPHA_TO_COVERAGE", offsetof(UberConfig, alphaToCoverage));
  constant("CFG_COVERAGE_THRESHOLD", offsetof(UberConfig, coverageThreshold));
  constant("CFG_PREV_OVERFLOW", offsetof(UberConfig, prevOverflow));
  constant("CFG_LAST_OUTPUTS", offsetof(UberConfig, lastOutputs));
  constant("CFG_LINE_WIDTH", offsetof(UberConfig, lineWidth));
  constant("CFG_LINE_TEX_OFFSET", offsetof(UberConfig, lineTexOffset));
  constant("CFG_LINE_TEX_MASK", offsetof(UberConfig, lineTexMask));
  constant("CFG_CHANNELS", offsetof(UberConfig, channels));
  constant("CFG_TCGS", offsetof(UberConfig, tcgs));
  constant("CFG_IND", offsetof(UberConfig, ind));
  constant("CFG_ATTRS", offsetof(UberConfig, attrs));
  constant("CFG_STAGES", offsetof(UberConfig, stages));
  constant("CFG_STAGE_WORDS", sizeof(UberStage));
  source += UberSource;
  switch (mode) {
  case DstAlphaMode::None:
    source += "@fragment\nfn fs_main(in: VertexOutput) -> @location(0) vec4f {\n  return tev_shade(in);\n}\n";
    break;
  case DstAlphaMode::Replace:
    // The blend constant is the EFB's alpha.
    source += "@fragment\nfn fs_main(in: VertexOutput) -> @location(0) vec4f {\n"
              "  return vec4f(tev_shade(in).rgb, 1.0);\n}\n";
    break;
  case DstAlphaMode::DualSource:
    source += "struct FragmentOutput {\n"
              "    @location(0) @blend_src(0) color: vec4f,\n"
              "    @location(0) @blend_src(1) blend: vec4f,\n"
              "}\n"
              "@fragment\nfn fs_main(in: VertexOutput) -> FragmentOutput {\n"
              "  let prev = tev_shade(in);\n"
              "  return FragmentOutput(vec4f(prev.rgb, 1.0), prev);\n}\n";
    break;
  }
  return source;
}

wgpu::ShaderModule uber_module(DstAlphaMode mode) {
  static std::mutex mutex;
  static std::array<wgpu::ShaderModule, 3> modules;
  std::scoped_lock lock{mutex};
  auto& module = modules[static_cast<size_t>(mode)];
  if (!module) {
    const std::string source = uber_source(mode);
    wgpu::ShaderSourceWGSL wgslDescriptor{};
    wgslDescriptor.code = source.c_str();
    const wgpu::ShaderModuleDescriptor descriptor{
        .nextInChain = &wgslDescriptor,
        .label = "GX Ubershader",
    };
    module = webgpu::g_device.CreateShaderModule(&descriptor);
  }
  return module;
}
} // namespace

PipelineConfig uber_pipeline_config(const PipelineConfig& config) noexcept {
  PipelineConfig uber = config;
  uber.shaderConfig = {};
  uber.shaderConfig.alphaToCoverage = config.shaderConfig.alphaToCoverage;
  return uber;
}

wgpu::RenderPipeline create_uber_pipeline(const PipelineConfig& config) {
  const auto mode = dst_alpha_mode(config);
  const auto label = fmt::format("GX Ubershader Pipeline {:x}", xxh3_hash(config));
  return build_pipeline(config, mode, {}, uber_module(mode), label.c_str());
}

u32 push_uber_config(const ShaderConfig& config, const ShaderInfo& info) noexcept {
  UberConfig words = build_config(config, info);
  if (config.lineMode == 3) {
    words.lineWidth = static_cast<f32>(g_gxState.pointSize) / 6.f;
    words.lineTexOffset = tex_offset_value(g_gxState.pointTexOffset);
    words.lineTexMask = point_texcoord_mask();
  } else if (config.lineMode != 0) {
    words.lineWidth = static_cast<f32>(g_gxState.lineWidth) / 6.f;
    words.lineTexOffset = tex_offset_value(g_gxState.lineTexOffset);
    words.lineTexMask = line_texcoord_mask();
  }
  // In the frame's storage, once for each configuration its draws have.
  static absl::flat_hash_map<HashType, u32> pushed;
  static u32 pushedFrame = UINT32_MAX;
  if (pushedFrame != gfx::current_frame()) {
    pushedFrame = gfx::current_frame();
    pushed.clear();
  }
  const auto [it, inserted] = pushed.try_emplace(xxh3_hash_s(&words, sizeof(words)));
  if (inserted) {
    const auto range = gfx::push_storage(reinterpret_cast<const u8*>(&words), sizeof(words));
    it->second = static_cast<u32>(range.offset / sizeof(u32));
  }
  return it->second;
}
} // namespace aurora::gx
