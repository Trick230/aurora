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

#include "../gpu/gpu.hpp"
#include "shader_hlsl.hpp"
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


template <typename Constant>
void for_each_config_word(const Constant& constant) {
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
}

} // namespace

std::string hlsl::uber_source(DstAlphaMode mode) {
  std::string source = shader_helpers();
  for_each_config_word([&source](std::string_view name, size_t offset) {
    source += fmt::format("static const uint {} = {}u;\n", name, offset / sizeof(u32));
  });
  source += uber_body();
  switch (mode) {
  case DstAlphaMode::None:
    source += "float4 fs_main(VertexOutput input) : SV_Target {\n  return tev_shade(input);\n}\n";
    break;
  case DstAlphaMode::Replace:
    // The blend constant is the EFB's alpha.
    source += "float4 fs_main(VertexOutput input) : SV_Target {\n"
              "  return vec4f(tev_shade(input).rgb, 1.0);\n}\n";
    break;
  case DstAlphaMode::DualSource:
    source += "struct FragmentOutput {\n"
              "    [[vk::location(0)]] [[vk::index(0)]] float4 color : SV_Target0;\n"
              "    [[vk::location(0)]] [[vk::index(1)]] float4 blend : SV_Target1;\n"
              "};\n"
              "FragmentOutput fs_main(VertexOutput input) {\n"
              "  float4 prev = tev_shade(input);\n"
              "  FragmentOutput result;\n"
              "  result.color = vec4f(prev.rgb, 1.0);\n"
              "  result.blend = prev;\n"
              "  return result;\n}\n";
    break;
  }
  return source;
}

PipelineConfig uber_pipeline_config(const PipelineConfig& config) noexcept {
  PipelineConfig uber = config;
  uber.shaderConfig = {};
  uber.shaderConfig.alphaToCoverage = config.shaderConfig.alphaToCoverage;
  return uber;
}

std::shared_ptr<rhi::RenderPipeline> create_uber_pipeline(const PipelineConfig& config) {
  const auto mode = dst_alpha_mode(config);
  struct Shaders {
    std::once_flag once;
    rhi::ShaderCode vertex, fragment;
  };
  static std::array<Shaders, 3> shaders;
  auto& entry = shaders[static_cast<size_t>(mode)];
  std::call_once(entry.once, [&] {
    const auto source = hlsl::uber_source(mode);
    entry.vertex = gpu::compile_shader(rhi::ShaderStage::Vertex, source, "vs_main");
    entry.fragment = gpu::compile_shader(rhi::ShaderStage::Fragment, source, "fs_main");
  });
  return build_pipeline(config, mode, entry.vertex, entry.fragment);
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
