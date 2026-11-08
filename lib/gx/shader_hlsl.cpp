// The GX shader generator in HLSL: the same structure as shader.cpp's WGSL, so that a
// fix to one can be ported to the other line by line. Both compile the same uniform
// bytes: a WGSL matCxR is an HLSL floatRxC, column by column, with M[i] the column in
// WGSL and the row in HLSL, which is why ind_mtx is declared as columns.
#include "../gfx/common.hpp"

#include "../internal.hpp"
#include "gx.hpp"
#include "gx_fmt.hpp"
#include "shader_hlsl.hpp"
#include "shader_info.hpp"

#include <dolphin/gx/GXEnum.h>

#include <string_view>
#include <utility>

namespace aurora::gx::hlsl {
using namespace fmt::literals;
using namespace std::string_literals;
using namespace std::string_view_literals;

static Module Log("aurora::gfx::gx");

static inline std::string_view chan_comp(GXTevColorChan chan) noexcept {
  switch (chan) {
  case GX_CH_RED:
    return "r";
  case GX_CH_GREEN:
    return "g";
  case GX_CH_BLUE:
    return "b";
  case GX_CH_ALPHA:
    return "a";
  default:
    return "?";
  }
}

static bool is_alpha_bump_channel(GXChannelID id) noexcept { return id == GX_ALPHA_BUMP || id == GX_ALPHA_BUMPN; }

static std::string tev_mask_expr(const std::string& value, u32 mask) {
  // t_IndTexCoord is already expanded into the 0..255 indirect sample domain.
  return fmt::format("(f32(u32({}) & 0x{:X}u) / 255.0)", value, mask);
}

static std::string alpha_bump_sel(size_t stageIdx, const ShaderConfig& config, const TevStage& stage) {
  if (stage.indTexStage >= config.numIndStages || stage.indTexAlphaSel == GX_ITBA_OFF) {
    return "0.0";
  }

  std::string baseCoord;
  switch (stage.indTexAlphaSel) {
    DEFAULT_FATAL("invalid indTexAlphaSel {} for stage {}", underlying(stage.indTexAlphaSel), stageIdx);
  case GX_ITBA_S:
    baseCoord = fmt::format("t_IndTexCoord{}.x", underlying(stage.indTexStage));
    break;
  case GX_ITBA_T:
    baseCoord = fmt::format("t_IndTexCoord{}.y", underlying(stage.indTexStage));
    break;
  case GX_ITBA_U:
    baseCoord = fmt::format("t_IndTexCoord{}.z", underlying(stage.indTexStage));
    break;
  case GX_ITBA_OFF:
    return "0.0";
  }

  switch (stage.indTexFormat) {
    DEFAULT_FATAL("invalid indirect format {} for stage {}", underlying(stage.indTexFormat), stageIdx);
  case GX_ITF_8:
    return tev_mask_expr(baseCoord, 0xF8u);
  case GX_ITF_5:
    return tev_mask_expr(baseCoord, 0xE0u);
  case GX_ITF_4:
    return tev_mask_expr(baseCoord, 0xF0u);
  case GX_ITF_3:
    return tev_mask_expr(baseCoord, 0xF8u);
  }
}

static std::string color_arg_reg(GXTevColorArg arg, size_t stageIdx, const ShaderConfig& config,
                                 const TevStage& stage) {
  switch (arg) {
    DEFAULT_FATAL("invalid color arg {}", underlying(arg));
  case GX_CC_CPREV:
    return "prev.rgb";
  case GX_CC_APREV:
    return "vec3f(prev.a)";
  case GX_CC_C0:
    return "tevreg0.rgb";
  case GX_CC_A0:
    return "vec3f(tevreg0.a)";
  case GX_CC_C1:
    return "tevreg1.rgb";
  case GX_CC_A1:
    return "vec3f(tevreg1.a)";
  case GX_CC_C2:
    return "tevreg2.rgb";
  case GX_CC_A2:
    return "vec3f(tevreg2.a)";
  case GX_CC_TEXC: {
    CHECK(stage.texMapId == GX_TEXMAP_NULL || stage.texMapId <= GX_TEXMAP7, "invalid texture {} for stage {}",
          underlying(stage.texMapId), stageIdx);
    const auto& swap = config.tevSwapTable[stage.tevSwapTex];
    return fmt::format("sampled{}.{}{}{}", stageIdx, chan_comp(swap.red), chan_comp(swap.green), chan_comp(swap.blue));
  }
  case GX_CC_TEXA: {
    CHECK(stage.texMapId == GX_TEXMAP_NULL || stage.texMapId <= GX_TEXMAP7, "invalid texture {} for stage {}",
          underlying(stage.texMapId), stageIdx);
    const auto& swap = config.tevSwapTable[stage.tevSwapTex];
    return fmt::format("vec3f(sampled{}.{})", stageIdx, chan_comp(swap.alpha));
  }
  case GX_CC_RASC: {
    // CHECK(stage.channelId != GX_COLOR_NULL, "unmapped color channel for stage {}", stageIdx);
    if (stage.channelId == GX_COLOR_ZERO || stage.channelId == GX_COLOR_NULL) {
      return "vec3f(0.0)";
    }
    if (is_alpha_bump_channel(stage.channelId)) {
      std::string alpha = alpha_bump_sel(stageIdx, config, stage);
      if (stage.channelId == GX_ALPHA_BUMPN) {
        alpha = fmt::format("({} * (255.0 / 248.0))", alpha);
      }
      return fmt::format("vec3f({})", alpha);
    }
    u32 idx = color_channel(stage.channelId);
    const auto& swap = config.tevSwapTable[stage.tevSwapRas];
    return fmt::format("rast{}.{}{}{}", idx, chan_comp(swap.red), chan_comp(swap.green), chan_comp(swap.blue));
  }
  case GX_CC_RASA: {
    // CHECK(stage.channelId != GX_COLOR_NULL, "unmapped color channel for stage {}", stageIdx);
    if (stage.channelId == GX_COLOR_ZERO || stage.channelId == GX_COLOR_NULL) {
      return "vec3f(0.0)";
    }
    if (is_alpha_bump_channel(stage.channelId)) {
      std::string alpha = alpha_bump_sel(stageIdx, config, stage);
      if (stage.channelId == GX_ALPHA_BUMPN) {
        alpha = fmt::format("({} * (255.0 / 248.0))", alpha);
      }
      return fmt::format("vec3f({})", alpha);
    }
    u32 idx = color_channel(stage.channelId);
    const auto& swap = config.tevSwapTable[stage.tevSwapRas];
    return fmt::format("vec3f(rast{}.{})", idx, chan_comp(swap.alpha));
  }
  case GX_CC_ONE:
    return "vec3f(1.0)";
  case GX_CC_HALF:
    return "vec3f(0.5)";
  case GX_CC_KONST: {
    switch (stage.kcSel) {
      DEFAULT_FATAL("invalid kcSel {}", underlying(stage.kcSel));
    case GX_TEV_KCSEL_8_8:
      return "vec3f(1.0)";
    case GX_TEV_KCSEL_7_8:
      return "vec3f(7.0/8.0)";
    case GX_TEV_KCSEL_6_8:
      return "vec3f(6.0/8.0)";
    case GX_TEV_KCSEL_5_8:
      return "vec3f(5.0/8.0)";
    case GX_TEV_KCSEL_4_8:
      return "vec3f(4.0/8.0)";
    case GX_TEV_KCSEL_3_8:
      return "vec3f(3.0/8.0)";
    case GX_TEV_KCSEL_2_8:
      return "vec3f(2.0/8.0)";
    case GX_TEV_KCSEL_1_8:
      return "vec3f(1.0/8.0)";
    case GX_TEV_KCSEL_K0:
      return "ubuf.kcolor0.rgb";
    case GX_TEV_KCSEL_K1:
      return "ubuf.kcolor1.rgb";
    case GX_TEV_KCSEL_K2:
      return "ubuf.kcolor2.rgb";
    case GX_TEV_KCSEL_K3:
      return "ubuf.kcolor3.rgb";
    case GX_TEV_KCSEL_K0_R:
      return "vec3f(ubuf.kcolor0.r)";
    case GX_TEV_KCSEL_K1_R:
      return "vec3f(ubuf.kcolor1.r)";
    case GX_TEV_KCSEL_K2_R:
      return "vec3f(ubuf.kcolor2.r)";
    case GX_TEV_KCSEL_K3_R:
      return "vec3f(ubuf.kcolor3.r)";
    case GX_TEV_KCSEL_K0_G:
      return "vec3f(ubuf.kcolor0.g)";
    case GX_TEV_KCSEL_K1_G:
      return "vec3f(ubuf.kcolor1.g)";
    case GX_TEV_KCSEL_K2_G:
      return "vec3f(ubuf.kcolor2.g)";
    case GX_TEV_KCSEL_K3_G:
      return "vec3f(ubuf.kcolor3.g)";
    case GX_TEV_KCSEL_K0_B:
      return "vec3f(ubuf.kcolor0.b)";
    case GX_TEV_KCSEL_K1_B:
      return "vec3f(ubuf.kcolor1.b)";
    case GX_TEV_KCSEL_K2_B:
      return "vec3f(ubuf.kcolor2.b)";
    case GX_TEV_KCSEL_K3_B:
      return "vec3f(ubuf.kcolor3.b)";
    case GX_TEV_KCSEL_K0_A:
      return "vec3f(ubuf.kcolor0.a)";
    case GX_TEV_KCSEL_K1_A:
      return "vec3f(ubuf.kcolor1.a)";
    case GX_TEV_KCSEL_K2_A:
      return "vec3f(ubuf.kcolor2.a)";
    case GX_TEV_KCSEL_K3_A:
      return "vec3f(ubuf.kcolor3.a)";
    }
  }
  case GX_CC_ZERO:
    return "vec3f(0.0)";
  }
}

static std::string alpha_arg_reg(GXTevAlphaArg arg, size_t stageIdx, const ShaderConfig& config,
                                 const TevStage& stage) {
  switch (arg) {
    DEFAULT_FATAL("invalid alpha arg {}", underlying(arg));
  case GX_CA_APREV:
    return "prev.a";
  case GX_CA_A0:
    return "tevreg0.a";
  case GX_CA_A1:
    return "tevreg1.a";
  case GX_CA_A2:
    return "tevreg2.a";
  case GX_CA_TEXA: {
    CHECK(stage.texMapId == GX_TEXMAP_NULL || stage.texMapId <= GX_TEXMAP7, "invalid texture {} for stage {}",
          underlying(stage.texMapId), stageIdx);
    const auto& swap = config.tevSwapTable[stage.tevSwapTex];
    return fmt::format("sampled{}.{}", stageIdx, chan_comp(swap.alpha));
  }
  case GX_CA_RASA: {
    // CHECK(stage.channelId != GX_COLOR_NULL, "unmapped color channel for stage {}", stageIdx);
    if (stage.channelId == GX_COLOR_ZERO || stage.channelId == GX_COLOR_NULL) {
      return "0.0";
    }
    if (is_alpha_bump_channel(stage.channelId)) {
      std::string alpha = alpha_bump_sel(stageIdx, config, stage);
      if (stage.channelId == GX_ALPHA_BUMPN) {
        alpha = fmt::format("({} * (255.0 / 248.0))", alpha);
      }
      return alpha;
    }
    u32 idx = color_channel(stage.channelId);
    const auto& swap = config.tevSwapTable[stage.tevSwapRas];
    return fmt::format("rast{}.{}", idx, chan_comp(swap.alpha));
  }
  case GX_CA_KONST: {
    switch (stage.kaSel) {
      DEFAULT_FATAL("invalid kaSel {}", underlying(stage.kaSel));
    case GX_TEV_KASEL_8_8:
      return "1.0";
    case GX_TEV_KASEL_7_8:
      return "(7.0/8.0)";
    case GX_TEV_KASEL_6_8:
      return "(6.0/8.0)";
    case GX_TEV_KASEL_5_8:
      return "(5.0/8.0)";
    case GX_TEV_KASEL_4_8:
      return "(4.0/8.0)";
    case GX_TEV_KASEL_3_8:
      return "(3.0/8.0)";
    case GX_TEV_KASEL_2_8:
      return "(2.0/8.0)";
    case GX_TEV_KASEL_1_8:
      return "(1.0/8.0)";
    case GX_TEV_KASEL_K0_R:
      return "ubuf.kcolor0.r";
    case GX_TEV_KASEL_K1_R:
      return "ubuf.kcolor1.r";
    case GX_TEV_KASEL_K2_R:
      return "ubuf.kcolor2.r";
    case GX_TEV_KASEL_K3_R:
      return "ubuf.kcolor3.r";
    case GX_TEV_KASEL_K0_G:
      return "ubuf.kcolor0.g";
    case GX_TEV_KASEL_K1_G:
      return "ubuf.kcolor1.g";
    case GX_TEV_KASEL_K2_G:
      return "ubuf.kcolor2.g";
    case GX_TEV_KASEL_K3_G:
      return "ubuf.kcolor3.g";
    case GX_TEV_KASEL_K0_B:
      return "ubuf.kcolor0.b";
    case GX_TEV_KASEL_K1_B:
      return "ubuf.kcolor1.b";
    case GX_TEV_KASEL_K2_B:
      return "ubuf.kcolor2.b";
    case GX_TEV_KASEL_K3_B:
      return "ubuf.kcolor3.b";
    case GX_TEV_KASEL_K0_A:
      return "ubuf.kcolor0.a";
    case GX_TEV_KASEL_K1_A:
      return "ubuf.kcolor1.a";
    case GX_TEV_KASEL_K2_A:
      return "ubuf.kcolor2.a";
    case GX_TEV_KASEL_K3_A:
      return "ubuf.kcolor3.a";
    }
  }
  case GX_CA_ZERO:
    return "0.0";
  }
}

static std::string tev_op(GXTevOp op, std::string_view bias, std::string_view scale, std::string_view a,
                          std::string_view b, std::string_view c, std::string_view d, std::string_view zero) {
  switch (op) {
    DEFAULT_FATAL("unimplemented tev op {}", underlying(op));
  case GX_TEV_ADD:
  case GX_TEV_SUB: {
    std::string_view neg = op == GX_TEV_SUB ? "-"sv : ""sv;
    return fmt::format("(({0}mix({1}, {2}, {3}) + {4}){5}){6}", neg, a, b, c, d, bias, scale);
  }
  case GX_TEV_COMP_R8_GT:
    return fmt::format("select({3}, {2}, round({0}.r * 255.0) > round({1}.r * 255.0)) + {4}", a, b, c, zero, d);
  case GX_TEV_COMP_R8_EQ:
    return fmt::format("select({3}, {2}, round({0}.r * 255.0) == round({1}.r * 255.0)) + {4}", a, b, c, zero, d);
  case GX_TEV_COMP_GR16_GT:
    return fmt::format(
        "select({3}, {2}, round(dot({0}.rg * 255.0, vec2(1.0, 256.0))) > round(dot({1}.rg * 255.0, vec2(1.0, 256.0))))"
        " + {4}",
        a, b, c, zero, d);
  case GX_TEV_COMP_GR16_EQ:
    return fmt::format(
        "select({3}, {2}, round(dot({0}.rg * 255.0, vec2(1.0, 256.0))) == round(dot({1}.rg * 255.0, vec2(1.0, 256.0))))"
        " + {4}",
        a, b, c, zero, d);
  case GX_TEV_COMP_BGR24_GT:
    return fmt::format(
        "select({3}, {2}, round(dot({0}.rgb * 255.0, vec3(1.0, 256.0, 65536.0))) > round(dot({1}.rgb * 255.0, "
        "vec3(1.0, 256.0, 65536.0)))) + {4}",
        a, b, c, zero, d);
  case GX_TEV_COMP_BGR24_EQ:
    return fmt::format(
        "select({3}, {2}, round(dot({0}.rgb * 255.0, vec3(1.0, 256.0, 65536.0))) == round(dot({1}.rgb * 255.0, "
        "vec3(1.0, 256.0, 65536.0)))) + {4}",
        a, b, c, zero, d);
  case GX_TEV_COMP_RGB8_GT:
    return fmt::format("select({3}, {2}, round({0} * 255.0) > round({1} * 255.0)) + {4}", a, b, c, zero, d);
  case GX_TEV_COMP_RGB8_EQ:
    return fmt::format("select({3}, {2}, round({0} * 255.0) == round({1} * 255.0)) + {4}", a, b, c, zero, d);
  }
}

// a, b and c come overflowed (or known not to need it).
static std::string tev_color_op(GXTevOp op, std::string_view bias, std::string_view scale, bool clamp,
                                std::string_view a, std::string_view b, std::string_view c, std::string_view d) {
  std::string expr = tev_op(op, bias, scale, a, b, c, d, "vec3(0)"sv);
  return clamp ? fmt::format("clamp({}, vec3f(0.0), vec3f(1.0))", expr)
               : fmt::format("clamp({}, vec3f(-4.0), vec3f(4.0))", expr);
}

// The packed comparisons (R8, GR16, BGR24) compare color operands: a and b are
// vec3f.
// a, b and c come overflowed (or known not to need it).
static std::string tev_alpha_op(GXTevOp op, std::string_view bias, std::string_view scale, bool clamp,
                                std::string_view a, std::string_view b, std::string_view c, std::string_view d) {
  std::string expr = tev_op(op, bias, scale, a, b, c, d, "0.0"sv);
  return clamp ? fmt::format("clamp({}, 0.0, 1.0)", expr) : fmt::format("clamp({}, -4.0, 4.0)", expr);
}

static std::string_view tev_bias(GXTevBias bias) {
  switch (bias) {
    DEFAULT_FATAL("invalid tev bias {}", underlying(bias));
  case GX_TB_ZERO:
    return ""sv;
  case GX_TB_ADDHALF:
    return " + 0.5"sv;
  case GX_TB_SUBHALF:
    return " - 0.5"sv;
  }
}

struct AlphaCompareExpr {
  std::string expr;
  int constant = -1;
};

static AlphaCompareExpr alpha_compare_const(bool value) { return {value ? "true"s : "false"s, value ? 1 : 0}; }

static AlphaCompareExpr alpha_compare_not(const AlphaCompareExpr& expr) {
  if (expr.constant != -1) {
    return alpha_compare_const(expr.constant == 0);
  }
  return {fmt::format("!{}", expr.expr), -1};
}

static AlphaCompareExpr alpha_compare_and(const AlphaCompareExpr& lhs, const AlphaCompareExpr& rhs) {
  if (lhs.constant == 0 || rhs.constant == 0) {
    return alpha_compare_const(false);
  }
  if (lhs.constant == 1) {
    return rhs;
  }
  if (rhs.constant == 1) {
    return lhs;
  }
  return {fmt::format("({} && {})", lhs.expr, rhs.expr), -1};
}

static AlphaCompareExpr alpha_compare_or(const AlphaCompareExpr& lhs, const AlphaCompareExpr& rhs) {
  if (lhs.constant == 1 || rhs.constant == 1) {
    return alpha_compare_const(true);
  }
  if (lhs.constant == 0) {
    return rhs;
  }
  if (rhs.constant == 0) {
    return lhs;
  }
  return {fmt::format("({} || {})", lhs.expr, rhs.expr), -1};
}

static AlphaCompareExpr alpha_compare_xor(const AlphaCompareExpr& lhs, const AlphaCompareExpr& rhs) {
  if (lhs.constant != -1 && rhs.constant != -1) {
    return alpha_compare_const(lhs.constant != rhs.constant);
  }
  if (lhs.constant == 0) {
    return rhs;
  }
  if (rhs.constant == 0) {
    return lhs;
  }
  if (lhs.constant == 1) {
    return alpha_compare_not(rhs);
  }
  if (rhs.constant == 1) {
    return alpha_compare_not(lhs);
  }
  return {fmt::format("({} != {})", lhs.expr, rhs.expr), -1};
}

static AlphaCompareExpr alpha_compare_xnor(const AlphaCompareExpr& lhs, const AlphaCompareExpr& rhs) {
  if (lhs.constant != -1 && rhs.constant != -1) {
    return alpha_compare_const(lhs.constant == rhs.constant);
  }
  if (lhs.constant == 0) {
    return alpha_compare_not(rhs);
  }
  if (rhs.constant == 0) {
    return alpha_compare_not(lhs);
  }
  if (lhs.constant == 1) {
    return rhs;
  }
  if (rhs.constant == 1) {
    return lhs;
  }
  return {fmt::format("({} == {})", lhs.expr, rhs.expr), -1};
}

static AlphaCompareExpr alpha_compare(GXCompare comp, u8 ref) {
  const auto iref = static_cast<u32>(ref);
  switch (comp) {
    DEFAULT_FATAL("invalid alpha comp {}", underlying(comp));
  case GX_NEVER:
    return alpha_compare_const(false);
  case GX_LESS:
    if (ref == 0) {
      return alpha_compare_const(false);
    }
    return {fmt::format("(alphaCompare < {}u)", iref), -1};
  case GX_LEQUAL:
    if (ref == 255) {
      return alpha_compare_const(true);
    }
    return {fmt::format("(alphaCompare <= {}u)", iref), -1};
  case GX_EQUAL:
    return {fmt::format("(alphaCompare == {}u)", iref), -1};
  case GX_NEQUAL:
    return {fmt::format("(alphaCompare != {}u)", iref), -1};
  case GX_GEQUAL:
    if (ref == 0) {
      return alpha_compare_const(true);
    }
    return {fmt::format("(alphaCompare >= {}u)", iref), -1};
  case GX_GREATER:
    if (ref == 255) {
      return alpha_compare_const(false);
    }
    return {fmt::format("(alphaCompare > {}u)", iref), -1};
  case GX_ALWAYS:
    return alpha_compare_const(true);
  }
}

static std::string_view tev_scale(GXTevScale scale) {
  switch (scale) {
    DEFAULT_FATAL("invalid tev scale {}", underlying(scale));
  case GX_CS_SCALE_1:
    return ""sv;
  case GX_CS_SCALE_2:
    return " * 2.0"sv;
  case GX_CS_SCALE_4:
    return " * 4.0"sv;
  case GX_CS_DIVIDE_2:
    return " / 2.0"sv;
  }
}

static inline std::string vtx_attr(const ShaderConfig& config, GXAttr attr) {
  const auto type = config.attrs[attr].attrType;
  if (type == GX_NONE) {
    if (attr == GX_VA_PNMTXIDX) {
      return "ubuf.current_pnmtx";
    }
    if (attr == GX_VA_NRM) {
      // Default normal
      return "vec3f(1.0, 0.0, 0.0)"s;
    }
    if (attr == GX_VA_CLR0 || attr == GX_VA_CLR1) {
      return "vec4f(0.0, 0.0, 0.0, 0.0)"s;
    }
    UNLIKELY FATAL("unmapped vtx attr {}", underlying(attr));
  }
  if (attr == GX_VA_POS) {
    return "in_pos"s;
  }
  if (attr == GX_VA_NRM) {
    return "in_nrm"s;
  }
  if (attr == GX_VA_CLR0 || attr == GX_VA_CLR1) {
    const auto idx = attr - GX_VA_CLR0;
    return fmt::format("in_clr{}", idx);
  }
  if (attr >= GX_VA_TEX0 && attr <= GX_VA_TEX7) {
    const auto idx = attr - GX_VA_TEX0;
    return fmt::format("in_tex{}_uv", idx);
  }
  if (attr == GX_VA_PNMTXIDX) {
    return "in_pnmtxidx"s;
  }
  if (attr >= GX_VA_TEX0MTXIDX && attr <= GX_VA_TEX7MTXIDX) {
    const auto idx = attr - GX_VA_TEX0MTXIDX;
    return fmt::format("in_texmtxidx{}", idx);
  }
  UNLIKELY FATAL("unhandled vtx attr {}", underlying(attr));
}

auto fetch_fixed16_attr(std::string_view fetchFn, const AttrConfig& mapping, std::string_view buf, 
                        std::string_view offs, bool le) -> std::string {
  // Some Adreno drivers appear sensitive to generated shaders that route 2- and
  // 3-component fixed-16 vertex attributes through reusable vector fetch helpers.
  // Emitting scalar component fetches at the call site avoids observed artifacts.
  if (mapping.cnt == 2) {
    const auto comp0 = fmt::format("{}_1({}, {} + 0u, {}u, {})", fetchFn, buf, offs, mapping.frac, le);
    const auto comp1 = fmt::format("{}_1({}, {} + 2u, {}u, {})", fetchFn, buf, offs, mapping.frac, le);
    return fmt::format("vec2f({}, {})", comp0, comp1);
  }
  if (mapping.cnt == 3) {
    const auto comp0 = fmt::format("{}_1({}, {} + 0u, {}u, {})", fetchFn, buf, offs, mapping.frac, le);
    const auto comp1 = fmt::format("{}_1({}, {} + 2u, {}u, {})", fetchFn, buf, offs, mapping.frac, le);
    const auto comp2 = fmt::format("{}_1({}, {} + 4u, {}u, {})", fetchFn, buf, offs, mapping.frac, le);
    return fmt::format("vec3f({}, {}, {})", comp0, comp1, comp2);
  }
  return fmt::format("{}_{}({}, {}, {}u, {})", fetchFn, mapping.cnt, buf, offs, mapping.frac, le);
}

auto fetch_attr(const AttrConfig& mapping, std::string_view buf, std::string_view offs, bool le) -> std::string {
  switch (mapping.compType) {
  case GX_U8:
    return fmt::format("fetch_u8_{}({}, {}, {}, {})", mapping.cnt, buf, offs, mapping.frac, le);
  case GX_S8:
    return fmt::format("fetch_s8_{}({}, {}, {}, {})", mapping.cnt, buf, offs, mapping.frac, le);
  case GX_U16:
    return fetch_fixed16_attr("fetch_u16"sv, mapping, buf, offs, le);
  case GX_S16:
    return fetch_fixed16_attr("fetch_s16"sv, mapping, buf, offs, le);
  case GX_F32:
    return fmt::format("fetch_f32_{}({}, {}, {})", mapping.cnt, buf, offs, le);
  case GX_RGBA8:
    return fmt::format("unpack4x8unorm(load_u32_raw({}, {}))", buf, offs);
  default:
    Log.fatal("fetch_attr: Unimplemented {}", static_cast<GXCompType>(mapping.compType));
  }
}

auto fetch_color_attr(const AttrConfig& mapping, std::string_view buf, std::string_view offs, bool le) -> std::string {
  switch (mapping.compType) {
  case GX_RGB565:
    return fmt::format("fetch_rgb565({}, {}, {})", buf, offs, le);
  case GX_RGB8:
    return fmt::format("fetch_rgb8({}, {}, {})", buf, offs, le);
  case GX_RGBX8:
    return fmt::format("fetch_rgbx8({}, {}, {})", buf, offs, le);
  case GX_RGBA4:
    return fmt::format("fetch_rgba4({}, {}, {})", buf, offs, le);
  case GX_RGBA6:
    return fmt::format("fetch_rgba6({}, {}, {})", buf, offs, le);
  case GX_RGBA8:
    return fmt::format("fetch_rgba8({}, {}, {})", buf, offs, le);
  default:
    Log.fatal("fetch_color_attr: Unimplemented {}", static_cast<GXCompType>(mapping.compType));
  }
}

struct AttrAddress {
  std::string offs;
  std::string_view buf;
  bool le;
};

auto attr_address(const AttrConfig& mapping, GXAttr attr, std::string_view vidx, u32 vtxStride, u32 dlExtra, u32 within)
    -> AttrAddress {
  const u32 dlOffset = mapping.offset + dlExtra;
  if (mapping.attrType == GX_INDEX8) {
    return {fmt::format("ARRAY_START({}) + raw_fetch_u8_1(vbuf, vtx_start + {} * {}u + {}u) * {}u + {}u",
                        attr - GX_VA_POS, vidx, vtxStride, dlOffset, mapping.stride, within),
            "abuf"sv, mapping.le};
  }
  if (mapping.attrType == GX_INDEX16) {
    return {
        fmt::format("ARRAY_START({}) + raw_fetch_u16_1(vbuf, vtx_start + {} * {}u + {}u, false) * {}u + {}u",
                    attr - GX_VA_POS, vidx, vtxStride, dlOffset, mapping.stride, within),
        "abuf"sv, mapping.le};
  }
  return {fmt::format("vtx_start + {} * {}u + {}u", vidx, vtxStride, dlOffset + within), "vbuf"sv, false};
}

auto attr_load(const ShaderConfig& config, GXAttr attr, std::string_view vidx) -> std::string {
  const auto& mapping = config.attrs[attr];
  if (mapping.attrType == GX_NONE) {
    return vtx_attr(config, attr);
  }
  const auto [offs, buf, le] = attr_address(mapping, attr, vidx, config.vtxStride, 0u, 0u);
  switch (attr) {
  case GX_VA_PNMTXIDX:
    return fmt::format("(raw_fetch_u8_1({}, {}) / 3u)", buf, offs);
  case GX_VA_TEX0MTXIDX:
  case GX_VA_TEX1MTXIDX:
  case GX_VA_TEX2MTXIDX:
  case GX_VA_TEX3MTXIDX:
  case GX_VA_TEX4MTXIDX:
  case GX_VA_TEX5MTXIDX:
  case GX_VA_TEX6MTXIDX:
  case GX_VA_TEX7MTXIDX:
    return fmt::format("raw_fetch_u8_1({}, {})", buf, offs);
  case GX_VA_POS: {
    const auto posLoad = fetch_attr(mapping, buf, offs, le);
    if (mapping.cnt == 2) {
      return fmt::format("vec3f({}, 0.0)", posLoad);
    }
    return posLoad;
  }
  case GX_VA_NRM:
    // NBT: normal only here; binormal/tangent loaded via attr_load_nbt_slice
    if (mapping.cnt > 3) {
      auto nrmMapping = mapping;
      nrmMapping.cnt = 3;
      return fetch_attr(nrmMapping, buf, offs, le);
    }
    return fetch_attr(mapping, buf, offs, le);
  case GX_VA_CLR0:
  case GX_VA_CLR1:
    return fetch_color_attr(mapping, buf, offs, le);
  case GX_VA_TEX0:
  case GX_VA_TEX1:
  case GX_VA_TEX2:
  case GX_VA_TEX3:
  case GX_VA_TEX4:
  case GX_VA_TEX5:
  case GX_VA_TEX6:
  case GX_VA_TEX7: {
    const auto texLoad = fetch_attr(mapping, buf, offs, le);
    if (mapping.cnt == 1) {
      return fmt::format("vec2f({}, 0.0)", texLoad);
    }
    return texLoad;
  }
  default:
    Log.fatal("attr_load: Unimplemented {}", attr);
  }
}

enum class NbtSlice : u8 {
  N,
  B,
  T,
};

auto attr_load_nbt_slice(const ShaderConfig& config, NbtSlice slice, std::string_view vidx) -> std::string {
  const auto& mapping = config.attrs[GX_VA_NRM];
  if (mapping.attrType == GX_NONE || mapping.cnt != 9) {
    Log.fatal("attr_load_nbt_slice: GX_TG_BINRM/TANGENT requires GX_NRM_NBT or GX_NRM_NBT3");
  }
  const auto sliceIdx = static_cast<u32>(slice);
  const auto compsize = comp_type_size(GX_VA_NRM, static_cast<GXCompType>(mapping.compType));
  u32 dlExtra = 0;
  if (mapping.nbt3) {
    if (mapping.attrType == GX_INDEX8) {
      dlExtra = sliceIdx;
    } else if (mapping.attrType == GX_INDEX16) {
      dlExtra = sliceIdx * 2u;
    }
  }
  const u32 within = sliceIdx * 3u * compsize;
  const auto [offs, buf, le] = attr_address(mapping, GX_VA_NRM, vidx, config.vtxStride, dlExtra, within);
  auto sliceMapping = mapping;
  sliceMapping.cnt = 3;
  return fetch_attr(sliceMapping, buf, offs, le);
}

static constexpr std::string_view nbt_slice_local(NbtSlice slice) noexcept {
  return slice == NbtSlice::B ? "in_binrm" : "in_tangent";
}

static constexpr bool is_emboss_texgen(GXTexGenType type) noexcept {
  return type >= GX_TG_BUMP0 && type <= GX_TG_BUMP7;
}

auto lighting_func(const ShaderConfig& config, const ColorChannelConfig& cc, u8 i, bool alpha) -> std::string {
  std::string_view swizzle = alpha ? ".a"sv : ""sv;
  std::string outVar;
  std::string_view posVar;
  if (UsePerPixelLighting) {
    outVar = fmt::format("rast{}", i);
    posVar = "input.mv_pos"sv;
  } else {
    outVar = fmt::format("output.cc{}", i);
    posVar = "mv_pos"sv;
  }
  std::string ambSrc, matSrc;
  if (cc.ambSrc == GX_SRC_VTX) {
    if (UsePerPixelLighting) {
      ambSrc = fmt::format("input.clr{}", i);
    } else {
      ambSrc = vtx_attr(config, static_cast<GXAttr>(GX_VA_CLR0 + i));
    }
  } else if (cc.ambSrc == GX_SRC_REG) {
    ambSrc = fmt::format("ubuf.cc{0}{1}_amb", i, alpha ? "a"sv : ""sv);
  }
  if (cc.matSrc == GX_SRC_VTX) {
    if (UsePerPixelLighting) {
      matSrc = fmt::format("input.clr{}", i);
    } else {
      matSrc = vtx_attr(config, static_cast<GXAttr>(GX_VA_CLR0 + i));
    }
  } else if (cc.matSrc == GX_SRC_REG) {
    matSrc = fmt::format("ubuf.cc{0}{1}_mat", i, alpha ? "a"sv : ""sv);
  }
  if (!cc.lightingEnabled) {
    return fmt::format("\n    {0}{2} = {1}{2};", outVar, matSrc, swizzle);
  }
  GXDiffuseFn diffFn = cc.diffFn;
  std::string lightAttnFn;
  if (cc.attnFn == GX_AF_NONE) {
    lightAttnFn = "attn = 1.0;"s;
  } else if (cc.attnFn == GX_AF_SPOT) {
    lightAttnFn = fmt::format(R"""(
          float cosine = max(0.0, dot(ldir, light.dir));
          float cos_attn = dot(light.cos_att, vec3f(1.0, cosine, cosine * cosine));
          float dist_attn = dot(light.dist_att, vec3f(1.0, dist, dist2));
          attn = max(0.0, cos_attn / dist_attn);)""");
  } else if (cc.attnFn == GX_AF_SPEC) {
    std::string_view normal = UsePerPixelLighting ? "input.mv_nrm"sv : "mv_nrm"sv;
    std::string dist_attn = diffFn != GX_DF_NONE
                                ? "max(0.0, dot(normalize(light.dist_att), vec3f(1.0, attn, attn * attn)));"
                                : "max(0.0, dot(light.dist_att, vec3f(1.0, attn, attn * attn)));";
    lightAttnFn = fmt::format(R"""(
          attn = select(0.0, max(0.0, dot({0}, light.dir)), dot({0}, ldir) >= 0.0);
          float cos_attn = dot(light.cos_att, vec3f(1.0, attn, attn * attn));
          float dist_attn = {1};
          attn = max(0.0, cos_attn / dist_attn);)""",
                              normal, dist_attn);
  }
  std::string_view lightDiffFn;
  if (diffFn == GX_DF_NONE) {
    lightDiffFn = "1.0"sv;
  } else if (diffFn == GX_DF_SIGN) {
    if (UsePerPixelLighting) {
      lightDiffFn = "dot(ldir, input.mv_nrm)"sv;
    } else {
      lightDiffFn = "dot(ldir, mv_nrm)"sv;
    }
  } else if (diffFn == GX_DF_CLAMP) {
    if (UsePerPixelLighting) {
      lightDiffFn = "max(0.0, dot(ldir, input.mv_nrm))"sv;
    } else {
      lightDiffFn = "max(0.0, dot(ldir, mv_nrm))"sv;
    }
  }
  return fmt::format(R"""(
    {{
      float4 lighting = {5};
      for (uint i = 0u; i < {1}u; i++) {{
          if ((ubuf.lightState{0}{9} & (1u << i)) == 0u) {{ continue; }}
          Light light = ubuf.lights[i];
          float3 ldir = light.pos - {6};
          float dist2 = dot(ldir, ldir);
          float dist = sqrt(dist2);
          ldir = ldir / dist;
          float attn = 0.0;{2}
          float diff = {3};
          lighting = lighting + (attn * diff * light.color);
      }}
      {7}{8} = ({4} * clamp(lighting, vec4f(0.0), vec4f(1.0))){8};
    }})""",
                     i, GX::MaxLights, lightAttnFn, lightDiffFn, matSrc, ambSrc, posVar, outVar, swizzle,
                     alpha ? "a"sv : ""sv);
}

static std::string_view attr_type(GXAttr attr) noexcept {
  if (attr == GX_VA_PNMTXIDX || (attr >= GX_VA_TEX0MTXIDX && attr <= GX_VA_TEX7MTXIDX)) {
    return "uint"sv;
  }
  if (attr == GX_VA_POS || attr == GX_VA_NRM) {
    return "float3"sv;
  }
  if (attr == GX_VA_CLR0 || attr == GX_VA_CLR1) {
    return "float4"sv;
  }
  return "float2"sv;
}

std::string build_shader_source(const ShaderConfig& config, DstAlphaMode dstAlphaMode) noexcept {
  const auto info = build_shader_info(config);
  std::string uniformPre;
  std::string uniBufAttrs;
  std::string texBindings;
  std::string vtxOutAttrs;
  std::string vtxInAttrs;
  std::string vtxXfrAttrsPre;
  std::string vtxXfrAttrs;
  size_t vtxOutIdx = 0;

  // Load points for line/point expansion
  std::string_view vidxAttr = "vidx"sv;
  if (config.lineMode == 0) {
    // Single-instance draws carry their vertex byte offset as firstInstance.
    vtxXfrAttrsPre += "\n    uint vtx_start = draw.vtx_start;";
  } else {
    vtxXfrAttrsPre += "\n    uint vtx_start = ubuf.vtx_start;";
  }
  if (config.lineMode != 0) {
    vtxInAttrs += ",\n    uint iidx : SV_InstanceID";
    uniBufAttrs +=
        "\n    float line_width;"
        "\n    float line_aspect_y;"
        "\n    float line_tex_offset;"
        "\n    uint line_texcoord_mask;";
    if (config.lineMode == 3) {
      // GX_POINTS: each instance = one vertex, expand to quad
      vtxXfrAttrsPre += fmt::format(
          "\n    uint in_vidx = iidx;"
          "\n    float3 in_pos = {};"
          "\n    uint in_pnmtxidx = {};"
          "\n    float3 mv_pos = mul(vec4f(in_pos, 1.0), ubuf.postex_mtx[in_pnmtxidx]);",
          attr_load(config, GX_VA_POS, "in_vidx"sv), attr_load(config, GX_VA_PNMTXIDX, "in_vidx"sv));
    } else {
      // GX_LINES / GX_LINESTRIP: each instance = two vertices, expand to quad
      vtxXfrAttrsPre += fmt::format(
          "\n    bool use_b = vidx >= 2u;"
          "\n    uint vidx_a = iidx * {}u;"
          "\n    uint vidx_b = vidx_a + 1u;"
          "\n    uint in_vidx = select(vidx_a, vidx_b, use_b);"
          "\n    float3 pos_a = {};"
          "\n    float3 pos_b = {};"
          "\n    float3 in_pos = select(pos_a, pos_b, use_b);"
          "\n    uint pnmtxidx_a = {};"
          "\n    uint pnmtxidx_b = {};"
          "\n    uint in_pnmtxidx = select(pnmtxidx_a, pnmtxidx_b, use_b);"
          "\n    float3 mv_pos_a = mul(vec4f(pos_a, 1.0), ubuf.postex_mtx[pnmtxidx_a]);"
          "\n    float3 mv_pos_b = mul(vec4f(pos_b, 1.0), ubuf.postex_mtx[pnmtxidx_b]);"
          "\n    float3 mv_pos = select(mv_pos_a, mv_pos_b, use_b);",
          config.lineMode == 1 ? 2 : 1, attr_load(config, GX_VA_POS, "vidx_a"sv),
          attr_load(config, GX_VA_POS, "vidx_b"sv), attr_load(config, GX_VA_PNMTXIDX, "vidx_a"sv),
          attr_load(config, GX_VA_PNMTXIDX, "vidx_b"sv));
    }
    vidxAttr = "in_vidx"sv;
  } else if (config.attrs[GX_VA_PNMTXIDX].attrType == GX_NONE) {
    vtxXfrAttrsPre += "\n    uint in_pnmtxidx = ubuf.current_pnmtx;";
  }

  // Load vertex attributes
  for (GXAttr attr = GX_VA_PNMTXIDX; attr <= GX_VA_TEX7; attr = static_cast<GXAttr>(attr + 1)) {
    const auto attrType = config.attrs[attr].attrType;
    if (attrType == GX_NONE) {
      continue;
    }
    // in_pnmtxidx and in_pos written above for line mode
    if ((attr != GX_VA_PNMTXIDX && attr != GX_VA_POS) || config.lineMode == 0) {
      vtxXfrAttrsPre += fmt::format("\n    {} {} = {};", attr_type(attr), vtx_attr(config, attr), attr_load(config, attr, vidxAttr));
    }
  }
  bool needsBinrm = false;
  bool needsTangent = false;
  for (int i = 0; i < info.sampledTexCoords.size(); ++i) {
    if (!info.sampledTexCoords.test(i)) {
      continue;
    }
    const bool emboss = is_emboss_texgen(config.tcgs[i].type);
    needsBinrm = needsBinrm || config.tcgs[i].src == GX_TG_BINRM || emboss;
    needsTangent = needsTangent || config.tcgs[i].src == GX_TG_TANGENT || emboss;
  }
  if (needsBinrm) {
    vtxXfrAttrsPre += fmt::format("\n    float3 {} = {};", nbt_slice_local(NbtSlice::B),
                                  attr_load_nbt_slice(config, NbtSlice::B, vidxAttr));
  }
  if (needsTangent) {
    vtxXfrAttrsPre += fmt::format("\n    float3 {} = {};", nbt_slice_local(NbtSlice::T),
                                  attr_load_nbt_slice(config, NbtSlice::T, vidxAttr));
  }

  if (config.lineMode == 0) {
    vtxXfrAttrsPre += fmt::format(
        "\n    float3 mv_pos = mul(vec4f({}, 1.0), ubuf.postex_mtx[in_pnmtxidx]);"
        "\n    output.pos = mul(vec4f(mv_pos, 1.0), ubuf.proj);",
        vtx_attr(config, GX_VA_POS));
  } else if (config.lineMode == 3) {
    // GX_POINTS: expand single vertex to axis-aligned screen-space square
    vtxXfrAttrsPre +=
        "\n    float4 clip_pos = mul(vec4f(mv_pos, 1.0), ubuf.proj);"
        "\n    float2 viewport_scale = ubuf.render_viewport_size / max(ubuf.logical_viewport_size, vec2f(1.0));"
        "\n    float point_size = ubuf.line_width * min(viewport_scale.x, viewport_scale.y);"
        "\n    float x_sign = select(-1.0, 1.0, (vidx & 1u) != 0u);"
        "\n    float y_sign = select(-1.0, 1.0, vidx >= 2u);"
        "\n    float2 offset_px = vec2f(x_sign, y_sign) * (point_size / 2.0);"
        "\n    float2 offset_ndc = (offset_px * 2.0) / ubuf.render_viewport_size;"
        "\n    output.pos = vec4f(clip_pos.xy + offset_ndc * clip_pos.w, clip_pos.zw);";
  } else {
    // GX_LINES / GX_LINESTRIP: expand line segment perpendicular to direction
    vtxXfrAttrsPre +=
        "\n    float4 clip_a = mul(vec4f(mv_pos_a, 1.0), ubuf.proj);"
        "\n    float4 clip_b = mul(vec4f(mv_pos_b, 1.0), ubuf.proj);"
        "\n    float2 ndc_a = clip_a.xy / clip_a.w;"
        "\n    float2 ndc_b = clip_b.xy / clip_b.w;"
        "\n    float2 viewport_scale = ubuf.render_viewport_size / max(ubuf.logical_viewport_size, vec2f(1.0));"
        "\n    float2 delta_px = (ndc_b - ndc_a) / 2.0 * ubuf.render_viewport_size;"
        "\n    float2 dir_px = select(vec2f(1.0, 0.0), normalize(delta_px), dot(delta_px, delta_px) > 1e-10);"
        "\n    float2 perp_px = vec2f(-dir_px.y, dir_px.x);"
        "\n    float line_width = ubuf.line_width * min(viewport_scale.x, viewport_scale.y);"
        "\n    float2 offset_px = perp_px * (line_width / 2.0) * select(-1.0, 1.0, (vidx & 1u) != 0u);"
        "\n    float2 offset_ndc = (offset_px * 2.0) / ubuf.render_viewport_size;"
        "\n    float4 clip_base = select(clip_a, clip_b, use_b);"
        "\n    output.pos = vec4f(clip_base.xy + offset_ndc * clip_base.w, clip_base.zw);";
  }
  // GX clips to -w <= z <= 0. Retail 2D passes put quads exactly on the near
  // plane (s16 z = -1 with the SDK ortho matrix), where rounding in the
  // projection drops them; pull z in by an ulp-sized margin as Dolphin does.
  if constexpr (UseReversedZ) {
    vtxXfrAttrsPre += "\n    output.pos.z = -output.pos.z * (1.0 - 1e-7);";
  } else {
    vtxXfrAttrsPre += "\n    output.pos.z = output.pos.z * (1.0 - 1e-7) + output.pos.w;";
  }
  vtxXfrAttrsPre += fmt::format(
      "\n    float3 nrm_tmp = mul(vec4f({}, 0.0), ubuf.nrm_mtx[in_pnmtxidx]);"
      "\n    float3 mv_nrm = select(nrm_tmp, normalize(nrm_tmp), dot(nrm_tmp, nrm_tmp) > 1e-10);",
      vtx_attr(config, GX_VA_NRM));
  if constexpr (EnableNormalVisualization) {
    vtxOutAttrs += fmt::format("\n    float3 nrm : TEXCOORD{};", vtxOutIdx++);
    vtxXfrAttrsPre += "\n    output.nrm = mv_nrm;";
  }

  uniBufAttrs += "\n    float4x4 proj;";
  uniBufAttrs += fmt::format("\n    float4x3 postex_mtx[{}];", MaxPnMtx + MaxTexMtx);
  uniBufAttrs += fmt::format("\n    float4x3 nrm_mtx[{}];", MaxPnMtx);
  std::string fragmentFnPre;
  std::string fragmentFn;

  static std::array regName{"prev"sv, "tevreg0"sv, "tevreg1"sv, "tevreg2"sv};
  // Whether each register holds 0..1 (a clamped stage wrote it last).
  std::array<bool, MaxTevRegs> colorNormalized{};
  std::array<bool, MaxTevRegs> alphaNormalized{};
  for (u32 idx = 0; idx < config.tevStageCount; ++idx) {
    const auto& stage = config.tevStages[idx];
    const auto color_arg = [&](GXTevColorArg arg) {
      auto value = color_arg_reg(arg, idx, config, stage);
      return tev_color_arg_is_normalized(arg, colorNormalized, alphaNormalized)
                 ? fmt::format("vec3f({})", value)
                 : fmt::format("tev_overflow_vec3f({})", value);
    };
    const auto alpha_arg = [&](GXTevAlphaArg arg) {
      auto value = alpha_arg_reg(arg, idx, config, stage);
      return tev_alpha_arg_is_normalized(arg, alphaNormalized) ? value : fmt::format("tev_overflow_f32({})", value);
    };
    // A packed alpha comparison reads color A and B from before this stage
    // writes its color result.
    const bool alphaCompareColor = tev_alpha_compares_color(stage.alphaOp.op);
    if (alphaCompareColor) {
      fragmentFn += fmt::format("\n    float3 tev_cmp_a{0} = {1};"
                                "\n    float3 tev_cmp_b{0} = {2};",
                                idx, color_arg(stage.colorPass.a), color_arg(stage.colorPass.b));
    }
    {
      std::string_view outReg = regName[stage.colorOp.outReg];
      std::string op = tev_color_op(stage.colorOp.op, tev_bias(stage.colorOp.bias), tev_scale(stage.colorOp.scale),
                                    stage.colorOp.clamp, color_arg(stage.colorPass.a), color_arg(stage.colorPass.b),
                                    color_arg(stage.colorPass.c), color_arg_reg(stage.colorPass.d, idx, config, stage));
      fragmentFn += fmt::format("\n    // TEV stage {2}\n    {0} = vec4f({1}, {0}.a);", outReg, op, idx);
    }
    {
      std::string_view outReg = regName[stage.alphaOp.outReg];
      std::string op = tev_alpha_op(
          stage.alphaOp.op, tev_bias(stage.alphaOp.bias), tev_scale(stage.alphaOp.scale), stage.alphaOp.clamp,
          alphaCompareColor ? fmt::format("tev_cmp_a{}", idx) : alpha_arg(stage.alphaPass.a),
          alphaCompareColor ? fmt::format("tev_cmp_b{}", idx) : alpha_arg(stage.alphaPass.b),
          alpha_arg(stage.alphaPass.c), alpha_arg_reg(stage.alphaPass.d, idx, config, stage));
      fragmentFn += fmt::format("\n    {0}.a = {1};", outReg, op);
    }
    colorNormalized[stage.colorOp.outReg] = stage.colorOp.clamp;
    alphaNormalized[stage.alphaOp.outReg] = stage.alphaOp.clamp;
  }
  bool prevColorNormalized = false;
  bool prevAlphaNormalized = false;

  {
    // GXRuntime recomp patch (0006): zero TEV stages is invalid (SDK state
    // always flushes >=1) but reachable from replayed streams with an
    // incomplete baseline; the unchecked count-1 index fabricated a wild
    // pointer. Warn once and read stage 0.
    if (config.tevStageCount == 0) {
      static bool warnedZeroTevStages = false;
      if (!warnedZeroTevStages) {
        warnedZeroTevStages = true;
        Log.warn("shader config with tevStageCount=0; treating as 1 stage");
      }
    }
    const u32 lastStageIdx = config.tevStageCount != 0 ? config.tevStageCount - 1 : 0;
    const auto& lastStage = config.tevStages[lastStageIdx];
    if (config.tevStageCount != 0) {
      prevColorNormalized = colorNormalized[lastStage.colorOp.outReg];
      prevAlphaNormalized = alphaNormalized[lastStage.alphaOp.outReg];
    }
    if (lastStage.colorOp.outReg != 0) {
      fragmentFn += fmt::format("\n    prev = vec4f({0}.rgb, prev.a);", regName[lastStage.colorOp.outReg]);
    }
    if (lastStage.alphaOp.outReg != 0) {
      fragmentFn += fmt::format("\n    prev.a = {0}.a;", regName[lastStage.alphaOp.outReg]);
    }
  }

  if (info.loadsTevReg.test(0)) {
    uniBufAttrs += "\n    float4 tevprev;";
    fragmentFnPre += "\n    float4 prev = ubuf.tevprev;";
  } else {
    fragmentFnPre += "\n    float4 prev = (float4)0;";
  }
  for (int i = 1 /* Skip TEVPREV */; i < info.loadsTevReg.size(); ++i) {
    if (info.loadsTevReg.test(i)) {
      uniBufAttrs += fmt::format("\n    float4 tevreg{};", i - 1);
      fragmentFnPre += fmt::format("\n    float4 tevreg{0} = ubuf.tevreg{0};", i - 1);
    } else if (info.writesTevReg.test(i)) {
      fragmentFnPre += fmt::format("\n    float4 tevreg{0} = (float4)0;", i - 1);
    }
  }

  if (info.lightingEnabled) {
    uniBufAttrs += fmt::format(FMT_STRING(R"""(
    Light lights[{}];
    uint lightState0;
    uint lightState1;
    uint lightState0a;
    uint lightState1a;)"""),
                               GX::MaxLights);
    uniformPre +=
        "\n"
        "struct Light {\n"
        "    float3 pos;\n"
        "    float3 dir;\n"
        "    float4 color;\n"
        "    float3 cos_att;\n"
        "    float3 dist_att;\n"
        "    float pad; // HLSL would pack what follows the array here\n"
        "};";
    if (UsePerPixelLighting) {
      vtxOutAttrs += fmt::format("\n    float3 mv_pos : TEXCOORD{};", vtxOutIdx++);
      vtxOutAttrs += fmt::format("\n    float3 mv_nrm : TEXCOORD{};", vtxOutIdx++);
      vtxXfrAttrs += fmt::format(FMT_STRING(R"""(
    output.mv_pos = mv_pos;
    output.mv_nrm = mv_nrm;)"""));
    }
  }

  for (int i = 0; i < info.sampledColorChannels.size(); ++i) {
    if (!info.sampledColorChannels.test(i)) {
      continue;
    }

    const auto& cc = config.colorChannels[i];
    const auto& cca = config.colorChannels[i + GX_ALPHA0];
    if (cc.lightingEnabled && cc.ambSrc == GX_SRC_REG) {
      uniBufAttrs += fmt::format("\n    float4 cc{0}_amb;", i);
    }
    if (cc.matSrc == GX_SRC_REG) {
      uniBufAttrs += fmt::format("\n    float4 cc{0}_mat;", i);
    }
    if (cca.lightingEnabled && cca.ambSrc == GX_SRC_REG) {
      uniBufAttrs += fmt::format("\n    float4 cc{0}a_amb;", i);
    }
    if (cca.matSrc == GX_SRC_REG) {
      uniBufAttrs += fmt::format("\n    float4 cc{0}a_mat;", i);
    }

    // Output vertex color if necessary
    if (UsePerPixelLighting) {
      if ((cc.lightingEnabled && cc.ambSrc == GX_SRC_VTX) || cc.matSrc == GX_SRC_VTX ||
          (cca.lightingEnabled && cca.ambSrc == GX_SRC_VTX) || cca.matSrc == GX_SRC_VTX) {
        vtxOutAttrs += fmt::format("\n    float4 clr{1} : TEXCOORD{0};", vtxOutIdx++, i);
        vtxXfrAttrs += fmt::format("\n    output.clr{} = {};", i, vtx_attr(config, static_cast<GXAttr>(GX_VA_CLR0 + i)));
      }
    }

    if (UsePerPixelLighting) {
      fragmentFnPre += fmt::format("\n    float4 rast{} = (float4)0;", i);
      fragmentFnPre += lighting_func(config, cc, i, false);
      fragmentFnPre += lighting_func(config, cca, i, true);
    } else {
      vtxOutAttrs += fmt::format("\n    float4 cc{1} : TEXCOORD{0};", vtxOutIdx++, i);
      vtxXfrAttrs += lighting_func(config, cc, i, false);
      vtxXfrAttrs += lighting_func(config, cca, i, true);
      fragmentFnPre += fmt::format("\n    float4 rast{0} = input.cc{0};", i);
    }
  }
  for (int i = 0; i < info.sampledKColors.size(); ++i) {
    if (info.sampledKColors.test(i)) {
      uniBufAttrs += fmt::format("\n    float4 kcolor{};", i);
    }
  }
  for (int i = 0; i < info.sampledTexCoords.size(); ++i) {
    if (!info.sampledTexCoords.test(i)) {
      continue;
    }
    const auto& tcg = config.tcgs[i];
    if (tcg.type == GX_TG_MTX3x4) {
      vtxOutAttrs += fmt::format("\n    float3 tex{1}_uvw : TEXCOORD{0};", vtxOutIdx++, i);
    } else {
      vtxOutAttrs += fmt::format("\n    float2 tex{1}_uv : TEXCOORD{0};", vtxOutIdx++, i);
    }
    if (is_emboss_texgen(tcg.type)) {
      // Emboss bump: offset the source texcoord by the light projected onto tangent/binormal
      const u32 lightIdx = tcg.type - GX_TG_BUMP0;
      vtxXfrAttrs += fmt::format(
          "\n    float3 bump_ldir{0} = normalize(ubuf.lights[{1}].pos - mv_pos);"
          "\n    float3 bump_tan{0} = mul(vec4f(in_tangent, 0.0), ubuf.nrm_mtx[in_pnmtxidx]);"
          "\n    float3 bump_bin{0} = mul(vec4f(in_binrm, 0.0), ubuf.nrm_mtx[in_pnmtxidx]);"
          "\n    output.tex{0}_uv = tc{2}_proj.xy + vec2f(dot(bump_ldir{0}, bump_tan{0}), dot(bump_ldir{0}, "
          "bump_bin{0}));",
          i, lightIdx, tcg.embossSrc);
      fragmentFnPre += fmt::format("\n    float2 tex{0}_uv = input.tex{0}_uv.xy;", i);
      continue;
    }
    if (tcg.src >= GX_TG_TEX0 && tcg.src <= GX_TG_TEX7) {
      vtxXfrAttrs += fmt::format("\n    float4 tc{} = vec4f({}, 1.0, 1.0);", i,
                                 vtx_attr(config, GXAttr(GX_VA_TEX0 + (tcg.src - GX_TG_TEX0))));
    } else if (tcg.src == GX_TG_POS) {
      vtxXfrAttrs += fmt::format("\n    float4 tc{} = vec4f({}, 1.0);", i, vtx_attr(config, GX_VA_POS));
    } else if (tcg.src == GX_TG_NRM) {
      vtxXfrAttrs += fmt::format("\n    float4 tc{} = vec4f({}, 1.0);", i, vtx_attr(config, GX_VA_NRM));
    } else if (tcg.src == GX_TG_COLOR0 || tcg.src == GX_TG_COLOR1) {
      // XF texgen sees the channel after lighting, not the vertex color
      // (Dolphin does the same); shader_info marks the channel as computed.
      static_assert(!UsePerPixelLighting, "color texgen needs per-vertex channel colors");
      vtxXfrAttrs += fmt::format("\n    float4 tc{} = output.cc{};", i, tcg.src - GX_TG_COLOR0);
    } else if (tcg.src == GX_TG_BINRM) {
      vtxXfrAttrs += fmt::format("\n    float4 tc{} = vec4f({}, 1.0);", i, nbt_slice_local(NbtSlice::B));
    } else if (tcg.src == GX_TG_TANGENT) {
      vtxXfrAttrs += fmt::format("\n    float4 tc{} = vec4f({}, 1.0);", i, nbt_slice_local(NbtSlice::T));
    } else
      UNLIKELY FATAL("unhandled tcg src {}", underlying(tcg.src));
    if (tcg.type == GX_TG_MTX2x4 || tcg.type == GX_TG_MTX3x4) {
      if (info.indexAttr.test(GX_VA_TEX0MTXIDX + i)) {
        vtxXfrAttrs += fmt::format("\n    float3 tc{0}_tmp = mul(tc{0}, ubuf.postex_mtx[in_texmtxidx{0} / 3u]);", i);
      } else if (tcg.mtx == GX_IDENTITY) {
        vtxXfrAttrs += fmt::format("\n    float3 tc{0}_tmp = tc{0}.xyz;", i);
      } else {
        u32 texMtxIdx = (tcg.mtx) / 3;
        vtxXfrAttrs += fmt::format("\n    float3 tc{0}_tmp = mul(tc{0}, ubuf.postex_mtx[{1}]);", i, texMtxIdx);
      }
      if (tcg.type == GX_TG_MTX2x4) {
        vtxXfrAttrs += fmt::format("\n    tc{0}_tmp.z = 1.0f;", i);
      }
    } else if (tcg.type == GX_TG_SRTG) {
      vtxXfrAttrs += fmt::format("\n    float3 tc{0}_tmp = vec3f(tc{0}.xy, 1.0f);", i);
    }
    if (tcg.normalize) {
      vtxXfrAttrs += fmt::format("\n    tc{0}_tmp = normalize(tc{0}_tmp);", i);
    }
    if (tcg.postMtx == GX_PTIDENTITY) {
      vtxXfrAttrs += fmt::format("\n    float3 tc{0}_proj = tc{0}_tmp;", i);
    } else {
      u32 postMtxIdx = (tcg.postMtx - GX_PTTEXMTX0) / 3;
      vtxXfrAttrs +=
          fmt::format("\n    float3 tc{0}_proj = mul(vec4f(tc{0}_tmp.xyz, 1.0), ubuf.postmtx[{1}]);", i, postMtxIdx);
    }
    // Apply line/point tex offset
    if (config.lineMode == 3) {
      // GX_POINTS: offset S for right columns, T for bottom rows
      vtxXfrAttrs += fmt::format(
          "\n    if ((ubuf.line_texcoord_mask & (1u << {0})) != 0u) {{"
          "\n        if ((vidx & 1u) != 0u) {{ tc{0}_proj.x += ubuf.line_tex_offset; }}"
          "\n        if (vidx >= 2u) {{ tc{0}_proj.y += ubuf.line_tex_offset; }}"
          "\n    }}",
          i);
    } else if (config.lineMode != 0) {
      // GX_LINES / GX_LINESTRIP: offset one axis for perpendicular side
      vtxXfrAttrs += fmt::format(
          "\n    if ((ubuf.line_texcoord_mask & (1u << {0})) != 0u && (vidx & 1u) != 0u) {{"
          "\n        tc{0}_proj.y += ubuf.line_tex_offset;"
          "\n    }}",
          i);
    }
    if (tcg.type == GX_TG_MTX3x4) {
      vtxXfrAttrs += fmt::format("\n    output.tex{0}_uvw = tc{0}_proj.xyz;", i);
      fragmentFnPre += fmt::format("\n    float2 tex{0}_uv = input.tex{0}_uvw.xy / input.tex{0}_uvw.z;", i);
    } else {
      vtxXfrAttrs += fmt::format("\n    output.tex{0}_uv = tc{0}_proj.xy;", i);
      fragmentFnPre += fmt::format("\n    float2 tex{0}_uv = input.tex{0}_uv.xy;", i);
    }
  }
  // Multiple TEV stages may reference the same indirect stage,
  // so we sample each indirect texture only once.
  const auto ind_scale = [](const GXIndTexScale s) -> std::string_view {
    switch (s) {
    case GX_ITS_1:
      return "1.0"sv;
    case GX_ITS_2:
      return "(1.0 / 2.0)"sv;
    case GX_ITS_4:
      return "(1.0 / 4.0)"sv;
    case GX_ITS_8:
      return "(1.0 / 8.0)"sv;
    case GX_ITS_16:
      return "(1.0 / 16.0)"sv;
    case GX_ITS_32:
      return "(1.0 / 32.0)"sv;
    case GX_ITS_64:
      return "(1.0 / 64.0)"sv;
    case GX_ITS_128:
      return "(1.0 / 128.0)"sv;
    case GX_ITS_256:
      return "(1.0 / 256.0)"sv;
    default:
      FATAL("unhandled indirect scale {}", underlying(s));
    }
  };
  for (int i = 0; i < info.usedIndStages.size(); ++i) {
    if (!info.usedIndStages.test(i)) {
      continue;
    }
    const auto& indStage = config.indStages[i];
    const u32 texCoordId = underlying(indStage.texCoordId);
    const u32 texMapId = underlying(indStage.texMapId);
    // GX applies the SU texture-coordinate scale before the indirect stage scale.
    // The shader carries normalized UVs, so convert that texel-space result back
    // into normalized coordinates for the indirect texture sample.
    const auto scaleExpr =
        fmt::format("tex{0}_uv * ubuf.texcoord_scale[{0}].xy * vec2f({1}, {2}) / ubuf.tex{3}_size_bias.xy",
                    texCoordId, ind_scale(indStage.scaleS), ind_scale(indStage.scaleT), texMapId);
    fragmentFnPre += fmt::format(
        "\n    // Indirect stage {0}"
        "\n    float3 t_IndTexCoord{0} = 255.0 * textureSampleBias(tex{1}, tex{1}_samp, {2}, "
        "ubuf.tex{1}_size_bias).abg;",
        i, texMapId, scaleExpr);
  }
  if (info.usedIndStages.any()) {
    fragmentFnPre += "\n    float2 t_TexCoord = vec2f(0.0);";
  }
  for (int i = 0; i < config.tevStageCount; ++i) {
    const auto& stage = config.tevStages[i];
    const bool needsIndirectCoord = stage.indTexMtxId != GX_ITM_OFF;
    const bool hasIndirectStage = stage.indTexStage < config.numIndStages;
    const bool needsTevTexCoord =
        needsIndirectCoord || stage.indTexWrapS != GX_ITW_OFF || stage.indTexWrapT != GX_ITW_OFF || stage.indTexAddPrev;
    const bool needsTextureSample = uses_texture_sample(stage);
    if (!needsTevTexCoord && !needsTextureSample) {
      continue;
    }
    const bool hasBaseTexCoord = stage.texCoordId != GX_TEXCOORD_NULL;
    const bool hasBaseTexture = stage.texMapId != GX_TEXMAP_NULL;
    const bool hasBaseCoord = hasBaseTexCoord && hasBaseTexture;
    std::string uvIn;
    if (needsTevTexCoord) {
      fragmentFnPre += fmt::format("\n    // TEV stage {} indirect", i);

      // Apply indirect texture matrix (produces a texel-space offset)
      std::string indirectOffsetTexel;
      if (needsIndirectCoord && hasIndirectStage) {
        std::string_view fmtShift;
        switch (stage.indTexFormat) {
        case GX_ITF_8:
          break;
        case GX_ITF_5:
          fmtShift = " / 8.0"sv;
          break;
        case GX_ITF_4:
          fmtShift = " / 16.0"sv;
          break;
        case GX_ITF_3:
          fmtShift = " / 32.0"sv;
          break;
        default:
          FATAL("unhandled indirect format {}", underlying(stage.indTexFormat));
        }
        if (fmtShift.empty()) {
          fragmentFnPre += fmt::format("\n    float3 ind{0}_coord = t_IndTexCoord{1};", i, underlying(stage.indTexStage));
        } else {
          fragmentFnPre += fmt::format("\n    float3 ind{0}_coord = floor(t_IndTexCoord{1}{2});", i,
                                       underlying(stage.indTexStage), fmtShift);
        }

        if (stage.indTexBiasSel != GX_ITB_NONE) {
          auto bias = stage.indTexFormat == GX_ITF_8 ? "-128.0"sv : "1.0"sv;
          auto biasS = "0.0"sv, biasT = "0.0"sv, biasU = "0.0"sv;
          if (stage.indTexBiasSel == GX_ITB_S || stage.indTexBiasSel == GX_ITB_ST || stage.indTexBiasSel == GX_ITB_SU ||
              stage.indTexBiasSel == GX_ITB_STU) {
            biasS = "1.0"sv;
          }
          if (stage.indTexBiasSel == GX_ITB_T || stage.indTexBiasSel == GX_ITB_ST || stage.indTexBiasSel == GX_ITB_TU ||
              stage.indTexBiasSel == GX_ITB_STU) {
            biasT = "1.0"sv;
          }
          if (stage.indTexBiasSel == GX_ITB_U || stage.indTexBiasSel == GX_ITB_SU || stage.indTexBiasSel == GX_ITB_TU ||
              stage.indTexBiasSel == GX_ITB_STU) {
            biasU = "1.0"sv;
          }
          fragmentFnPre += fmt::format("\n    ind{0}_coord = ind{0}_coord + vec3f({1}, {2}, {3}) * {4};", i, biasS,
                                       biasT, biasU, bias);
        }

        if (stage.indTexMtxId >= GX_ITM_0 && stage.indTexMtxId <= GX_ITM_2) {
          // Static 2x3 matrix: dot(mat_row, vec3(S,T,U)) * scale
          u32 mtxIdx = stage.indTexMtxId - GX_ITM_0;
          fragmentFnPre += fmt::format(
              "\n    float4 ind{0}_c0 = ubuf.ind_mtx[{1}];"
              "\n    float4 ind{0}_c1 = ubuf.ind_mtx[{2}];",
              i, mtxIdx * 2, mtxIdx * 2 + 1);
          indirectOffsetTexel = fmt::format(
              "vec2f("
              "dot(vec3f(ind{0}_c0.xz, ind{0}_c1.x), ind{0}_coord), "
              "dot(vec3f(ind{0}_c0.yw, ind{0}_c1.y), ind{0}_coord)"
              ") * ind{0}_c1.z",
              i);
        } else if (stage.indTexMtxId >= GX_ITM_S0 && stage.indTexMtxId <= GX_ITM_S2 && hasBaseCoord) {
          // Dynamic S: result = scaled texcoord * ind_coord.x * scale / 256
          u32 mtxIdx = stage.indTexMtxId - GX_ITM_S0;
          u32 regTexCoord = underlying(stage.texCoordId);
          indirectOffsetTexel = fmt::format(
              "tex{1}_uv * ubuf.texcoord_scale[{1}].xy * ind{0}_coord.x"
              " * ubuf.ind_mtx[{2}].z / 256.0",
              i, regTexCoord, mtxIdx * 2 + 1);
        } else if (stage.indTexMtxId >= GX_ITM_T0 && stage.indTexMtxId <= GX_ITM_T2 && hasBaseCoord) {
          // Dynamic T: result = scaled texcoord * ind_coord.y * scale / 256
          u32 mtxIdx = stage.indTexMtxId - GX_ITM_T0;
          u32 regTexCoord = underlying(stage.texCoordId);
          indirectOffsetTexel = fmt::format(
              "tex{1}_uv * ubuf.texcoord_scale[{1}].xy * ind{0}_coord.y"
              " * ubuf.ind_mtx[{2}].z / 256.0",
              i, regTexCoord, mtxIdx * 2 + 1);
        }
      }

      // Don't convert to/from texel space if we can avoid it
      const bool useSimpleCoords = stage.indTexMtxId == GX_ITM_OFF && !stage.indTexAddPrev;

      // Wrap base coord and combine with the indirect translation.
      auto wrap_comp = [](GXIndTexWrap wrap, std::string&& coord) -> std::string {
        switch (wrap) {
        case GX_ITW_OFF:
          return std::move(coord);
        case GX_ITW_256:
          return fmt::format("({} % 256.0)", coord);
        case GX_ITW_128:
          return fmt::format("({} % 128.0)", coord);
        case GX_ITW_64:
          return fmt::format("({} % 64.0)", coord);
        case GX_ITW_32:
          return fmt::format("({} % 32.0)", coord);
        case GX_ITW_16:
          return fmt::format("({} % 16.0)", coord);
        case GX_ITW_0:
          return "0.0";
        default:
          FATAL("unhandled indirect wrap {}", underlying(wrap));
        }
      };
      std::string baseCoordExpr;
      if (hasBaseCoord) {
        u32 texCoordId = underlying(stage.texCoordId);
        if (useSimpleCoords) {
          baseCoordExpr = fmt::format("tex{}_uv", texCoordId);
        } else {
          fragmentFnPre +=
              fmt::format("\n    float2 ind{0}_texel = tex{1}_uv * ubuf.texcoord_scale[{1}].xy;", i, texCoordId);
          baseCoordExpr = fmt::format("ind{}_texel", i);
        }
      }
      std::string wrappedExpr = baseCoordExpr;
      if (!baseCoordExpr.empty() && (stage.indTexWrapS != GX_ITW_OFF || stage.indTexWrapT != GX_ITW_OFF)) {
        wrappedExpr = fmt::format("vec2f({}, {})", wrap_comp(stage.indTexWrapS, fmt::format("{}.x", baseCoordExpr)),
                                  wrap_comp(stage.indTexWrapT, fmt::format("{}.y", baseCoordExpr)));
      }

      std::string finalCoord;
      if (!wrappedExpr.empty() && !indirectOffsetTexel.empty()) {
        finalCoord = fmt::format("{} + ({})", wrappedExpr, indirectOffsetTexel);
      } else if (!wrappedExpr.empty()) {
        finalCoord = wrappedExpr;
      } else {
        finalCoord = indirectOffsetTexel;
      }

      if (info.usedIndStages.any() && !finalCoord.empty()) {
        if (stage.indTexAddPrev) {
          fragmentFnPre += fmt::format("\n    t_TexCoord += {};", finalCoord);
        } else {
          fragmentFnPre += fmt::format("\n    t_TexCoord = {};", finalCoord);
        }

        if (needsTextureSample && hasBaseTexture) {
          u32 texMapId = underlying(stage.texMapId);
          if (useSimpleCoords) {
            fragmentFnPre += fmt::format("\n    float2 ind{0}_uv = t_TexCoord;", i);
          } else {
            fragmentFnPre += fmt::format("\n    float2 ind{0}_uv = t_TexCoord / ubuf.tex{1}_size_bias.xy;", i, texMapId);
          }
          uvIn = fmt::format("ind{0}_uv", i);
        }
      }
    }
    if (!needsTextureSample) {
      continue;
    }

    if (stage.texMapId == GX_TEXMAP_NULL) {
      // Tex order disabled: the hardware feeds TEV white, as Dolphin does.
      fragmentFnPre += fmt::format("\n    float4 sampled{} = vec4f(1.0);", i);
      continue;
    }
    CHECK(stage.texCoordId != GX_TEXCOORD_NULL, "unmapped texcoord for stage {}", i);
    if (uvIn.empty()) {
      // No indirect texturing
      uvIn = fmt::format("tex{0}_uv", underlying(stage.texCoordId));
    }
    fragmentFnPre +=
        fmt::format("\n    float4 sampled{0} = textureSampleBias(tex{1}, tex{1}_samp, {2}, ubuf.tex{1}_size_bias);", i,
                    underlying(stage.texMapId), uvIn);
  }
  if (info.usesPTTexMtx.any())
    uniBufAttrs += fmt::format("\n    float4x3 postmtx[{}];", MaxPTTexMtx);
  if (info.usesFog) {
    uniformPre +=
        "\n"
        "struct Fog {\n"
        "    float4 color;\n"
        "    float a;\n"
        "    float b;\n"
        "    float c;\n"
        "    float pad;\n"
        "};";
    uniBufAttrs += "\n    Fog fog;";

    const std::string_view fogDepth = UseReversedZ ? "(1.0 - input.pos.z)" : "input.pos.z";
    if ((config.fogType & 0x08) != 0) {
      fragmentFn += fmt::format("\n    // Orthographic fog\n    float fogBase = ubuf.fog.a * {};", fogDepth);
    } else {
      fragmentFn +=
          fmt::format("\n    // Perspective fog\n    float fogBase = ubuf.fog.a / (ubuf.fog.b - {});", fogDepth);
    }
    if (config.fogRangeEnabled) {
      fragmentFn += "\n    fogBase *= asfloat(abuf.Load((ubuf.fog_range_base + u32(input.pos.x)) * 4u));";
    }
    fragmentFn += "\n    float fogF = clamp(fogBase - ubuf.fog.c, 0.0, 1.0);";
    switch (config.fogType) {
      DEFAULT_FATAL("invalid fog type {}", config.fogType);
    case GX_FOG_PERSP_LIN:
    case GX_FOG_ORTHO_LIN:
      fragmentFn += "\n    float fogZ = fogF;";
      break;
    case GX_FOG_PERSP_EXP:
    case GX_FOG_ORTHO_EXP:
      fragmentFn += "\n    float fogZ = 1.0 - exp2(-8.0 * fogF);";
      break;
    case GX_FOG_PERSP_EXP2:
    case GX_FOG_ORTHO_EXP2:
      fragmentFn += "\n    float fogZ = 1.0 - exp2(-8.0 * fogF * fogF);";
      break;
    case GX_FOG_PERSP_REVEXP:
    case GX_FOG_ORTHO_REVEXP:
      fragmentFn += "\n    float fogZ = exp2(-8.0 * (1.0 - fogF));";
      break;
    case GX_FOG_PERSP_REVEXP2:
    case GX_FOG_ORTHO_REVEXP2:
      fragmentFn +=
          "\n    fogF = 1.0 - fogF;"
          "\n    float fogZ = exp2(-8.0 * fogF * fogF);";
      break;
    }
    fragmentFn += "\n    prev = vec4f(mix(prev.rgb, ubuf.fog.color.rgb, clamp(fogZ, 0.0, 1.0)), prev.a);";
  }
  uniBufAttrs += fmt::format("\n    float4 texcoord_scale[{}];", MaxTexCoord);
  if (info.usedIndTexMtxs.any()) {
    uniBufAttrs += "\n    float4 ind_mtx[6];";
  }
  for (int i = 0; i < info.sampledTextures.size(); ++i) {
    if (!info.sampledTextures.test(i)) {
      continue;
    }
    uniBufAttrs += fmt::format("\n    float4 tex{}_size_bias;", i);
    texBindings += fmt::format(
        "\nTexture2D tex{0} : register(t{1}, space2);"
        "\nSamplerState tex{0}_samp : register(s{2}, space2);",
        i, i * 2, i * 2 + 1);
  }
  if (!prevColorNormalized && !prevAlphaNormalized) {
    fragmentFn += "\n    prev = tev_overflow_vec4f(prev);";
  } else if (!prevColorNormalized) {
    fragmentFn += "\n    prev = vec4f(tev_overflow_vec3f(prev.rgb), prev.a);";
  } else if (!prevAlphaNormalized) {
    fragmentFn += "\n    prev.a = tev_overflow_f32(prev.a);";
  }
  if (config.alphaCompare) {
    const auto comp0 = alpha_compare(config.alphaCompare.comp0, config.alphaCompare.ref0);
    const auto comp1 = alpha_compare(config.alphaCompare.comp1, config.alphaCompare.ref1);
    AlphaCompareExpr pass;
    switch (config.alphaCompare.op) {
      DEFAULT_FATAL("invalid alpha compare op {}", underlying(config.alphaCompare.op));
    case GX_AOP_AND:
      pass = alpha_compare_and(comp0, comp1);
      break;
    case GX_AOP_OR:
      pass = alpha_compare_or(comp0, comp1);
      break;
    case GX_AOP_XOR:
      pass = alpha_compare_xor(comp0, comp1);
      break;
    case GX_AOP_XNOR:
      pass = alpha_compare_xnor(comp0, comp1);
      break;
    }
    const auto discard = alpha_compare_not(pass);
    if (config.alphaToCoverage) {
      // Coverage from how far alpha is past the edge, in pixels: the cutout's
      // edge spread over about one pixel's samples.
      fragmentFn += fmt::format("\n    // Alpha to coverage"
                                "\n    prev.a = clamp((prev.a - {:.6f}) / max(fwidth(prev.a), 0.0001) + 0.5, 0.0, 1.0);",
                                alpha_cutout_threshold(config.alphaCompare));
    } else if (discard.constant == 1) {
      fragmentFn += "\n    // Alpha compare\n    discard;";
    } else if (discard.constant != 0) {
      fragmentFn +=
          "\n    // Alpha compare"
          "\n    uint alphaCompare = u32(round(clamp(prev.a, 0.0, 1.0) * 255.0));";
      fragmentFn += fmt::format("\n    if ({}) {{ discard; }}", discard.expr);
    }
  }
  if constexpr (EnableNormalVisualization) {
    fragmentFn += "\n    prev = vec4f(input.nrm, prev.a);";
  }
  std::string_view fragmentOutput = "float4"sv;
  std::string_view fragmentSemantic = " : SV_Target"sv;
  std::string_view fragmentReturn = "prev"sv;
  if (dstAlphaMode == DstAlphaMode::Replace) {
    // The blend constant is the EFB's alpha.
    fragmentFn += "\n    prev.a = 1.0;";
  } else if (dstAlphaMode == DstAlphaMode::DualSource) {
    uniformPre +=
        "\nstruct FragmentOutput {\n"
        "    [[vk::location(0)]] [[vk::index(0)]] float4 color : SV_Target0;\n"
        "    [[vk::location(0)]] [[vk::index(1)]] float4 blend : SV_Target1;\n"
        "};\n"
        "FragmentOutput make_output(float4 prev) {\n"
        "    FragmentOutput result;\n"
        "    result.color = float4(prev.rgb, 1.0);\n"
        "    result.blend = prev;\n"
        "    return result;\n"
        "}";
    fragmentOutput = "FragmentOutput"sv;
    fragmentSemantic = ""sv;
    fragmentReturn = "make_output(prev)"sv;
  }

  auto shaderSource = fmt::format(R"""(
{11}
{8}

struct Uniform {{
    uint vtx_start;
    uint current_pnmtx;
    float2 render_viewport_size;
    float2 logical_viewport_size;
    uint fog_range_base;
    uint pad;
    uint4 array_start[3];{0}
}};
ByteAddressBuffer vbuf : register(t0, space0);
ByteAddressBuffer abuf : register(t1, space0);
ConstantBuffer<Uniform> ubuf : register(b0, space1);{1}

struct VertexOutput {{
    float4 pos : SV_Position;{2}
}};

VertexOutput vs_main(
    uint vidx : SV_VertexID{3}
) {{
    VertexOutput output = (VertexOutput)0;{7}{4}
    return output;
}}

{9} fs_main(VertexOutput input){12} {{{6}{5}
    return {10};
}}
)""",
                                        uniBufAttrs, texBindings, vtxOutAttrs, vtxInAttrs, vtxXfrAttrs, fragmentFn,
                                        fragmentFnPre, vtxXfrAttrsPre, uniformPre, fragmentOutput, fragmentReturn,
                                        shader_helpers(), fragmentSemantic);
  return shaderSource;
}

std::string shader_helpers() noexcept {
  return R"""(
#define f32 float
#define u32 uint
#define mix lerp
// A texture's sample: size_bias.z its LOD bias, size_bias.w the axes (bit 0 s,
// bit 1 t) that repeat in the shader on a clamped sampler (see
// edge_clamped_axes): coordinates within the texture stay clamped, so a 2D
// sprite drawn larger than 1:1 never blends in its far side at its edges.
#define textureSampleBias(tex, samp, uv, size_bias) sample_texture(tex, samp, uv, size_bias)
float4 sample_texture(Texture2D tex, SamplerState samp, float2 uv, float4 size_bias) {
  if (size_bias.w == 0.0) {
    return tex.SampleBias(samp, uv, size_bias.z);
  }
  const float2 repeats = float2(fmod(size_bias.w, 2.0), floor(size_bias.w / 2.0));
  const float2 outside = float2(uv.x < 0.0 || uv.x > 1.0, uv.y < 0.0 || uv.y > 1.0);
  const float2 wrapped = lerp(uv, uv - floor(uv), repeats * outside);
  const float scale = exp2(size_bias.z);
  return tex.SampleGrad(samp, wrapped, ddx(uv) * scale, ddy(uv) * scale);
}
// WGSL's order of arguments, on HLSL 2021's select.
#define select(if_false, if_true, condition) select((condition), (if_true), (if_false))
#define extractBits(value, offset, bits) (((value) >> (offset)) & ((1u << (bits)) - 1u))
#define ARRAY_START(i) ubuf.array_start[(i) / 4u][(i) % 4u]

struct DrawConstants {
  uint vtx_start;
};
[[vk::push_constant]] ConstantBuffer<DrawConstants> draw : register(b0, space3);

float2 vec2f(float x) { return float2(x, x); }
float2 vec2f(float2 v) { return v; }
float2 vec2f(uint2 v) { return float2(v); }
float2 vec2f(int2 v) { return float2(v); }
float2 vec2f(float x, float y) { return float2(x, y); }
float2 vec2(float x, float y) { return float2(x, y); }
float3 vec3f(float x) { return float3(x, x, x); }
float3 vec3f(float3 v) { return v; }
float3 vec3f(uint3 v) { return float3(v); }
float3 vec3f(int3 v) { return float3(v); }
float3 vec3f(float x, float y, float z) { return float3(x, y, z); }
float3 vec3f(float2 v, float z) { return float3(v, z); }
float3 vec3(float x) { return float3(x, x, x); }
float3 vec3(float x, float y, float z) { return float3(x, y, z); }
float4 vec4f(float x) { return float4(x, x, x, x); }
float4 vec4f(float4 v) { return v; }
float4 vec4f(uint4 v) { return float4(v); }
float4 vec4f(int4 v) { return float4(v); }
float4 vec4f(float x, float y, float z, float w) { return float4(x, y, z, w); }
float4 vec4f(float3 v, float w) { return float4(v, w); }
float4 vec4f(float2 v, float z, float w) { return float4(v, z, w); }
float4 vec4f(float2 a, float2 b) { return float4(a, b); }

uint bswap32(uint v, bool le) {
  if (le) {
    return v;
  }
  return ((v & 0x000000FFu) << 24u) |
         ((v & 0x0000FF00u) << 8u) |
         ((v & 0x00FF0000u) >> 8u) |
         ((v & 0xFF000000u) >> 24u);
}

uint bswap16(uint v, bool le) {
  return le ? v : (((v & 0xFFu) << 8u) | (v >> 8u));
}

uint load_word(ByteAddressBuffer p, uint word_idx) {
  return p.Load(word_idx * 4u);
}

uint load_u8(ByteAddressBuffer p, uint byte_off) {
  uint word = load_word(p, byte_off / 4u);
  uint shift = (byte_off & 3u) * 8u;
  return (word >> shift) & 0xFFu;
}

uint load_u32_raw(ByteAddressBuffer p, uint byte_off) {
  uint word_idx = byte_off >> 2u;
  uint sub = byte_off & 3u;
  uint lo = load_word(p, word_idx);
  if (sub == 0u) {
    return lo;
  }
  uint hi = load_word(p, word_idx + 1u);
  uint shift = sub * 8u;
  return (lo >> shift) | (hi << (32u - shift));
}

uint load_u16(ByteAddressBuffer p, uint byte_off, bool le) {
  uint word_idx = byte_off >> 2u;
  uint sub = byte_off & 3u;
  uint word = load_word(p, word_idx);
  if (sub <= 2u) {
    return bswap16(extractBits(word, sub * 8u, 16u), le);
  }
  uint next = load_word(p, word_idx + 1u);
  uint raw = extractBits(word, 24u, 8u) | (extractBits(next, 0u, 8u) << 8u);
  return bswap16(raw, le);
}

uint load_u24(ByteAddressBuffer p, uint byte_off, bool le) {
  uint raw = load_u32_raw(p, byte_off) & 0x00FFFFFFu;
  if (le) {
    return raw;
  }
  return ((raw & 0x0000FFu) << 16u) |
         (raw & 0x00FF00u) |
         ((raw & 0xFF0000u) >> 16u);
}

uint load_u32(ByteAddressBuffer p, uint byte_off, bool le) {
  return bswap32(load_u32_raw(p, byte_off), le);
}

float load_f32(ByteAddressBuffer p, uint byte_off, bool le) {
  return asfloat(load_u32(p, byte_off, le));
}

float4 unpack4x8unorm(uint v) {
  return float4(v & 0xFFu, (v >> 8u) & 0xFFu, (v >> 16u) & 0xFFu, v >> 24u) / 255.0;
}

uint raw_fetch_u8_1(ByteAddressBuffer p, uint byte_off) {
  return load_u8(p, byte_off);
}

uint2 raw_fetch_u8_2(ByteAddressBuffer p, uint byte_off) {
  uint word_idx = byte_off >> 2u;
  uint sub = byte_off & 3u;
  uint word = load_word(p, word_idx);
  if (sub <= 2u) {
    uint shift = sub * 8u;
    return uint2(extractBits(word, shift + 0u, 8u), extractBits(word, shift + 8u, 8u));
  }
  uint next = load_word(p, word_idx + 1u);
  return uint2(extractBits(word, 24u, 8u), extractBits(next, 0u, 8u));
}

uint3 raw_fetch_u8_3(ByteAddressBuffer p, uint byte_off) {
  uint raw = load_u32_raw(p, byte_off);
  return uint3(extractBits(raw, 0u, 8u), extractBits(raw, 8u, 8u), extractBits(raw, 16u, 8u));
}

uint4 raw_fetch_u8_4(ByteAddressBuffer p, uint byte_off) {
  uint raw = load_u32_raw(p, byte_off);
  return uint4(extractBits(raw, 0u, 8u), extractBits(raw, 8u, 8u), extractBits(raw, 16u, 8u), raw >> 24u);
}

uint raw_fetch_u16_1(ByteAddressBuffer p, uint byte_off, bool le) {
  return load_u16(p, byte_off, le);
}

uint2 raw_fetch_u16_2(ByteAddressBuffer p, uint byte_off, bool le) {
  return uint2(load_u16(p, byte_off + 0u, le), load_u16(p, byte_off + 2u, le));
}

uint3 raw_fetch_u16_3(ByteAddressBuffer p, uint byte_off, bool le) {
  return uint3(load_u16(p, byte_off + 0u, le), load_u16(p, byte_off + 2u, le), load_u16(p, byte_off + 4u, le));
}

uint4 raw_fetch_u16_4(ByteAddressBuffer p, uint byte_off, bool le) {
  return uint4(load_u16(p, byte_off + 0u, le), load_u16(p, byte_off + 2u, le), load_u16(p, byte_off + 4u, le),
               load_u16(p, byte_off + 6u, le));
}

float raw_fetch_f32_1(ByteAddressBuffer p, uint byte_off, bool le) {
  return load_f32(p, byte_off, le);
}

float2 raw_fetch_f32_2(ByteAddressBuffer p, uint byte_off, bool le) {
  return float2(load_f32(p, byte_off + 0u, le), load_f32(p, byte_off + 4u, le));
}

float3 raw_fetch_f32_3(ByteAddressBuffer p, uint byte_off, bool le) {
  return float3(load_f32(p, byte_off + 0u, le), load_f32(p, byte_off + 4u, le), load_f32(p, byte_off + 8u, le));
}

float4 raw_fetch_f32_4(ByteAddressBuffer p, uint byte_off, bool le) {
  return float4(load_f32(p, byte_off + 0u, le), load_f32(p, byte_off + 4u, le), load_f32(p, byte_off + 8u, le),
                load_f32(p, byte_off + 12u, le));
}

float fetch_u8_1(ByteAddressBuffer p, uint byte_off, uint fraction, bool le) {
  return float(raw_fetch_u8_1(p, byte_off)) / float(1u << fraction);
}

float fetch_s8_1(ByteAddressBuffer p, uint byte_off, uint fraction, bool le) {
  int v = asint(raw_fetch_u8_1(p, byte_off) << 24u) >> 24;
  return float(v) / float(1u << fraction);
}

float2 fetch_u8_2(ByteAddressBuffer p, uint byte_off, uint fraction, bool le) {
  return float2(raw_fetch_u8_2(p, byte_off)) / float(1u << fraction);
}

float2 fetch_s8_2(ByteAddressBuffer p, uint byte_off, uint fraction, bool le) {
  int2 v = asint(raw_fetch_u8_2(p, byte_off) << 24u) >> 24;
  return float2(v) / float(1u << fraction);
}

float3 fetch_u8_3(ByteAddressBuffer p, uint byte_off, uint fraction, bool le) {
  return float3(raw_fetch_u8_3(p, byte_off)) / float(1u << fraction);
}

float3 fetch_s8_3(ByteAddressBuffer p, uint byte_off, uint fraction, bool le) {
  int3 v = asint(raw_fetch_u8_3(p, byte_off) << 24u) >> 24;
  return float3(v) / float(1u << fraction);
}

float4 fetch_u8_4(ByteAddressBuffer p, uint byte_off, uint fraction, bool le) {
  return float4(raw_fetch_u8_4(p, byte_off)) / float(1u << fraction);
}

float4 fetch_s8_4(ByteAddressBuffer p, uint byte_off, uint fraction, bool le) {
  int4 v = asint(raw_fetch_u8_4(p, byte_off) << 24u) >> 24;
  return float4(v) / float(1u << fraction);
}

float fetch_u16_1(ByteAddressBuffer p, uint byte_off, uint fraction, bool le) {
  return float(raw_fetch_u16_1(p, byte_off, le)) / float(1u << fraction);
}

float fetch_s16_1(ByteAddressBuffer p, uint byte_off, uint fraction, bool le) {
  int v = asint(raw_fetch_u16_1(p, byte_off, le) << 16u) >> 16;
  return float(v) / float(1u << fraction);
}

float2 fetch_u16_2(ByteAddressBuffer p, uint byte_off, uint fraction, bool le) {
  return float2(raw_fetch_u16_2(p, byte_off, le)) / float(1u << fraction);
}

float2 fetch_s16_2(ByteAddressBuffer p, uint byte_off, uint fraction, bool le) {
  int2 v = asint(raw_fetch_u16_2(p, byte_off, le) << 16u) >> 16;
  return float2(v) / float(1u << fraction);
}

float3 fetch_u16_3(ByteAddressBuffer p, uint byte_off, uint fraction, bool le) {
  return float3(raw_fetch_u16_3(p, byte_off, le)) / float(1u << fraction);
}

float3 fetch_s16_3(ByteAddressBuffer p, uint byte_off, uint fraction, bool le) {
  int3 v = asint(raw_fetch_u16_3(p, byte_off, le) << 16u) >> 16;
  return float3(v) / float(1u << fraction);
}

float4 fetch_u16_4(ByteAddressBuffer p, uint byte_off, uint fraction, bool le) {
  return float4(raw_fetch_u16_4(p, byte_off, le)) / float(1u << fraction);
}

float4 fetch_s16_4(ByteAddressBuffer p, uint byte_off, uint fraction, bool le) {
  int4 v = asint(raw_fetch_u16_4(p, byte_off, le) << 16u) >> 16;
  return float4(v) / float(1u << fraction);
}

float fetch_f32_1(ByteAddressBuffer p, uint byte_off, bool le) {
  return raw_fetch_f32_1(p, byte_off, le);
}

float2 fetch_f32_2(ByteAddressBuffer p, uint byte_off, bool le) {
  return raw_fetch_f32_2(p, byte_off, le);
}

float3 fetch_f32_3(ByteAddressBuffer p, uint byte_off, bool le) {
  return raw_fetch_f32_3(p, byte_off, le);
}

float4 fetch_f32_4(ByteAddressBuffer p, uint byte_off, bool le) {
  return raw_fetch_f32_4(p, byte_off, le);
}

float4 fetch_rgb565(ByteAddressBuffer p, uint byte_off, bool le) {
  uint v = load_u16(p, byte_off, le);
  return float4(float((v >> 11u) & 0x1Fu) / 31.0, float((v >> 5u) & 0x3Fu) / 63.0, float(v & 0x1Fu) / 31.0, 1.0);
}

float4 fetch_rgb8(ByteAddressBuffer p, uint byte_off, bool le) {
  uint3 v = raw_fetch_u8_3(p, byte_off);
  return float4(float(v.x), float(v.y), float(v.z), 255.0) / 255.0;
}

float4 fetch_rgbx8(ByteAddressBuffer p, uint byte_off, bool le) {
  uint4 v = raw_fetch_u8_4(p, byte_off);
  return float4(float(v.x), float(v.y), float(v.z), 255.0) / 255.0;
}

float4 fetch_rgba4(ByteAddressBuffer p, uint byte_off, bool le) {
  uint v = load_u16(p, byte_off, le);
  return float4(float((v >> 12u) & 0x0Fu), float((v >> 8u) & 0x0Fu), float((v >> 4u) & 0x0Fu), float(v & 0x0Fu)) / 15.0;
}

float4 fetch_rgba6(ByteAddressBuffer p, uint byte_off, bool le) {
  uint v = load_u24(p, byte_off, le);
  return float4(float((v >> 18u) & 0x3Fu), float((v >> 12u) & 0x3Fu), float((v >> 6u) & 0x3Fu), float(v & 0x3Fu)) / 63.0;
}

float4 fetch_rgba8(ByteAddressBuffer p, uint byte_off, bool le) {
  return float4(raw_fetch_u8_4(p, byte_off)) / 255.0;
}

float tev_overflow_f32(float value) {
  float byte_space = value * 255.0;
  return (byte_space - floor(byte_space / 256.0) * 256.0) / 255.0;
}

float3 tev_overflow_vec3f(float3 value) {
  float3 byte_space = value * 255.0;
  return (byte_space - floor(byte_space / 256.0) * 256.0) / 255.0;
}

float4 tev_overflow_vec4f(float4 value) {
  float4 byte_space = value * 255.0;
  return (byte_space - floor(byte_space / 256.0) * 256.0) / 255.0;
}
)""";
}
} // namespace aurora::gx::hlsl
