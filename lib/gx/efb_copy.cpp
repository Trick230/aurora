// EFB to texture copies: a copy trigger in the stream resolves the EFB into
// a texture the guest address identifies, for later draws to sample.
#include "gx.hpp"

#include "../gfx/clear.hpp"
#include "../gfx/tex_copy_conv.hpp"
#include "../gfx/texture.hpp"

#include <algorithm>
#include <cmath>

namespace aurora::gx {
namespace {
// The copy's size at the render target's scale.
aurora::Vec2<uint32_t> scale_copy_dst(u32 logicalWidth, u32 logicalHeight) {
  const auto [logicalFbWidth, logicalFbHeight] = EfbSize;
  const auto [targetWidth, targetHeight] = gfx::get_render_target_size();
  if (targetWidth == 0 || targetHeight == 0) {
    return {logicalWidth, logicalHeight};
  }

  const float scaleX = static_cast<float>(targetWidth) / static_cast<float>(logicalFbWidth);
  const float scaleY = static_cast<float>(targetHeight) / static_cast<float>(logicalFbHeight);
  const auto scaledWidth = std::max<u32>(static_cast<u32>(std::lround(static_cast<float>(logicalWidth) * scaleX)), 1);
  const auto scaledHeight = std::max<u32>(static_cast<u32>(std::lround(static_cast<float>(logicalHeight) * scaleY)), 1);
  return {scaledWidth, scaledHeight};
}
} // namespace

void copy_tex(const void* dest, GXBool clear) noexcept {
  const auto rect = map_logical_scissor(g_gxState.texCopySrc);
  const auto [dstWidth, dstHeight] = scale_copy_dst(g_gxState.texCopyDstWidth, g_gxState.texCopyDstHeight);
  const auto texCopyFmt = g_gxState.texCopyFmt;

  const auto& copy = resolve_copy_texture({
      .dest = dest,
      .width = dstWidth,
      .height = dstHeight,
      .format = texCopyFmt,
  });

  const bool efbAlpha = efb_has_alpha(g_gxState.pixelFmt);
  const auto clearColor = clear && g_gxState.colorUpdate;
  const auto clearAlpha = clear && g_gxState.alphaUpdate && efbAlpha;
  const auto clearDepth = clear && g_gxState.depthUpdate;
  gfx::resolve_pass(copy.handle, rect, clearColor, clearAlpha, clearDepth, g_gxState.clearColor, clear_depth_value(),
                    texCopyFmt, !efbAlpha);
}
} // namespace aurora::gx
