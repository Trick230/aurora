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

  const GXState::CopyTextureKey key{
      .dest = dest,
      .width = dstWidth,
      .height = dstHeight,
      .format = texCopyFmt,
  };
  auto it = g_gxState.copyTextureCache.find(key);
  if (it == g_gxState.copyTextureCache.end()) {
    gfx::TextureHandle handle;
    if (gfx::tex_copy_conv::needs_conversion(texCopyFmt)) {
      handle = gfx::new_conv_texture(dstWidth, dstHeight, texCopyFmt, "Copy Conv Texture");
    } else {
      // Configure the texture swizzle to use alpha 1.0 if targeting RGB565 or EFB doesn't have alpha
      const auto fmt = texCopyFmt == GX_TF_RGB565 || g_gxState.pixelFmt == GX_PF_RGB8_Z24 ||
                               g_gxState.pixelFmt == GX_PF_RGB565_Z16
                           ? GX_TF_RGB565
                           : GX_TF_RGBA8;
      handle = gfx::new_render_texture(dstWidth, dstHeight, fmt, "Resolved Texture");
    }
    it = g_gxState.copyTextureCache.emplace(key, GXState::CopyTextureRef{.handle = handle, .revision = 0}).first;
  }
  auto& handle = it->second;

  if (g_gxState.alphaUpdate && g_gxState.dstAlpha != UINT32_MAX) {
    if (!clear) {
      // TODO: figure out the right behavior here.
      // should the copy have a specific alpha value but the EFB remains untouched?
    }
    // Overwrite alpha before resolving
    gfx::push_draw_command(gfx::clear::DrawData{
        .pipeline = gfx::pipeline_ref(gfx::clear::PipelineConfig{
            .clearColor = false,
            .clearAlpha = true,
            .clearDepth = false,
        }),
        .color = wgpu::Color{0.f, 0.f, 0.f, g_gxState.dstAlpha / 255.f},
    });
  }
  const auto clearColor = clear && g_gxState.colorUpdate;
  const auto clearAlpha = clear && g_gxState.alphaUpdate;
  const auto clearDepth = clear && g_gxState.depthUpdate;
  gfx::resolve_pass(handle.handle, rect, clearColor, clearAlpha, clearDepth, g_gxState.clearColor, clear_depth_value(),
                    texCopyFmt);
  ++handle.revision;
  g_gxState.copyTextures[dest] = handle;
}
} // namespace aurora::gx
