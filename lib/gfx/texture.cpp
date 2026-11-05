#include "common.hpp"

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "aurora/aurora.h"
#include "texture.hpp"
#include "texture_convert.hpp"
#include "../gx/gx_fmt.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

#include <fmt/format.h>

namespace aurora::gfx {
using webgpu::g_device;

namespace {
Module Log("aurora::gfx");

Extent3D physical_size(Extent3D size, TextureFormatInfo info) {
  const uint32_t width = ((size.width + info.blockWidth - 1) / info.blockWidth) * info.blockWidth;
  const uint32_t height = ((size.height + info.blockHeight - 1) / info.blockHeight) * info.blockHeight;
  return {.width = width, .height = height, .depthOrArrayLayers = size.depthOrArrayLayers};
}

rhi::TextureViewDesc view_desc(u32 format) noexcept {
  rhi::TextureViewDesc desc;
  switch (format) {
  case GX_TF_R8_PC:
    desc.r = desc.g = desc.b = desc.a = rhi::Swizzle::R;
    break;
  case GX_TF_RG8_PC:
    desc.r = desc.g = desc.b = rhi::Swizzle::R;
    desc.a = rhi::Swizzle::G;
    break;
  default:
    break;
  }
  return desc;
}

void upload_mips(TextureRef& ref, ArrayRef<uint8_t> data, const char* what) {
  uint32_t offset = 0;
  for (uint32_t mip = 0; mip < ref.mipCount; ++mip) {
    const Extent3D mipSize{
        .width = std::max(ref.size.width >> mip, 1u),
        .height = std::max(ref.size.height >> mip, 1u),
        .depthOrArrayLayers = ref.size.depthOrArrayLayers,
    };
    const auto info = format_info(ref.format);
    const auto physicalSize = physical_size(mipSize, info);
    const uint32_t widthBlocks = physicalSize.width / info.blockWidth;
    const uint32_t heightBlocks = physicalSize.height / info.blockHeight;
    const uint32_t bytesPerRow = widthBlocks * info.blockSize;
    const uint32_t dataSize = bytesPerRow * heightBlocks * mipSize.depthOrArrayLayers;
    CHECK(offset + dataSize <= data.size(), "{}: expected at least {} bytes, got {}", what, offset + dataSize,
          data.size());
    queue_texture_upload_data(data.data() + offset, bytesPerRow, heightBlocks, ref.texture, mip, physicalSize);
    offset += dataSize;
  }
  if (data.size() != UINT32_MAX && offset < data.size()) {
    Log.warn("{}: texture used {} bytes, but given {} bytes", what, offset, data.size());
  }
}

TextureHandle new_target(uint32_t width, uint32_t height, rhi::Format format, u32 gxFormat) {
  std::shared_ptr<rhi::Texture> texture =
      g_device->create_texture({width, height, 1, 1, format, rhi::TextureSampled | rhi::TextureColorTarget});
  std::shared_ptr<rhi::TextureView> view = texture->create_view({});
  return std::make_shared<TextureRef>(std::move(texture), std::move(view), Extent3D{width, height, 1}, format, 1,
                                      gxFormat);
}
} // namespace

TextureHandle new_static_texture_2d(uint32_t width, uint32_t height, uint32_t mips, u32 format, ArrayRef<uint8_t> data,
                                    bool tlut, const char* label) noexcept {
  auto handle = new_dynamic_texture_2d(width, height, mips, format, label);
  auto& ref = *handle;

  ConvertedTexture converted;
  if (ref.gxFormat != InvalidTextureFormat) {
    if (tlut) {
      CHECK(ref.size.height == 1, "new_static_texture_2d[{}]: expected tlut height 1, got {}", label, ref.size.height);
      CHECK(ref.mipCount == 1, "new_static_texture_2d[{}]: expected tlut mipCount 1, got {}", label, ref.mipCount);
      converted = convert_tlut(ref.gxFormat, ref.size.width, data);
    } else {
      converted = convert_texture(ref.gxFormat, ref.size.width, ref.size.height, ref.mipCount, data);
    }
    if (!converted.data.empty()) {
      data = converted.data;
      ref.hasArbitraryMips = converted.hasArbitraryMips;
    }
  }
  upload_mips(ref, data, label);
  return handle;
}

TextureHandle new_dynamic_texture_2d(uint32_t width, uint32_t height, uint32_t mips, u32 gxFormat,
                                     const char* label) noexcept {
  (void)label;
  const auto format = to_rhi(gxFormat);
  std::shared_ptr<rhi::Texture> texture = g_device->create_texture({width, height, mips, 1, format, rhi::TextureSampled});
  std::shared_ptr<rhi::TextureView> view = texture->create_view(view_desc(gxFormat));
  return std::make_shared<TextureRef>(std::move(texture), std::move(view), Extent3D{width, height, 1}, format, mips,
                                      gxFormat);
}

TextureHandle new_render_texture(uint32_t width, uint32_t height, u32 gxFormat, const char* label) noexcept {
  (void)label;
  return new_target(width, height, webgpu::g_graphicsConfig.surfaceFormat, gxFormat);
}

TextureHandle new_conv_texture(uint32_t width, uint32_t height, u32 gxFormat, const char* label) noexcept {
  (void)label;
  return new_target(width, height, to_rhi(gxFormat), gxFormat);
}

void write_texture(TextureRef& ref, ArrayRef<uint8_t> data) noexcept {
  ConvertedTexture converted;
  if (ref.gxFormat != InvalidTextureFormat) {
    converted = convert_texture(ref.gxFormat, ref.size.width, ref.size.height, ref.mipCount, data);
    ref.hasArbitraryMips = converted.hasArbitraryMips;
    if (!converted.data.empty()) {
      data = converted.data;
    }
  }
  upload_mips(ref, data, "write_texture");
}
} // namespace aurora::gfx
