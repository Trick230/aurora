#include "texture.hpp"

#include "../internal.hpp"

#include <algorithm>
#include <cstdint>


namespace aurora::gfx {
namespace {
Module Log("aurora::gfx");

constexpr u32 div_ceil(u32 value, u32 divisor) noexcept { return (value + divisor - 1) / divisor; }
} // namespace

TextureFormatInfo format_info(rhi::Format format) noexcept {
  switch (format) {
    DEFAULT_FATAL("unimplemented texture format {}", static_cast<int>(format));
  case rhi::Format::R8Unorm:
    return {1, 1, 1, false};
  case rhi::Format::RG8Unorm:
  case rhi::Format::R16Sint:
    return {1, 1, 2, false};
  case rhi::Format::RGBA8Unorm:
  case rhi::Format::BGRA8Unorm:
  case rhi::Format::R32Float:
    return {1, 1, 4, false};
  case rhi::Format::BC1:
    return {4, 4, 8, true};
  case rhi::Format::BC3:
  case rhi::Format::BC5:
  case rhi::Format::BC7:
    return {4, 4, 16, true};
  }
}

uint64_t calc_texture_size(rhi::Format format, u32 width, u32 height, u32 mips) noexcept {
  const auto info = format_info(format);
  uint64_t total = 0;
  for (uint32_t mip = 0; mip < mips; ++mip) {
    const uint32_t mipWidth = std::max(width >> mip, 1u);
    const uint32_t mipHeight = std::max(height >> mip, 1u);
    const uint64_t widthBlocks = div_ceil(mipWidth, info.blockWidth);
    const uint64_t heightBlocks = div_ceil(mipHeight, info.blockHeight);
    const uint64_t mipBytes = widthBlocks * heightBlocks * info.blockSize;
    total += mipBytes;
  }
  return total;
}

bool is_block_aligned(rhi::Format format, uint32_t width, uint32_t height) noexcept {
  if (width == 0 || height == 0) {
    return false;
  }

  const auto info = format_info(format);
  return !info.compressed || (width % info.blockWidth == 0 && height % info.blockHeight == 0);
}
} // namespace aurora::gfx
