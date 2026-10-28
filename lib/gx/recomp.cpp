#include "recomp.hpp"

#include "gx.hpp"

#include <algorithm>
#include <cmath>

namespace aurora::gx::recomp {
namespace {
ResolveGuestAddressFn g_resolve_guest_address = nullptr;
void* g_resolve_guest_user = nullptr;

std::uint32_t texture_physical_base(std::uint32_t reg) noexcept {
  return (reg & 0x00FFFFFFu) << 5;  // 24-bit: reaches Wii MEM2
}

bool retail_texture_has_mips(const GXTexObj_& obj) noexcept {
  switch (GXTexObj_::get_bits(obj.mode0, 3, 5)) {
  case 1:
  case 2:
  case 5:
  case 6:
    return true;
  default:
    return false;
  }
}

// GXGetTexBufferSize takes a level count; mode1 holds the highest LOD index,
// so a mip-filtered texture clamped to LOD 0 still has its base level.
std::uint32_t mip_levels_for_buffer_size(const GXTexObj_& obj) noexcept {
  const auto lod = obj.max_lod();
  if (!std::isfinite(lod) || lod <= 0.f) {
    return 1;
  }
  return std::min<std::uint32_t>(static_cast<std::uint32_t>(lod) + 1, 11u);
}
} // namespace

void set_guest_address_resolver(ResolveGuestAddressFn resolve, void* user) {
  g_resolve_guest_address = resolve;
  g_resolve_guest_user = user;
}

void clear_guest_address_resolver() {
  g_resolve_guest_address = nullptr;
  g_resolve_guest_user = nullptr;
}

bool resolve_guest_address(std::uint32_t address, std::uint32_t size, AddressSpace space, ResourceKind resource,
                           const void** data, std::uint32_t* available) {
  if (data != nullptr) {
    *data = nullptr;
  }
  if (available != nullptr) {
    *available = 0;
  }
  if (g_resolve_guest_address == nullptr || data == nullptr || available == nullptr || size == 0) {
    return false;
  }
  return g_resolve_guest_address(g_resolve_guest_user, address, size, space, resource, data, available);
}

std::uint32_t texture_byte_size(const GXTexObj_& obj) noexcept {
  const auto width = obj.width();
  const auto height = obj.height();
  if (width == 0 || height == 0 || width > UINT16_MAX || height > UINT16_MAX) {
    return 0;
  }
  const bool mips = obj.has_mips() || retail_texture_has_mips(obj);
  return GXGetTexBufferSize(static_cast<u16>(width), static_cast<u16>(height), obj.format(), mips ? GX_TRUE : GX_FALSE,
                            static_cast<u8>(mips ? mip_levels_for_buffer_size(obj) : 0));
}

bool resolve_texture_data(GXTexObj_& obj) noexcept {
  if (obj.has_data()) {
    return true;
  }
  if (obj.image0 == UINT32_MAX) {
    return false;
  }
  const std::uint32_t byteSize = texture_byte_size(obj);
  if (byteSize == 0) {
    return false;
  }
  const std::uint32_t physicalBase = texture_physical_base(obj.image3);
  const void* data = nullptr;
  std::uint32_t available = 0;
  if (!resolve_guest_address(physicalBase, byteSize, AddressSpace::Physical, ResourceKind::Texture, &data,
                             &available)) {
    return false;
  }
  obj.data = data;
  if (retail_texture_has_mips(obj)) {
    obj.flags |= 1u;
  }
  return true;
}

bool resolve_texture_tlut(std::uint32_t textureSlot, GXTlutObj_* out) noexcept {
  if (out == nullptr || textureSlot >= MaxTextures || !g_gxState.loadedTextureTlutValid[textureSlot]) {
    return false;
  }
  const std::uint32_t tmemOffset = g_gxState.loadedTextureTlutTmemOffsets[textureSlot];
  if (tmemOffset >= g_gxState.loadedTmemTluts.size()) {
    return false;
  }
  *out = g_gxState.loadedTmemTluts[tmemOffset];
  if (out->data == nullptr || out->numEntries == 0) {
    return false;
  }
  out->format = g_gxState.loadedTextureTlutFormats[textureSlot];
  return true;
}

} // namespace aurora::gx::recomp

// The SDK's GXGetTexBufferSize: bytes of a texture in GX tile layout, with
// maxLod levels when mips is set.
extern "C" u32 GXGetTexBufferSize(u16 width, u16 height, u32 fmt, GXBool mips, u8 maxLod) {
  s32 shiftX = 0;
  s32 shiftY = 0;
  switch (fmt) {
  case GX_TF_I4:
  case GX_TF_C4:
  case GX_TF_CMPR:
  case GX_CTF_R4:
  case GX_CTF_Z4:
    shiftX = 3;
    shiftY = 3;
    break;
  case GX_TF_I8:
  case GX_TF_IA4:
  case GX_TF_C8:
  case GX_TF_Z8:
  case GX_CTF_RA4:
  case GX_CTF_A8:
  case GX_CTF_R8:
  case GX_CTF_G8:
  case GX_CTF_B8:
  case GX_CTF_Z8M:
  case GX_CTF_Z8L:
    shiftX = 3;
    shiftY = 2;
    break;
  case GX_TF_IA8:
  case GX_TF_RGB565:
  case GX_TF_RGB5A3:
  case GX_TF_RGBA8:
  case GX_TF_C14X2:
  case GX_TF_Z16:
  case GX_TF_Z24X8:
  case GX_CTF_RA8:
  case GX_CTF_RG8:
  case GX_CTF_GB8:
  case GX_CTF_Z16L:
    shiftX = 2;
    shiftY = 2;
    break;
  default:
    break;
  }
  u32 bitSize = fmt == GX_TF_RGBA8 || fmt == GX_TF_Z24X8 ? 64 : 32;
  u32 bufLen = 0;
  if (mips) {
    while (maxLod != 0) {
      const u32 tileX = ((width + (1 << shiftX) - 1) >> shiftX);
      const u32 tileY = ((height + (1 << shiftY) - 1) >> shiftY);
      bufLen += bitSize * tileX * tileY;

      if (width == 1 && height == 1) {
        return bufLen;
      }

      width = (width < 2) ? 1 : width / 2;
      height = (height < 2) ? 1 : height / 2;
      --maxLod;
    };
  } else {
    const u32 tileX = ((width + (1 << shiftX) - 1) >> shiftX);
    const u32 tileY = ((height + (1 << shiftY) - 1) >> shiftY);
    bufLen = bitSize * tileX * tileY;
  }

  return bufLen;
}
