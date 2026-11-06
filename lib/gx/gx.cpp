#include "gx.hpp"

#include "pipeline.hpp"
#include "../gpu/gpu.hpp"
#include "../internal.hpp"
#include "../gfx/common.hpp"
#include "../gfx/lru_cache.hpp"
#include "../gfx/tex_copy_conv.hpp"
#include "../gfx/tex_palette_conv.hpp"
#include "../gfx/texture.hpp"
#include "../gfx/texture_convert.hpp"
#include "../gfx/texture_replacement.hpp"
#include "gx_fmt.hpp"
#include "recomp.hpp"

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>

#include <atomic>
#include <bit>
#include <cfloat>
#include <cmath>
#include <mutex>
#include <optional>
#include <utility>

static aurora::Module Log("aurora::gx");

namespace aurora::gx {
using gpu::g_device;
using gpu::g_graphicsConfig;

GXState g_gxState{};

static rhi::Sampler* sEmptySampler;
static std::unique_ptr<rhi::Texture> sEmptyTexture;
static std::unique_ptr<rhi::TextureView> sEmptyTextureView;
static std::mutex sBindGroupLayoutMutex;
static std::unique_ptr<rhi::BindGroupLayout> sTextureBindGroupLayout;
static std::unique_ptr<rhi::PipelineLayout> sPipelineLayout;
static std::unique_ptr<rhi::BindGroup> sEmptyTextureBindGroup;
rhi::BindGroup* g_emptyTextureBindGroup = nullptr;

namespace {
struct DynamicPaletteKey {
  const void* sourceIdentity = nullptr;
  u32 width = 0;
  u32 height = 0;
  u32 format = 0;

  bool operator==(const DynamicPaletteKey& rhs) const = default;
  template <typename H>
  friend H AbslHashValue(H h, const DynamicPaletteKey& key) {
    return H::combine(std::move(h), key.sourceIdentity, key.width, key.height, key.format);
  }
};

struct DynamicPaletteEntry {
  gfx::TextureHandle handle;
  u32 sourceRevision = 0;
  u32 tlutDataVersion = 0;
};

struct CachedTlutTextureEntry {
  gfx::TextureHandle handle;
  u32 tlutDataVersion = 0;
};

struct TlutObjectCache {
  CachedTlutTextureEntry tlutTexture;
  absl::flat_hash_map<DynamicPaletteKey, DynamicPaletteEntry> dynamicPaletteTextures;
};

absl::flat_hash_map<u32, TlutObjectCache> s_tlutObjectCaches;
std::atomic_bool s_staticTextureCacheClearPending = false;

// Retail FIFO textures (recomp::resolve_texture_data) have no texture object
// identity. Key them by what the hardware sees and revalidate the contents by
// hash at most once per frame, so animated data still re-uploads.
struct RetailTextureKey {
  const void* data = nullptr;
  const void* tlutData = nullptr;
  u32 width = 0;
  u32 height = 0;
  u32 format = 0;
  u32 mipCount = 0;
  u32 tlutFormat = 0;
  u32 tlutEntries = 0;

  bool operator==(const RetailTextureKey& rhs) const = default;
  template <typename H>
  friend H AbslHashValue(H h, const RetailTextureKey& key) {
    return H::combine(std::move(h), key.data, key.tlutData, key.width, key.height, key.format, key.mipCount,
                      key.tlutFormat, key.tlutEntries);
  }
};

struct RetailTextureEntry {
  gfx::TextureHandle handle;
  HashType hash = 0;
  u32 lastFrame = 0;
};

// Behind it, the textures by contents (from upstream 7af8def): data a new key
// shows (the same texture at another address, as after a heap reload) reuses
// the GPU texture instead of decoding and uploading it again.
struct ContentTextureKey {
  HashType hash = 0;
  u32 width = 0;
  u32 height = 0;
  u32 format = 0;
  u32 mipCount = 0;
  u32 tlutFormat = 0;

  bool operator==(const ContentTextureKey& rhs) const = default;
  template <typename H>
  friend H AbslHashValue(H h, const ContentTextureKey& key) {
    return H::combine(std::move(h), key.hash, key.width, key.height, key.format, key.mipCount, key.tlutFormat);
  }
};

struct ContentTextureEntry {
  gfx::TextureHandle handle;
  u32 lastFrame = 0;
};

constexpr u32 RetailTextureMaxIdleFrames = 600;
absl::flat_hash_map<RetailTextureKey, RetailTextureEntry> s_retailTextures;
absl::flat_hash_map<ContentTextureKey, ContentTextureEntry> s_contentTextures;
u32 s_retailTextureSweepFrame = 0;

template <typename Create>
gfx::TextureHandle resolve_retail_texture(const GXTexObj_& obj, const GXTlutObj_* tlut, Create&& create) {
  const u32 frame = gfx::current_frame();
  if (frame - s_retailTextureSweepFrame >= RetailTextureMaxIdleFrames) {
    s_retailTextureSweepFrame = frame;
    const auto idle = [frame](const auto& it) { return frame - it.second.lastFrame > RetailTextureMaxIdleFrames; };
    absl::erase_if(s_retailTextures, idle);
    absl::erase_if(s_contentTextures, idle);
  }

  const RetailTextureKey key{
      .data = obj.data,
      .tlutData = tlut != nullptr ? tlut->data : nullptr,
      .width = obj.width(),
      .height = obj.height(),
      .format = obj.format(),
      .mipCount = obj.mip_count(),
      .tlutFormat = tlut != nullptr ? static_cast<u32>(tlut->format) : 0,
      .tlutEntries = tlut != nullptr ? tlut->numEntries : 0u,
  };
  auto& entry = s_retailTextures[key];
  if (entry.handle && entry.lastFrame == frame) {
    return entry.handle;
  }
  HashType hash = xxh3_hash_s(obj.data, recomp::texture_byte_size(obj));
  if (tlut != nullptr) {
    hash = xxh3_hash_s(tlut->data, static_cast<size_t>(tlut->numEntries) * sizeof(u16), hash);
  }
  const ContentTextureKey contentKey{
      .hash = hash,
      .width = key.width,
      .height = key.height,
      .format = key.format,
      .mipCount = key.mipCount,
      .tlutFormat = key.tlutFormat,
  };
  if (!entry.handle) {
    auto& content = s_contentTextures[contentKey];
    if (!content.handle) {
      content.handle = create();
    }
    content.lastFrame = frame;
    entry.handle = content.handle;
  } else if (entry.hash != hash) {
    // Data that changes in place (animated) gets a texture of its own, not kept
    // by contents.
    entry.handle = create();
  } else if (const auto it = s_contentTextures.find(contentKey); it != s_contentTextures.end()) {
    it->second.lastFrame = frame;
  }
  entry.hash = hash;
  entry.lastFrame = frame;
  return entry.handle;
}

void do_clear_static_texture_cache() noexcept {
  s_retailTextures.clear();
  s_contentTextures.clear();
}

DynamicPaletteKey make_dynamic_palette_key(const GXTexObj_& obj, const GXState::CopyTextureRef& source) {
  return {
      .sourceIdentity = source.handle.get(),
      .width = obj.width(),
      .height = obj.height(),
      .format = obj.format(),
  };
}

gfx::TextureHandle get_tlut_texture(const GXTlutObj_& tlut) {
  if (tlut.tlutObjId != 0) {
    auto& cache = s_tlutObjectCaches[tlut.tlutObjId];
    if (cache.tlutTexture.handle && cache.tlutTexture.tlutDataVersion == tlut.tlutDataVersion) {
      return cache.tlutTexture.handle;
    }
    cache.dynamicPaletteTextures.clear();
  }

  const auto handle = gfx::new_static_texture_2d(
      tlut.numEntries, 1, 1, gfx::tlut_texture_format(tlut.format),
      {static_cast<const u8*>(tlut.data), static_cast<size_t>(tlut.numEntries) * sizeof(u16)}, true, "Loaded TLUT");
  if (tlut.tlutObjId != 0) {
    auto& cache = s_tlutObjectCaches[tlut.tlutObjId];
    cache.tlutTexture.handle = handle;
    cache.tlutTexture.tlutDataVersion = tlut.tlutDataVersion;
  }
  return handle;
}

gfx::TextureHandle resolve_static_texture(const GXTexObj_& obj) {
  if (s_staticTextureCacheClearPending.exchange(false, std::memory_order_acq_rel)) {
    do_clear_static_texture_cache();
  }

  const auto create = [&obj] {
    if (const auto replacement = gfx::texture_replacement::find_replacement(obj); replacement.has_value()) {
      return *replacement;
    }
#if DEBUG
    const auto name = gfx::texture_replacement::build_texture_replacement_name(obj);
    const auto nameStr = name.c_str();
#else
    const auto nameStr = "GX Static Texture";
#endif
    return gfx::new_static_texture_2d(obj.width(), obj.height(), obj.mip_count(), obj.format(),
                                      {static_cast<const uint8_t*>(obj.data), UINT32_MAX}, false, nameStr);
  };
  return resolve_retail_texture(obj, nullptr, create);
}

gfx::TextureHandle resolve_static_palette_texture(const GXTexObj_& obj, const GXTlutObj_& tlut) {
  if (s_staticTextureCacheClearPending.exchange(false, std::memory_order_acq_rel)) {
    do_clear_static_texture_cache();
  }

  const auto create = [&obj, &tlut]() -> gfx::TextureHandle {
    if (const auto replacement = gfx::texture_replacement::find_replacement(obj, tlut); replacement.has_value()) {
      return *replacement;
    }
    auto converted = gfx::convert_texture_palette(
        obj.format(), obj.width(), obj.height(), obj.mip_count(), {static_cast<const u8*>(obj.data), UINT32_MAX},
        tlut.format, tlut.numEntries, {static_cast<const u8*>(tlut.data), static_cast<size_t>(tlut.numEntries) * 2});
    if (converted.data.empty()) {
      return {};
    }
    auto handle =
        gfx::new_static_texture_2d(obj.width(), obj.height(), obj.mip_count(), GX_TF_RGBA8_PC,
                                   {converted.data.data(), converted.data.size()}, false, "GX Static Palette Texture");
    handle->hasArbitraryMips = converted.hasArbitraryMips;
    return handle;
  };
  return resolve_retail_texture(obj, &tlut, create);
}

gfx::TextureHandle resolve_dynamic_palette_texture(const GXTexObj_& obj, const GXState::CopyTextureRef& source,
                                                   const GXTlutObj_& tlut) {

  const auto tlutHandle = get_tlut_texture(tlut);
  auto& tlutCache = s_tlutObjectCaches[tlut.tlutObjId];
  auto& entry = tlutCache.dynamicPaletteTextures[make_dynamic_palette_key(obj, source)];
  if (!entry.handle) {
    // Use source size instead of target (logical) size
    entry.handle = gfx::new_conv_texture(source.handle->size.width, source.handle->size.height, GX_TF_RGBA8,
                                         "GX Dynamic Palette Texture");
  }
  if (entry.sourceRevision != source.revision || entry.tlutDataVersion != tlut.tlutDataVersion) {
    gfx::queue_palette_conv({
        .variant = obj.format() == GX_TF_C4 ? gfx::tex_palette_conv::Variant::FromFloat4
                                            : gfx::tex_palette_conv::Variant::FromFloat8,
        .src = source.handle,
        .dst = entry.handle,
        .tlut = tlutHandle,
    });
    entry.sourceRevision = source.revision;
    entry.tlutDataVersion = tlut.tlutDataVersion;
  }
  return entry.handle;
}

u32 resolved_format_for_handle(const gfx::TextureHandle& handle) {
  if (!handle) {
    return GX_TF_RGBA8;
  }
  if (handle->gxFormat != gfx::InvalidTextureFormat) {
    return handle->gxFormat;
  }
  return GX_TF_RGBA8_PC;
}
} // namespace

gfx::Viewport map_logical_viewport(const gfx::Viewport& logicalViewport) noexcept {
  const auto [logicalFbWidth, logicalFbHeight] = EfbSize;
  const auto [targetWidth, targetHeight] = gfx::get_render_target_size();
  if (targetWidth == 0 || targetHeight == 0) {
    return logicalViewport;
  }

  const float scaleX = static_cast<float>(targetWidth) / static_cast<float>(logicalFbWidth);
  const float scaleY = static_cast<float>(targetHeight) / static_cast<float>(logicalFbHeight);
  // Edges on whole target pixels, as map_logical_scissor and copies put
  // them: at a fractional scale, a viewport ending inside a pixel leaves it
  // half drawn, and a copy of the region (a downscaled pass for bloom) then
  // spreads whatever it held.
  const float left = std::round(logicalViewport.left * scaleX);
  const float top = std::round(logicalViewport.top * scaleY);
  return {
      .left = left,
      .top = top,
      .width = std::round((logicalViewport.left + logicalViewport.width) * scaleX) - left,
      .height = std::round((logicalViewport.top + logicalViewport.height) * scaleY) - top,
      .znear = logicalViewport.znear,
      .zfar = logicalViewport.zfar,
  };
}

gfx::ClipRect map_logical_scissor(const gfx::ClipRect& logicalScissor) noexcept {
  const auto [logicalFbWidth, logicalFbHeight] = EfbSize;
  const auto [targetWidth, targetHeight] = gfx::get_render_target_size();
  if (targetWidth == 0 || targetHeight == 0) {
    return logicalScissor;
  }

  const float scaleX = static_cast<float>(targetWidth) / static_cast<float>(logicalFbWidth);
  const float scaleY = static_cast<float>(targetHeight) / static_cast<float>(logicalFbHeight);

  const float left = static_cast<float>(logicalScissor.x) * scaleX;
  const float top = static_cast<float>(logicalScissor.y) * scaleY;
  const float right = static_cast<float>(logicalScissor.x + logicalScissor.width) * scaleX;
  const float bottom = static_cast<float>(logicalScissor.y + logicalScissor.height) * scaleY;

  const auto mappedLeft = std::clamp(static_cast<int32_t>(std::round(left)), 0, static_cast<int32_t>(targetWidth));
  const auto mappedTop = std::clamp(static_cast<int32_t>(std::round(top)), 0, static_cast<int32_t>(targetHeight));
  const auto mappedRight =
      std::clamp(static_cast<int32_t>(std::round(right)), mappedLeft, static_cast<int32_t>(targetWidth));
  const auto mappedBottom =
      std::clamp(static_cast<int32_t>(std::round(bottom)), mappedTop, static_cast<int32_t>(targetHeight));

  return {
      .x = mappedLeft,
      .y = mappedTop,
      .width = mappedRight - mappedLeft,
      .height = mappedBottom - mappedTop,
  };
}

void set_logical_viewport(const gfx::Viewport& viewport) noexcept {
  g_gxState.logicalViewport = viewport;
  set_render_viewport(map_logical_viewport(viewport));
}

void set_render_viewport(const gfx::Viewport& viewport) noexcept {
  g_gxState.renderViewport = viewport;
  gfx::set_viewport(viewport);
}

void set_logical_scissor(const gfx::ClipRect& scissor) noexcept {
  g_gxState.logicalScissor = scissor;
  set_render_scissor(map_logical_scissor(g_gxState.logicalScissor));
}

void set_render_scissor(const gfx::ClipRect& scissor) noexcept {
  g_gxState.renderScissor = scissor;
  gfx::set_scissor(scissor);
}

const gfx::TextureBind& get_texture(GXTexMapID id) noexcept { return g_gxState.textures[static_cast<size_t>(id)]; }

// From upstream b922153: copy textures by destination, size and format. The
// current copy at each destination stays pinned (on hardware it is in guest
// memory, and the game may sample one long after making it); one a copy of
// another size or format replaced is kept for reuse within a budget, and goes
// once idle, so a game copying many sizes doesn't grow VRAM forever.
namespace {
void retire_copy_texture(const GXState::CopyTextureRef& copy) {
  // The palette conversions of a retired copy key by its pointer.
  for (auto& [_, cache] : s_tlutObjectCaches) {
    absl::erase_if(cache.dynamicPaletteTextures,
                   [&](const auto& entry) { return entry.first.sourceIdentity == copy.handle.get(); });
  }
}

gfx::LruCache<GXState::CopyTextureKey, GXState::CopyTextureRef> s_copyCache{
    gfx::RenderTextureCacheLimits,
    retire_copy_texture,
};
} // namespace

GXState::CopyTextureRef& resolve_copy_texture(const GXState::CopyTextureKey& key) noexcept {
  auto* copy = s_copyCache.pin(key);
  auto& current = g_gxState.copyTextures[key.dest];
  if (current && current.key != key) {
    s_copyCache.unpin(current.key);
  }
  if (copy == nullptr) {
    auto handle = gfx::tex_copy_conv::needs_conversion(key.format)
                      ? gfx::new_conv_texture(key.width, key.height, key.format, "Copy Conv Texture")
                      // The copy writes alpha 1 itself where the EFB has none (and RGB565's blit).
                      : gfx::new_render_texture(key.width, key.height, GX_TF_RGBA8, "Resolved Texture");
    const u64 bytes = u64{key.width} * key.height * 4;
    copy = &s_copyCache.insert_pinned(key, {.handle = std::move(handle), .key = key}, bytes);
  }
  ++copy->revision;
  current = *copy;
  return current;
}

void end_frame_copy_textures() noexcept { s_copyCache.end_frame(); }

void clear_copy_texture_cache() noexcept {
  g_gxState.copyTextures.clear();
  s_copyCache.clear();
  for (auto& [_, cache] : s_tlutObjectCaches) {
    cache.dynamicPaletteTextures.clear();
  }
}

void clear_static_texture_cache() noexcept { s_staticTextureCacheClearPending.store(true, std::memory_order_release); }

void resolve_sampled_textures(const ShaderInfo& info) noexcept {

  for (u32 i = 0; i < MaxTextures; ++i) {
    if (!info.sampledTextures.test(i)) {
      continue;
    }

    GXTexObj_ obj = g_gxState.loadedTextures[i];
    recomp::resolve_texture_data(obj);
    auto& textureBind = g_gxState.textures[i];

    gfx::TextureHandle handle;
    const auto copyIt = g_gxState.copyTextures.find(obj.data);
    const GXState::CopyTextureRef* copyRef = copyIt != g_gxState.copyTextures.end() ? &copyIt->second : nullptr;
    if (copyRef != nullptr) {
      gfx::note_copy_sampled(copyRef->handle);
    }
    if (is_palette_format(obj.format())) {
      GXTlutObj_ recompTlut{};
      const GXTlutObj_* tlut = recomp::resolve_texture_tlut(i, &recompTlut) ? &recompTlut : nullptr;
      if (tlut != nullptr) {
        if (copyRef != nullptr) {
          handle = resolve_dynamic_palette_texture(obj, *copyRef, *tlut);
        } else if (obj.has_data()) {
          handle = resolve_static_palette_texture(obj, *tlut);
        }
      }
    } else if (copyRef != nullptr) {
      handle = copyRef->handle;
    } else if (obj.has_data()) {
      handle = resolve_static_texture(obj);
    }

    obj.mFormat = resolved_format_for_handle(handle);
    textureBind = gfx::TextureBind{obj, std::move(handle)};
  }
}

static inline rhi::BlendFactor to_blend_factor(GXBlendFactor fac, bool isDst) {
  switch (fac) {
    DEFAULT_FATAL("invalid blend factor {}", underlying(fac));
  case GX_BL_ZERO:
    return rhi::BlendFactor::Zero;
  case GX_BL_ONE:
    return rhi::BlendFactor::One;
  case GX_BL_SRCCLR: // + GX_BL_DSTCLR
    if (isDst) {
      return rhi::BlendFactor::Src;
    } else {
      return rhi::BlendFactor::Dst;
    }
  case GX_BL_INVSRCCLR: // + GX_BL_INVDSTCLR
    if (isDst) {
      return rhi::BlendFactor::OneMinusSrc;
    } else {
      return rhi::BlendFactor::OneMinusDst;
    }
  case GX_BL_SRCALPHA:
    return rhi::BlendFactor::SrcAlpha;
  case GX_BL_INVSRCALPHA:
    return rhi::BlendFactor::OneMinusSrcAlpha;
  case GX_BL_DSTALPHA:
    return rhi::BlendFactor::DstAlpha;
  case GX_BL_INVDSTALPHA:
    return rhi::BlendFactor::OneMinusDstAlpha;
  }
}

static inline rhi::Compare to_compare_function(GXCompare func) {
  switch (func) {
    DEFAULT_FATAL("invalid depth fn {}", underlying(func));
  case GX_NEVER:
    return rhi::Compare::Never;
  case GX_LESS:
    return UseReversedZ ? rhi::Compare::Greater : rhi::Compare::Less;
  case GX_EQUAL:
    return rhi::Compare::Equal;
  case GX_LEQUAL:
    return UseReversedZ ? rhi::Compare::GreaterEqual : rhi::Compare::LessEqual;
  case GX_GREATER:
    return UseReversedZ ? rhi::Compare::Less : rhi::Compare::Greater;
  case GX_NEQUAL:
    return rhi::Compare::NotEqual;
  case GX_GEQUAL:
    return UseReversedZ ? rhi::Compare::LessEqual : rhi::Compare::GreaterEqual;
  case GX_ALWAYS:
    return rhi::Compare::Always;
  }
}

struct BlendState {
  rhi::BlendComponent color;
  rhi::BlendComponent alpha;
};

static inline BlendState to_blend_state(GXBlendMode mode, GXBlendFactor srcFac, GXBlendFactor dstFac, GXLogicOp op,
                                        u32 dstAlpha, bool dualSource) {
  rhi::BlendComponent color;
  switch (mode) {
    DEFAULT_FATAL("unsupported blend mode {}", underlying(mode));
  case GX_BM_NONE:
    color = {rhi::BlendFactor::One, rhi::BlendFactor::Zero, rhi::BlendOp::Add};
    break;
  case GX_BM_BLEND:
    color = {to_blend_factor(srcFac, false), to_blend_factor(dstFac, true), rhi::BlendOp::Add};
    break;
  case GX_BM_SUBTRACT:
    color = {rhi::BlendFactor::One, rhi::BlendFactor::One, rhi::BlendOp::ReverseSubtract};
    break;
  case GX_BM_LOGIC:
    switch (op) {
      DEFAULT_FATAL("unsupported logic op {}", underlying(op));
    case GX_LO_CLEAR:
      color = {rhi::BlendFactor::Zero, rhi::BlendFactor::Zero, rhi::BlendOp::Add};
      break;
    case GX_LO_COPY:
      color = {rhi::BlendFactor::One, rhi::BlendFactor::Zero, rhi::BlendOp::Add};
      break;
    case GX_LO_NOOP:
      color = {rhi::BlendFactor::Zero, rhi::BlendFactor::One, rhi::BlendOp::Add};
      break;
    }
    break;
  }
  if (dualSource) {
    // The source alpha the color blend reads is the TEV's, in the second output.
    const auto remap = [](rhi::BlendFactor factor) {
      return factor == rhi::BlendFactor::SrcAlpha           ? rhi::BlendFactor::Src1Alpha
             : factor == rhi::BlendFactor::OneMinusSrcAlpha ? rhi::BlendFactor::OneMinusSrc1Alpha
                                                            : factor;
    };
    color.src = remap(color.src);
    color.dst = remap(color.dst);
  }
  if (dstAlpha != UINT32_MAX) {
    return {color, {rhi::BlendFactor::Constant, rhi::BlendFactor::Zero, rhi::BlendOp::Add}};
  }
  return {color, color};
}

// GX's front faces are clockwise, the RHI's counter-clockwise.
static inline rhi::CullMode to_cull_mode(GXCullMode mode) {
  switch (mode) {
    DEFAULT_FATAL("unsupported cull mode {}", underlying(mode));
  case GX_CULL_FRONT:
  case GX_CULL_ALL: // render() drops those draws
    return rhi::CullMode::Back;
  case GX_CULL_BACK:
    return rhi::CullMode::Front;
  case GX_CULL_NONE:
    return rhi::CullMode::None;
  }
}

std::shared_ptr<rhi::RenderPipeline> build_pipeline(const PipelineConfig& config, DstAlphaMode dstAlphaMode,
                                                    const rhi::ShaderCode& vertex,
                                                    const rhi::ShaderCode& fragment) noexcept {
  const auto blend = to_blend_state(config.blendMode, config.blendFacSrc, config.blendFacDst, config.blendOp,
                                    config.dstAlpha, dstAlphaMode == DstAlphaMode::DualSource);
  rhi::RenderPipelineDesc desc;
  desc.layout = sPipelineLayout.get();
  desc.vertex = &vertex;
  desc.vertex_entry = "vs_main";
  desc.fragment = &fragment;
  desc.fragment_entry = "fs_main";
  desc.color_format = g_graphicsConfig.surfaceFormat;
  desc.color_write = (config.colorUpdate ? rhi::WriteR | rhi::WriteG | rhi::WriteB : 0) | (config.alphaUpdate ? rhi::WriteA : 0);
  desc.blend = true;
  desc.blend_color = blend.color;
  desc.blend_alpha = blend.alpha;
  desc.depth_format = g_graphicsConfig.depthFormat;
  desc.depth_write = config.depthCompare && config.depthUpdate;
  desc.depth_compare = config.depthCompare ? to_compare_function(config.depthFunc) : rhi::Compare::Always;
  desc.cull = to_cull_mode(config.cullMode);
  desc.samples = config.msaaSamples;
  desc.alpha_to_coverage = config.shaderConfig.alphaToCoverage != 0;
  return g_device->create_render_pipeline(desc);
}

void populate_pipeline_config(PipelineConfig& config, GXPrimitive primitive, GXVtxFmt fmt) noexcept {

  const auto& vtxFmt = g_gxState.vtxFmts[fmt];
  config.shaderConfig.fogType = g_gxState.fog.type;
  config.shaderConfig.fogRangeEnabled = g_gxState.fog.type != GX_FOG_NONE && g_gxState.fog.rangeEnabled;
  u8 vtxOffset = 0;
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    const auto attr = static_cast<GXAttr>(i);
    const auto type = g_gxState.vtxDesc[i];
    auto& mapping = config.shaderConfig.attrs[i];
    if (type == GX_NONE) {
      mapping = {};
      continue;
    }
    const auto& attrFmt = vtxFmt.attrs[i];
    const auto cnt = comp_cnt_count(attr, attrFmt.cnt);
    const bool nbt3 = attr == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3;
    mapping = AttrConfig{
        .attrType = static_cast<u8>(type),
        .cnt = cnt,
        .compType = static_cast<u8>(attrFmt.type),
        .offset = vtxOffset,
        .stride = 0,
        .frac = attrFmt.frac,
        .le = false,
        .nbt3 = nbt3,
    };
    switch (type) {
    case GX_DIRECT: {
      vtxOffset += comp_type_size(attr, attrFmt.type) * cnt;
      break;
    }
    case GX_INDEX8:
      mapping.stride = g_gxState.arrays[i].stride;
      mapping.le = g_gxState.arrays[i].le;
      vtxOffset += nbt3 ? 3 : 1;
      break;
    case GX_INDEX16:
      mapping.stride = g_gxState.arrays[i].stride;
      mapping.le = g_gxState.arrays[i].le;
      vtxOffset += nbt3 ? 6 : 2;
      break;
    default:
      Log.fatal("populate_pipeline_config: Invalid vertex type {}", type);
    }
  }
  config.shaderConfig.vtxStride = vtxOffset;
  if (primitive == GX_LINES) {
    config.shaderConfig.lineMode = 1;
  } else if (primitive == GX_LINESTRIP) {
    config.shaderConfig.lineMode = 2;
  } else if (primitive == GX_POINTS) {
    config.shaderConfig.lineMode = 3;
  } else {
    config.shaderConfig.lineMode = 0;
  }
  config.shaderConfig.tevSwapTable = g_gxState.tevSwapTable;
  for (u8 i = 0; i < g_gxState.numTevStages; ++i) {
    config.shaderConfig.tevStages[i] = g_gxState.tevStages[i];
  }
  config.shaderConfig.tevStageCount = g_gxState.numTevStages;
  for (u8 i = 0; i < g_gxState.numIndStages; ++i) {
    config.shaderConfig.indStages[i] = g_gxState.indStages[i];
  }
  config.shaderConfig.numIndStages = g_gxState.numIndStages;
  for (u8 i = 0; i < MaxColorChannels; ++i) {
    const auto& cc = g_gxState.colorChannelConfig[i];
    if (cc.lightingEnabled) {
      config.shaderConfig.colorChannels[i] = cc;
    } else {
      // Only matSrc matters when lighting disabled
      config.shaderConfig.colorChannels[i] = {
          .matSrc = cc.matSrc,
      };
    }
  }
  for (u8 i = 0; i < g_gxState.numTexGens; ++i) {
    config.shaderConfig.tcgs[i] = g_gxState.tcgs[i];
    if (!g_gxState.dualTexTrans) {
      config.shaderConfig.tcgs[i].postMtx = GX_PTIDENTITY;
      config.shaderConfig.tcgs[i].normalize = false;
    }
  }
  if (g_gxState.alphaCompare) {
    config.shaderConfig.alphaCompare = g_gxState.alphaCompare;
  }
  const auto cullMode = config.shaderConfig.lineMode == 0 ? g_gxState.cullMode : GX_CULL_NONE;
  config = {
      .msaaSamples = gfx::get_sample_count(),
      .shaderConfig = config.shaderConfig,
      .depthFunc = g_gxState.depthFunc,
      .cullMode = cullMode,
      .blendMode = g_gxState.blendMode,
      .blendFacSrc = g_gxState.blendFacSrc,
      .blendFacDst = g_gxState.blendFacDst,
      .blendOp = g_gxState.blendOp,
      // Only an EFB that keeps alpha takes the constant.
      .dstAlpha = g_gxState.alphaUpdate && efb_has_alpha(g_gxState.pixelFmt) ? g_gxState.dstAlpha : UINT32_MAX,
      .depthCompare = g_gxState.depthCompare,
      .depthUpdate = g_gxState.depthUpdate,
      .alphaUpdate = g_gxState.alphaUpdate,
      .colorUpdate = g_gxState.colorUpdate,
  };
  // The coverage alpha would replace the fragment's alpha in an EFB that keeps it.
  const bool efbAlpha = g_gxState.alphaUpdate && g_gxState.pixelFmt == GX_PF_RGBA6_Z24;
  config.shaderConfig.alphaToCoverage = config.msaaSamples > 1 && g_gxState.blendMode == GX_BM_NONE && !efbAlpha &&
                                        alpha_cutout_threshold(g_gxState.alphaCompare) >= 0.f;
}

GXBindGroups build_bind_groups(const ShaderInfo& info) noexcept {
  if (!info.sampledTextures.any() && !info.sampledIndTextures.any()) {
    // Don't bother re-binding anything
    return {};
  }
  std::array<gfx::TextureBinding, MaxTextures> bindings{};
  for (u32 i = 0; i < MaxTextures; ++i) {
    const auto& tex = g_gxState.textures[i];
    if (tex && (info.sampledTextures[i] || info.sampledIndTextures[i])) {
      bindings[i].texture = tex.ref;
      bindings[i].sampler = gfx::sampler_ref(tex.get_descriptor());
    }
  }
  return {
      .textureBindGroup = gfx::bind_group_ref(bindings.data(), MaxTextures, sTextureBindGroupLayout.get(),
                                              sEmptyTextureView.get(), sEmptySampler),
  };
}

void initialize() noexcept {
  std::array<rhi::BindingLayout, MaxTextures * 2> entries{};
  for (u32 i = 0; i < MaxTextures; ++i) {
    entries[i * 2] = {i * 2, rhi::BindingKind::Texture};
    entries[i * 2 + 1] = {i * 2 + 1, rhi::BindingKind::Sampler};
  }
  sTextureBindGroupLayout = g_device->create_bind_group_layout({entries.data(), entries.size()});

  rhi::SamplerDesc emptySampler;
  emptySampler.min = rhi::Filter::Nearest;
  emptySampler.mag = rhi::Filter::Nearest;
  emptySampler.mip = rhi::Filter::Nearest;
  emptySampler.u = rhi::AddressMode::ClampToEdge;
  emptySampler.v = rhi::AddressMode::ClampToEdge;
  sEmptySampler = gfx::sampler_ref(emptySampler);
  sEmptyTexture = g_device->create_texture({1, 1, 1, 1, rhi::Format::RGBA8Unorm, rhi::TextureSampled});
  sEmptyTextureView = sEmptyTexture->create_view({});

  std::array<rhi::BindingValue, MaxTextures * 2> values{};
  for (u32 i = 0; i < MaxTextures; ++i) {
    values[i * 2].binding = i * 2;
    values[i * 2].view = sEmptyTextureView.get();
    values[i * 2 + 1].binding = i * 2 + 1;
    values[i * 2 + 1].sampler = sEmptySampler;
  }
  sEmptyTextureBindGroup = g_device->create_bind_group({sTextureBindGroupLayout.get(), values.data(), values.size()});
  g_emptyTextureBindGroup = sEmptyTextureBindGroup.get();

  rhi::BindGroupLayout* layouts[] = {gfx::g_staticBindGroupLayout, gfx::g_uniformBindGroupLayout,
                                     sTextureBindGroupLayout.get()};
  sPipelineLayout = g_device->create_pipeline_layout({layouts, 3, true});
}

void shutdown() noexcept {
  // TODO we should probably store this all in g_state.gx instead
  g_emptyTextureBindGroup = nullptr;
  sEmptyTextureBindGroup.reset();
  sEmptyTextureView.reset();
  sEmptyTexture.reset();
  sPipelineLayout.reset();
  sTextureBindGroupLayout.reset();
  for (auto& item : g_gxState.textures) {
    item.ref.reset();
  }
  s_tlutObjectCaches.clear();
  g_gxState.loadedTextures.fill({});
  g_gxState.loadedTmemTluts.fill({});
  g_gxState.loadedTextureTlutValid.fill(false);
  g_gxState.loadedTextureTlutTmemOffsets.fill(0);
  g_gxState.loadedTextureTlutFormats.fill(GX_TL_IA8);
  clear_copy_texture_cache();
}
} // namespace aurora::gx

namespace rhi = aurora::rhi;

static rhi::AddressMode to_address_mode(GXTexWrapMode mode) {
  switch (mode) {
    DEFAULT_FATAL("invalid wrap mode {}", underlying(mode));
  case GX_CLAMP:
    return rhi::AddressMode::ClampToEdge;
  case GX_REPEAT:
    return rhi::AddressMode::Repeat;
  case GX_MIRROR:
    return rhi::AddressMode::MirrorRepeat;
  }
}

struct FilterMode {
  rhi::Filter filter;
  rhi::Filter mip;
  bool mips;
};

static FilterMode to_filter_mode(GXTexFilter filter) {
  switch (filter) {
    DEFAULT_FATAL("invalid filter mode {}", static_cast<int>(filter));
  case GX_NEAR:
    return {rhi::Filter::Nearest, rhi::Filter::Nearest, false};
  case GX_LINEAR:
    return {rhi::Filter::Linear, rhi::Filter::Nearest, false};
  case GX_NEAR_MIP_NEAR:
    return {rhi::Filter::Nearest, rhi::Filter::Nearest, true};
  case GX_LIN_MIP_NEAR:
    return {rhi::Filter::Linear, rhi::Filter::Nearest, true};
  case GX_NEAR_MIP_LIN:
    return {rhi::Filter::Nearest, rhi::Filter::Linear, true};
  case GX_LIN_MIP_LIN:
    return {rhi::Filter::Linear, rhi::Filter::Linear, true};
  }
}

static u16 to_anisotropy(GXAnisotropy aniso) {
  switch (aniso) {
    DEFAULT_FATAL("invalid aniso {}", static_cast<int>(aniso));
  case GX_ANISO_1:
  case GX_MAX_ANISOTROPY:
    return 1;
  case GX_ANISO_2:
    return std::max<u16>(aurora::gpu::g_graphicsConfig.textureAnisotropy / 2, 1);
  case GX_ANISO_4:
    return std::max<u16>(aurora::gpu::g_graphicsConfig.textureAnisotropy, 1);
  }
}

rhi::SamplerDesc aurora::gfx::TextureBind::get_descriptor() const noexcept {
  const auto min = to_filter_mode(texObj.min_filter());
  const auto mag = to_filter_mode(texObj.mag_filter());
  rhi::SamplerDesc desc;
  desc.min = min.filter;
  desc.mag = mag.filter;
  desc.mip = min.mip;
  desc.min_lod = texObj.min_lod();
  desc.max_lod = texObj.max_lod();
  u16 maxAnisotropy = aurora::g_config.forceTextureAnisotropy
                          ? std::max<u16>(aurora::gpu::g_graphicsConfig.textureAnisotropy, 1)
                          : to_anisotropy(texObj.max_aniso());
  if (ref && ref->isReplacement) {
    desc.min_lod = 0.f;
    desc.max_lod = 1000.f;
  } else if (!min.mips) {
    desc.min_lod = 0.f;
    desc.max_lod = 0.f;
  }
  if ((ref && ref->hasArbitraryMips) || !min.mips) {
    maxAnisotropy = 1;
  } else if (maxAnisotropy > 1) {
    desc.mag = rhi::Filter::Linear;
    desc.min = rhi::Filter::Linear;
    desc.mip = rhi::Filter::Linear;
  }
  desc.u = to_address_mode(texObj.wrap_s());
  desc.v = to_address_mode(texObj.wrap_t());
  desc.anisotropy = maxAnisotropy;
  return desc;
} // namespace aurora::gx
