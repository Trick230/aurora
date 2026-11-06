#include "common.hpp"

#include "ambient_occlusion.hpp"
#include "clear.hpp"
#include "../internal.hpp"
#include "../gpu/gpu.hpp"
#include "../gx/pipeline.hpp"
#include "frame_interp.hpp"
#include "pipeline_cache.hpp"
#include "render_worker.hpp"
#include "tex_copy_conv.hpp"
#include "tex_palette_conv.hpp"
#include "texture_replacement.hpp"
#include "texture.hpp"
#include "../window.hpp"

#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <optional>
#include <ranges>
#include <string>
#include <thread>

#include <absl/container/flat_hash_map.h>
#include <magic_enum.hpp>


namespace aurora::gfx {
static Module Log("aurora::gfx");

using gpu::g_device;


constexpr uint64_t StagingBufferSize = UniformBufferSize + VertexBufferSize + IndexBufferSize + StorageBufferSize +
                                       (UseTextureBuffer ? TextureUploadSize : 0);
constexpr size_t FrameSlotCount = 2;
constexpr size_t StagingBufferCount = FrameSlotCount + 3;

struct StagingHighWater {
  uint32_t verts = 0;
  uint32_t uniforms = 0;
  uint32_t indices = 0;
  uint32_t storage = 0;
  uint32_t textureUpload = 0;
  size_t textureUploadCount = 0;
};

struct ShaderDrawCommand {
  ShaderType type;
  union {
    clear::DrawData clear;
    gx::DrawData gx;
  };
};
enum class CommandType {
  SetViewport,
  SetScissor,
  Draw,
};
struct Command {
  CommandType type;
  union Data {
    Viewport setViewport;
    ClipRect setScissor;
    ShaderDrawCommand draw;
  } data;
};
namespace {
struct CachedBindGroup {
  std::unique_ptr<rhi::BindGroup> bindGroup;
  // A bind group doesn't keep what it shows alive.
  std::array<TextureHandle, gx::MaxTextures> textures;
  uint32_t lastUsedFrame = 0;
};

constexpr uint32_t BindGroupCacheRetainFrames = 32;
constexpr uint32_t BindGroupCacheSweepPeriod = 16;
} // namespace

static absl::flat_hash_map<BindGroupRef, CachedBindGroup> g_cachedBindGroups;
static absl::flat_hash_map<SamplerRef, std::unique_ptr<rhi::Sampler>> g_cachedSamplers;
static std::mutex g_bindGroupCacheMutex;
static std::mutex g_samplerCacheMutex;

static std::unique_ptr<rhi::Buffer> g_vertexBuffer;
static std::unique_ptr<rhi::Buffer> g_uniformBufferOwner;
rhi::Buffer* g_uniformBuffer = nullptr;
static std::unique_ptr<rhi::Buffer> g_indexBuffer;
static std::unique_ptr<rhi::Buffer> g_storageBuffer;
// What a frame writes on the CPU, and the list it records. The slot's ticket is
// that list's last submission: both can be used again once it's done.
static std::array<std::unique_ptr<rhi::Buffer>, StagingBufferCount> g_stagingBuffers;
static std::array<std::unique_ptr<rhi::CommandList>, StagingBufferCount> g_stagingLists;
static std::array<std::atomic<uint64_t>, StagingBufferCount> g_stagingTickets;
static uint32_t g_frameIndex = UINT32_MAX;
static PipelineRef g_currentPipeline;
// Texture group bound at slot 2 (0: the empty one) and whether the index
// buffer is bound, so draws only re-set what changed.
static BindGroupRef g_currentTextureBindGroup;
static bool g_indexBufferBound;
// Whether the groups every GX draw shares are bound on the current pipeline layout.
static bool g_gxGroupsBound;
static bool g_viewportEmpty;
static std::unique_ptr<rhi::BindGroupLayout> g_staticBindGroupLayoutOwner;
rhi::BindGroupLayout* g_staticBindGroupLayout = nullptr;
static std::unique_ptr<rhi::BindGroup> g_staticBindGroup;
static std::unique_ptr<rhi::BindGroupLayout> g_uniformBindGroupLayoutOwner;
rhi::BindGroupLayout* g_uniformBindGroupLayout = nullptr;
static std::unique_ptr<rhi::BindGroup> g_uniformBindGroup;

// for imgui debug
AuroraStats g_stats{};
uint32_t g_drawCallCount = 0;
uint32_t g_mergedDrawCallCount = 0;
uint32_t g_droppedDrawCallCount = 0;

using CommandList = std::vector<Command>;

struct RenderPass {
  std::shared_ptr<rhi::Texture> colorTexture;
  std::shared_ptr<rhi::TextureView> colorView;
  std::shared_ptr<rhi::Texture> resolveTexture; // MSAA resolve target; null if msaaSamples == 1
  std::shared_ptr<rhi::Texture> depthTexture;
  std::shared_ptr<rhi::Texture> copySourceTexture;
  std::shared_ptr<rhi::TextureView> copySourceView;
  std::shared_ptr<rhi::TextureView> copySourceDepthView;
  Extent3D targetSize;
  uint32_t msaaSamples = 1;

  TextureHandle resolveTarget;
  GXTexFmt resolveFormat = GX_TF_RGBA8;
  bool resolveOpaqueAlpha = false;
  ClipRect resolveRect;
  Range resolveUniformRange;
  Vec4<float> clearColorValue{0.f, 0.f, 0.f, 0.f};
  float clearDepthValue = gx::UseReversedZ ? 0.f : 1.f;
  CommandList commands;
  bool clearColor = true;
  bool clearDepth = true;
  bool observable = false;
  bool sealed = false;
  std::vector<tex_palette_conv::ConvRequest> paletteConvs;
  // Ambient occlusion over what the pass drew, at its end and before its copy.
  bool ambientOcclusion = false;
  bool aoResolvedOnly = false; // applied to the resolve target alone
  Range aoParams;
  ClipRect aoRect;
  // Its copy is sampled later in the same frame, so replays redo it.
  bool copyReadInFrame = false;
};

struct FrameOp {
  uint32_t index = 0;
  RenderPass* renderPass = nullptr;
  StagingHighWater highWater;
  std::vector<const TextureUpload*> textureUploads;
};

using RenderPassList = std::deque<RenderPass>;
struct FramePacket {
  RenderPassList renderPasses;
  std::deque<FrameOp> ops;
  std::deque<TextureUpload> textureUploads;
  ByteBuffer verts;
  ByteBuffer uniforms;
  ByteBuffer indices;
  ByteBuffer storage;
  ByteBuffer textureUpload;
  rhi::CommandList* list = nullptr;
  uint64_t frameId = 0;
  uint32_t frameIndex = 0;
  size_t stagingBuffer = 0;
  StagingHighWater copied;
  AuroraStats stats{};
  // With frame interpolation: a CPU copy of the uniform stream, and the GX
  // draws' blocks in it.
  ByteBuffer uniformShadow;
  std::vector<interp::DrawRecord> interpDraws;
  std::vector<interp::ViewMatrix> interpCameras;
};

static std::array<FramePacket, FrameSlotCount> g_framePackets;
static size_t g_lastUniformShadowSize = 0;
static FramePacket* g_recordingFrame = nullptr;
static size_t g_recordingFrameSlot = 0;
static uint64_t g_nextFrameId = 1;
static render_worker::FrameSlotPool g_frameSlots{FrameSlotCount};
static render_worker::FrameSlotPool g_stagingSlots{StagingBufferCount};
static u32 g_currentRenderPass = UINT32_MAX;
static Viewport g_cachedViewport;
static ClipRect g_cachedScissor;

// Ambient occlusion goes over the world once it is drawn: before the first 2D
// draw after it, an EFB copy, or the frame's end, and once per depth clear so
// nothing is darkened twice.
struct AoState {
  bool pending = false; // the world was drawn since the depth was cleared
  bool done = false;    // and has its occlusion
  Mat4x4<float> proj;   // what it was drawn with
  Viewport viewport;
  ao::Fog fog;
};
static AoState g_ao;

// The frame's copies nothing has sampled since: target texture and pass.
static std::vector<std::pair<const rhi::Texture*, uint32_t>> g_unreadCopies;
// While interpolation replays a frame: copies nothing in it samples keep what
// the frame itself made (the game's state, for the next frame) and aren't redone.
static bool g_replayingFrame = false;

static void set_efb_targets(RenderPass& pass) {
  const bool multisampled = gpu::g_graphicsConfig.msaaSamples > 1;
  pass.colorTexture = gpu::g_frameBuffer.texture;
  pass.colorView = gpu::g_frameBuffer.view;
  pass.resolveTexture = multisampled ? gpu::g_frameBufferResolved.texture : nullptr;
  pass.depthTexture = gpu::g_depthBuffer.texture;
  pass.copySourceTexture = multisampled ? gpu::g_frameBufferResolved.texture : gpu::g_frameBuffer.texture;
  pass.copySourceView = multisampled ? gpu::g_frameBufferResolved.view : gpu::g_frameBuffer.view;
  pass.copySourceDepthView = gpu::g_depthBuffer.view;
  pass.targetSize = {gpu::g_frameBuffer.size.width, gpu::g_frameBuffer.size.height, 1};
  pass.msaaSamples = gpu::g_graphicsConfig.msaaSamples;
}

static FramePacket& current_frame_packet() {
  CHECK(g_recordingFrame != nullptr, "No active frame packet");
  return *g_recordingFrame;
}

static RenderPassList& current_render_passes() { return current_frame_packet().renderPasses; }

static StagingHighWater current_high_water(const FramePacket& frame) noexcept {
  return {
      .verts = static_cast<uint32_t>(frame.verts.size()),
      .uniforms = static_cast<uint32_t>(frame.uniforms.size()),
      .indices = static_cast<uint32_t>(frame.indices.size()),
      .storage = static_cast<uint32_t>(frame.storage.size()),
      .textureUpload = static_cast<uint32_t>(frame.textureUpload.size()),
      .textureUploadCount = frame.textureUploads.size(),
  };
}

static FrameOp capture_frame_op(FramePacket& frame, uint32_t index) {
  FrameOp op{
      .index = index,
      .renderPass = index < frame.renderPasses.size() ? &frame.renderPasses[index] : nullptr,
      .highWater = current_high_water(frame),
  };
  op.textureUploads.reserve(op.highWater.textureUploadCount);
  for (size_t i = 0; i < op.highWater.textureUploadCount; ++i) {
    op.textureUploads.push_back(&frame.textureUploads[i]);
  }
  return op;
}

static void seal_pass(FramePacket& frame, uint32_t passIndex) {
  if (passIndex >= frame.renderPasses.size()) {
    return;
  }
  auto& pass = frame.renderPasses[passIndex];
  if (pass.sealed) {
    return;
  }
  pass.sealed = true;
}

static void encode_op(rhi::CommandList& list, FramePacket& frame, const FrameOp& op);
static void render(rhi::CommandList& list, FramePacket& frame, RenderPass& passInfo);
// The frame interpolation last replayed, the oldest one still drawn: groups it
// binds stay however far the game has moved on. The render worker's.
static uint32_t g_replayedFrameIndex = UINT32_MAX;

static void expire_cached_bind_groups();
static void push_command(CommandType type, const Command::Data& data);
static void replay_frame(rhi::CommandList& list, FramePacket& frame);

static void enqueue_op(FramePacket& frame, size_t frameSlot, uint32_t opIndex) {
  if (opIndex >= frame.ops.size()) {
    return;
  }
  auto op = frame.ops[opIndex];
  render_worker::enqueue_encode_pass(frame.frameId, opIndex, [frameSlot, op = std::move(op)] {
    if (op.renderPass == nullptr) {
      return;
    }
    auto& packet = g_framePackets[frameSlot];
    encode_op(*packet.list, packet, op);
  });
}

static void enqueue_pass(FramePacket& frame, size_t frameSlot, uint32_t passIndex) {
  seal_pass(frame, passIndex);
  const auto opIndex = static_cast<uint32_t>(frame.ops.size());
  frame.ops.emplace_back(capture_frame_op(frame, passIndex));
  enqueue_op(frame, frameSlot, opIndex);
}

void queue_texture_upload(TextureUpload upload) {
  if (g_currentRenderPass != UINT32_MAX) {
    ASSERT(!current_render_passes()[g_currentRenderPass].sealed,
           "Attempted to append texture upload to sealed render pass {}", g_currentRenderPass);
  }
  current_frame_packet().textureUploads.emplace_back(std::move(upload));
}

void queue_texture_upload_data(const uint8_t* data, uint32_t bytesPerRow, uint32_t rowsPerImage,
                               std::shared_ptr<rhi::Texture> texture, uint32_t mip, Extent3D size) {
  const uint64_t uploadSize = uint64_t{AURORA_ALIGN(bytesPerRow, rhi::CopyRowAlignment)} * rowsPerImage;
  auto& frame = current_frame_packet();
  if (AURORA_ALIGN(frame.textureUpload.size(), rhi::CopyOffsetAlignment) + uploadSize <= TextureUploadSize) {
    const auto range = push_texture_data(data, bytesPerRow, rowsPerImage);
    queue_texture_upload(TextureUpload{range.offset, bytesPerRow, std::move(texture), mip, size, nullptr});
    return;
  }

  std::shared_ptr<rhi::Buffer> buffer = g_device->create_buffer({uploadSize, rhi::MemoryKind::Upload});
  auto* dst = static_cast<uint8_t*>(buffer->mapped());
  for (uint32_t row = 0; row < rowsPerImage; ++row) {
    memcpy(dst, data, bytesPerRow);
    data += bytesPerRow;
    dst += AURORA_ALIGN(bytesPerRow, rhi::CopyRowAlignment);
  }
  queue_texture_upload(TextureUpload{0, bytesPerRow, std::move(texture), mip, size, std::move(buffer)});
}

// The GX draw command last pushed, for draws that merge into it. It is still
// the pass's last command while no other command went in after it (the
// serial), the same pass is open and the frame has not moved on.
static struct {
  gx::DrawData* draw = nullptr;
  uint64_t serial = 0;
  uint32_t pass = UINT32_MAX;
  uint32_t frame = UINT32_MAX;
} g_lastGxDraw;
static uint64_t g_commandSerial = 0;

static inline void push_command(CommandType type, const Command::Data& data) {
  if (g_currentRenderPass == UINT32_MAX)
    UNLIKELY {
      Log.warn("Dropping command {}", magic_enum::enum_name(type));
      return;
    }
  ++g_commandSerial;
  auto& renderPass = current_render_passes()[g_currentRenderPass];
  ASSERT(!renderPass.sealed, "Attempted to append command {} to sealed render pass {}", magic_enum::enum_name(type),
         g_currentRenderPass);
  renderPass.commands.push_back({
      .type = type,
      .data = data,
  });
}

template <>
gx::DrawData* get_last_draw_command() {
  const auto& last = g_lastGxDraw;
  if (last.serial != g_commandSerial || last.pass != g_currentRenderPass || last.frame != g_frameIndex) {
    return nullptr;
  }
  return last.draw;
}

static void push_draw_command(ShaderDrawCommand data) {
  push_command(CommandType::Draw, Command::Data{.draw = data});
  ++g_drawCallCount;
}

Vec2<uint32_t> get_render_target_size() noexcept {
  if (g_currentRenderPass < current_render_passes().size()) {
    const auto& size = current_render_passes()[g_currentRenderPass].targetSize;
    return {size.width, size.height};
  }
  const auto windowSize = window::get_window_size();
  return {windowSize.fb_width, windowSize.fb_height};
}

void set_viewport(const Viewport& cmd) noexcept {
  if (cmd != g_cachedViewport) {
    push_command(CommandType::SetViewport, Command::Data{.setViewport = cmd});
    g_cachedViewport = cmd;
  }
}

void set_scissor(const ClipRect& cmd) noexcept {
  if (cmd != g_cachedScissor) {
    push_command(CommandType::SetScissor, Command::Data{.setScissor = cmd});
    g_cachedScissor = cmd;
  }
}

template <>
void push_draw_command(clear::DrawData data) {
  push_draw_command(ShaderDrawCommand{.type = ShaderType::Clear, .clear = data});
}

template <>
PipelineRef pipeline_ref(const clear::PipelineConfig& config) {
  return find_pipeline_blocking(ShaderType::Clear, config, [=] { return create_pipeline(config); });
}

static void add_ambient_occlusion(RenderPass& pass) {
  const auto& vp = g_ao.viewport;
  const auto left = std::clamp(static_cast<int32_t>(std::floor(vp.left)), 0, static_cast<int32_t>(pass.targetSize.width));
  const auto top = std::clamp(static_cast<int32_t>(std::floor(vp.top)), 0, static_cast<int32_t>(pass.targetSize.height));
  const auto right =
      std::clamp(static_cast<int32_t>(std::ceil(vp.left + vp.width)), left, static_cast<int32_t>(pass.targetSize.width));
  const auto bottom =
      std::clamp(static_cast<int32_t>(std::ceil(vp.top + vp.height)), top, static_cast<int32_t>(pass.targetSize.height));
  g_ao.pending = false;
  g_ao.done = true;
  if (right <= left || bottom <= top) {
    return;
  }
  pass.ambientOcclusion = true;
  pass.aoParams = ao::push_params(g_ao.proj, vp, pass.targetSize, g_ao.fog);
  pass.aoRect = {left, top, right - left, bottom - top};
}

// A world draw: perspective, writing depth, over most of the EFB (not a model
// in a corner of the HUD). A 2D draw after one ends the world's pass.
static void note_draw_for_ambient_occlusion() {
  const auto& state = gx::g_gxState;
  if (state.projType != GX_ORTHOGRAPHIC) {
    const auto& vp = g_cachedViewport;
    const auto& size = current_render_passes()[g_currentRenderPass].targetSize;
    const float area = std::max(vp.width, 0.f) * std::max(vp.height, 0.f);
    if (!g_ao.done && state.depthCompare && state.depthUpdate && vp.znear < vp.zfar &&
        area * 2.f >= static_cast<float>(size.width) * static_cast<float>(size.height)) {
      g_ao.pending = true;
      g_ao.proj = state.proj;
      g_ao.viewport = vp;
      g_ao.fog = {.type = static_cast<uint32_t>(state.fog.type), .a = state.fog.a, .b = state.fog.b, .c = state.fog.c};
    }
    return;
  }
  if (!g_ao.pending) {
    return;
  }
  // The pass goes on as another, which keeps what this one drew.
  auto& prevPass = current_render_passes()[g_currentRenderPass];
  add_ambient_occlusion(prevPass);
  prevPass.observable = true;
  enqueue_pass(current_frame_packet(), g_recordingFrameSlot, g_currentRenderPass);
  RenderPass newPass{
      .colorTexture = prevPass.colorTexture,
      .colorView = prevPass.colorView,
      .resolveTexture = prevPass.resolveTexture,
      .depthTexture = prevPass.depthTexture,
      .copySourceTexture = prevPass.copySourceTexture,
      .copySourceView = prevPass.copySourceView,
      .copySourceDepthView = prevPass.copySourceDepthView,
      .targetSize = prevPass.targetSize,
      .msaaSamples = prevPass.msaaSamples,
      .clearColor = false,
      .clearDepth = false,
  };
  newPass.commands.reserve(2048);
  current_render_passes().emplace_back(std::move(newPass));
  ++g_currentRenderPass;
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_cachedScissor});
}

void resolve_pass(TextureHandle texture, ClipRect rect, bool clearColor, bool clearAlpha, bool clearDepth,
                  Vec4<float> clearColorValue, float clearDepthValue, GXTexFmt resolveFormat, bool opaqueAlpha) {
  // Resolve current render pass
  auto& prevPass = current_render_passes()[g_currentRenderPass];
  if (g_ao.pending) {
    add_ambient_occlusion(prevPass);
    // When the copy takes it all and the color is cleared after, only what it
    // copies needs the occlusion: the resolved EFB, a sample a pixel.
    const auto& size = prevPass.targetSize;
    prevPass.aoResolvedOnly = prevPass.resolveTexture != nullptr && clearColor && clearAlpha && rect.x <= 0 &&
                              rect.y <= 0 && rect.x + rect.width >= static_cast<int32_t>(size.width) &&
                              rect.y + rect.height >= static_cast<int32_t>(size.height);
  }
  if (clearDepth) {
    g_ao = {};
  }
  if (texture) {
    const rhi::Texture* target = texture->texture.get();
    std::erase_if(g_unreadCopies, [target](const auto& entry) { return entry.first == target; });
    g_unreadCopies.emplace_back(target, g_currentRenderPass);
  }
  prevPass.resolveTarget = std::move(texture);
  prevPass.resolveRect = rect;
  prevPass.resolveFormat = resolveFormat;
  prevPass.resolveOpaqueAlpha = opaqueAlpha && !gx::is_depth_format(resolveFormat);
  // tex_copy_conv's uniform: the crop region in UV space, and whether alpha reads 1
  const auto srcW = static_cast<float>(prevPass.targetSize.width);
  const auto srcH = static_cast<float>(prevPass.targetSize.height);
  struct {
    float offset[2];
    float scale[2];
    uint32_t opaqueAlpha;
    uint32_t pad;
  } const uniform{
      .offset = {static_cast<float>(rect.x) / srcW, static_cast<float>(rect.y) / srcH},
      .scale = {static_cast<float>(rect.width) / srcW, static_cast<float>(rect.height) / srcH},
      .opaqueAlpha = prevPass.resolveOpaqueAlpha,
      .pad = 0,
  };
  prevPass.resolveUniformRange = push_uniform(uniform);
  enqueue_pass(current_frame_packet(), g_recordingFrameSlot, g_currentRenderPass);

  // Populate new render pass from previous. A clear of the whole target is the
  // pass's load op; of part of it (or of only color or alpha), a clear draw.
  const auto msaaSamples = prevPass.msaaSamples;
  const auto& targetSize = prevPass.targetSize;
  const auto left = std::clamp<int32_t>(rect.x, 0, static_cast<int32_t>(targetSize.width));
  const auto top = std::clamp<int32_t>(rect.y, 0, static_cast<int32_t>(targetSize.height));
  const auto right = std::clamp<int32_t>(rect.x + rect.width, left, static_cast<int32_t>(targetSize.width));
  const auto bottom = std::clamp<int32_t>(rect.y + rect.height, top, static_cast<int32_t>(targetSize.height));
  const ClipRect clearRect{.x = left, .y = top, .width = right - left, .height = bottom - top};
  const bool fullTarget = left == 0 && top == 0 && right == static_cast<int32_t>(targetSize.width) &&
                          bottom == static_cast<int32_t>(targetSize.height);
  const bool fullColorClear = clearColor && clearAlpha && fullTarget;
  RenderPass newPass{
      .colorTexture = prevPass.colorTexture,
      .colorView = prevPass.colorView,
      .resolveTexture = prevPass.resolveTexture,
      .depthTexture = prevPass.depthTexture,
      .copySourceTexture = prevPass.copySourceTexture,
      .copySourceView = prevPass.copySourceView,
      .copySourceDepthView = prevPass.copySourceDepthView,
      .targetSize = prevPass.targetSize,
      .msaaSamples = msaaSamples,
      .clearColorValue = clearColorValue,
      .clearDepthValue = clearDepthValue,
      .clearColor = fullColorClear,
      .clearDepth = clearDepth && fullTarget,
  };
  newPass.commands.reserve(2048);
  current_render_passes().emplace_back(std::move(newPass));
  ++g_currentRenderPass;

  const bool drawClearColor = clearColor && !fullColorClear;
  const bool drawClearAlpha = clearAlpha && !fullColorClear;
  const bool drawClearDepth = clearDepth && !fullTarget;
  if (clearRect.width > 0 && clearRect.height > 0 && (drawClearColor || drawClearAlpha || drawClearDepth)) {
    push_draw_command(clear::DrawData{
        .pipeline = pipeline_ref(clear::PipelineConfig{
            .msaaSamples = msaaSamples,
            .clearColor = drawClearColor,
            .clearAlpha = drawClearAlpha,
            .clearDepth = drawClearDepth,
        }),
        .color = {clearColorValue.x(), clearColorValue.y(), clearColorValue.z(), clearColorValue.w()},
        .depth = clearDepthValue,
        .rect = clearRect,
    });
  }
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_cachedScissor});
}

void note_copy_sampled(const TextureHandle& texture) {
  if (!texture || g_recordingFrame == nullptr) {
    return;
  }
  const rhi::Texture* target = texture->texture.get();
  for (auto it = g_unreadCopies.begin(); it != g_unreadCopies.end(); ++it) {
    if (it->first == target) {
      current_render_passes()[it->second].copyReadInFrame = true;
      g_unreadCopies.erase(it);
      return;
    }
  }
}

void queue_palette_conv(tex_palette_conv::ConvRequest req) {
  auto& renderPass = current_render_passes()[g_currentRenderPass];
  ASSERT(!renderPass.sealed, "Attempted to append palette conversion to sealed render pass {}", g_currentRenderPass);
  renderPass.paletteConvs.push_back(std::move(req));
}

uint32_t get_sample_count() noexcept {
  CHECK(g_currentRenderPass != UINT32_MAX, "get_sample_count called outside of a frame");
  return current_render_passes()[g_currentRenderPass].msaaSamples;
}

void clear_caches() noexcept {
  std::lock_guard lock{g_bindGroupCacheMutex};
  g_cachedBindGroups.clear();
}

template <>
void push_draw_command(gx::DrawData data) {
  if (ao::enabled()) {
    note_draw_for_ambient_occlusion();
  }
  const uint64_t serial = g_commandSerial;
  push_draw_command(ShaderDrawCommand{.type = ShaderType::GX, .gx = data});
  if (g_commandSerial != serial) {
    g_lastGxDraw = {
        .draw = &current_render_passes()[g_currentRenderPass].commands.back().data.draw.gx,
        .serial = g_commandSerial,
        .pass = g_currentRenderPass,
        .frame = g_frameIndex,
    };
  }
}

template <>
PipelineRef pipeline_ref(const gx::PipelineConfig& config) {
  return find_pipeline(ShaderType::GX, config, [=] { return create_pipeline(config); });
}

PipelineRef uber_pipeline_ref(const gx::PipelineConfig& config) {
  return find_pipeline_blocking(ShaderType::GXUber, config, [=] { return gx::create_uber_pipeline(config); });
}

void initialize() {
  g_frameIndex = 0;
  render_worker::initialize();
  tex_copy_conv::initialize();
  tex_palette_conv::initialize();
  texture_replacement::initialize();
  ao::initialize();

  g_uniformBufferOwner = g_device->create_buffer({UniformBufferSize});
  g_uniformBuffer = g_uniformBufferOwner.get();
  g_vertexBuffer = g_device->create_buffer({VertexBufferSize});
  g_indexBuffer = g_device->create_buffer({IndexBufferSize});
  g_storageBuffer = g_device->create_buffer({StorageBufferSize});
  for (size_t i = 0; i < StagingBufferCount; ++i) {
    g_stagingBuffers[i] = g_device->create_buffer({StagingBufferSize, rhi::MemoryKind::Upload});
    g_stagingLists[i] = g_device->create_command_list();
    g_stagingTickets[i].store(0, std::memory_order_release);
  }

  {
    constexpr std::array layoutEntries{
        rhi::BindingLayout{0, rhi::BindingKind::StorageRead}, // Vertex data buffer
        rhi::BindingLayout{1, rhi::BindingKind::StorageRead}, // Storage data buffer
    };
    g_staticBindGroupLayoutOwner = g_device->create_bind_group_layout({layoutEntries.data(), layoutEntries.size()});
    g_staticBindGroupLayout = g_staticBindGroupLayoutOwner.get();
    std::array<rhi::BindingValue, 2> values{};
    values[0].binding = 0;
    values[0].buffer = g_vertexBuffer.get();
    values[0].size = VertexBufferSize;
    values[1].binding = 1;
    values[1].buffer = g_storageBuffer.get();
    values[1].size = StorageBufferSize;
    g_staticBindGroup = g_device->create_bind_group({g_staticBindGroupLayout, values.data(), values.size()});
  }

  {
    constexpr std::array layoutEntries{
        rhi::BindingLayout{0, rhi::BindingKind::Uniform, true},
    };
    g_uniformBindGroupLayoutOwner = g_device->create_bind_group_layout({layoutEntries.data(), layoutEntries.size()});
    g_uniformBindGroupLayout = g_uniformBindGroupLayoutOwner.get();
    rhi::BindingValue value{};
    value.binding = 0;
    value.buffer = g_uniformBuffer;
    value.size = gx::MaxUniformSize;
    g_uniformBindGroup = g_device->create_bind_group({g_uniformBindGroupLayout, &value, 1});
    interp::g_vertexUniformBindGroup = g_uniformBindGroup.get();
  }

  gx::initialize();
  initialize_pipeline_cache();
  if (g_config.frameInterpolation) {
    interp::initialize(g_config.vsync, g_config.presentRate);
  }
}

void shutdown() {
  interp::shutdown();
  render_worker::synchronize();
  render_worker::shutdown();
  g_device->wait_idle();
  shutdown_pipeline_cache();
  tex_copy_conv::shutdown();
  tex_palette_conv::shutdown();
  texture_replacement::shutdown();
  ao::shutdown();
  gx::shutdown();

  {
    std::lock_guard lock{g_bindGroupCacheMutex};
    g_cachedBindGroups.clear();
  }
  {
    std::lock_guard lock{g_samplerCacheMutex};
    g_cachedSamplers.clear();
  }
  for (auto& packet : g_framePackets) {
    packet = {};
  }
  g_recordingFrame = nullptr;
  g_currentRenderPass = UINT32_MAX;
  interp::g_vertexUniformBindGroup = nullptr;
  g_staticBindGroup.reset();
  g_staticBindGroupLayout = nullptr;
  g_staticBindGroupLayoutOwner.reset();
  g_uniformBindGroup.reset();
  g_uniformBindGroupLayout = nullptr;
  g_uniformBindGroupLayoutOwner.reset();
  g_vertexBuffer.reset();
  g_uniformBuffer = nullptr;
  g_uniformBufferOwner.reset();
  g_indexBuffer.reset();
  g_storageBuffer.reset();
  for (size_t i = 0; i < StagingBufferCount; ++i) {
    g_stagingLists[i].reset();
    g_stagingBuffers[i].reset();
  }
  g_frameIndex = UINT32_MAX;
  g_frameSlots.reset();
  g_stagingSlots.reset();
}

static size_t acquire_frame_slot() {
  while (true) {
    if (const auto slot = g_frameSlots.try_acquire()) {
      return *slot;
    }
    std::this_thread::sleep_for(std::chrono::microseconds{100});
  }
}

// A free slot the GPU has finished with.
static size_t acquire_staging_slot() {
  while (true) {
    std::array<size_t, StagingBufferCount> busy;
    size_t busyCount = 0;
    std::optional<size_t> found;
    while (const auto slot = g_stagingSlots.try_acquire()) {
      if (g_device->is_done(g_stagingTickets[*slot].load(std::memory_order_acquire))) {
        found = slot;
        break;
      }
      busy[busyCount++] = *slot;
    }
    for (size_t i = 0; i < busyCount; ++i) {
      g_stagingSlots.release(busy[i]);
    }
    if (found) {
      return *found;
    }
    std::this_thread::sleep_for(std::chrono::microseconds{100});
  }
}

bool begin_frame() {
  const size_t frameSlot = acquire_frame_slot();
  const size_t stagingSlot = acquire_staging_slot();

  auto& frame = g_framePackets[frameSlot];
  frame = {};
  frame.frameId = g_nextFrameId++;
  frame.frameIndex = g_frameIndex;
  frame.stagingBuffer = stagingSlot;
  frame.list = g_stagingLists[stagingSlot].get();
  if (interp::enabled()) {
    frame.uniformShadow.reserve_extra(g_lastUniformShadowSize);
  }
  g_recordingFrame = &frame;
  g_recordingFrameSlot = frameSlot;

  size_t bufferOffset = 0;
  auto* const stagingData = static_cast<u8*>(g_stagingBuffers[stagingSlot]->mapped());
  const auto mapBuffer = [&](ByteBuffer& buf, uint64_t size) {
    if (size <= 0) {
      return;
    }
    buf = ByteBuffer{stagingData + bufferOffset, static_cast<size_t>(size)};
    bufferOffset += size;
  };
  mapBuffer(frame.verts, VertexBufferSize);
  mapBuffer(frame.uniforms, UniformBufferSize);
  mapBuffer(frame.indices, IndexBufferSize);
  mapBuffer(frame.storage, StorageBufferSize);
  if constexpr (UseTextureBuffer) {
    mapBuffer(frame.textureUpload, TextureUploadSize);
  }

  g_drawCallCount = 0;
  g_mergedDrawCallCount = 0;
  g_droppedDrawCallCount = 0;

  current_render_passes().emplace_back();
  auto& pass = current_render_passes()[0];
  set_efb_targets(pass);
  g_ao = {};
  g_unreadCopies.clear();
  pass.clearColorValue = gx::g_gxState.clearColor;
  pass.clearDepthValue = gx::clear_depth_value();
  g_currentRenderPass = 0;
  // Refresh render viewport/scissor from logical in case FB size changed
  g_cachedViewport = gx::map_logical_viewport(gx::g_gxState.logicalViewport);
  g_cachedScissor = gx::map_logical_scissor(gx::g_gxState.logicalScissor);
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_cachedScissor});
  render_worker::enqueue_begin_frame(frame.frameId, [frameSlot] { g_framePackets[frameSlot].list->begin(); });
  return true;
}

void finish() {
  if (g_recordingFrame == nullptr) {
    return;
  }
  if (g_currentRenderPass != UINT32_MAX) {
    auto& frame = current_frame_packet();
    frame.uniforms.append_zeroes(gx::MaxUniformSize);
    auto& pass = frame.renderPasses[g_currentRenderPass];
    pass.observable = true;
    if (g_ao.pending) {
      add_ambient_occlusion(pass);
    }
    enqueue_pass(frame, g_recordingFrameSlot, g_currentRenderPass);
    g_currentRenderPass = UINT32_MAX;
  }
}

static void publish_frame_stats(const AuroraStats& stats) {
  g_stats.drawCallCount = stats.drawCallCount;
  g_stats.mergedDrawCallCount = stats.mergedDrawCallCount;
  g_stats.droppedDrawCallCount = stats.droppedDrawCallCount;
  g_stats.lastVertSize = stats.lastVertSize;
  g_stats.lastUniformSize = stats.lastUniformSize;
  g_stats.lastIndexSize = stats.lastIndexSize;
  g_stats.lastStorageSize = stats.lastStorageSize;
  g_stats.lastTextureUploadSize = stats.lastTextureUploadSize;
}

void end_frame(EndFrameCallback callback) {
  ASSERT(g_currentRenderPass == UINT32_MAX, "end_frame called before finish finalized the current render pass");
  auto& frame = current_frame_packet();
  frame.stats.drawCallCount = g_drawCallCount;
  frame.stats.mergedDrawCallCount = g_mergedDrawCallCount;
  frame.stats.droppedDrawCallCount = g_droppedDrawCallCount;
  frame.stats.lastVertSize = frame.verts.size();
  frame.stats.lastUniformSize = frame.uniforms.size();
  frame.stats.lastIndexSize = frame.indices.size();
  frame.stats.lastStorageSize = frame.storage.size();
  frame.stats.lastTextureUploadSize = frame.textureUpload.size();

  const size_t frameSlot = g_recordingFrameSlot;
  const uint64_t frameId = frame.frameId;
  g_currentRenderPass = UINT32_MAX;
  for (auto& array : gx::g_gxState.arrays) {
    array.cachedRange = {};
  }
  gx::end_frame_copy_textures();
  ++g_frameIndex;
  g_recordingFrame = nullptr;


  const size_t stagingSlot = frame.stagingBuffer;
  render_worker::enqueue_end_frame(frameId, [frameSlot, stagingSlot, callback = std::move(callback)]() mutable {
    auto& packet = g_framePackets[frameSlot];
    auto* const list = packet.list;
    const auto stats = packet.stats;
    const auto finished = [frameSlot, stagingSlot, stats](uint64_t ticket) {
      publish_frame_stats(stats);
      g_frameSlots.release(frameSlot);
      expire_cached_bind_groups();
      g_stagingTickets[stagingSlot].store(ticket, std::memory_order_release);
      g_stagingSlots.release(stagingSlot);
    };
    if (interp::enabled() && callback) {
      // The frame is submitted, for its copies and side effects, when
      // interpolation reaches it; what is shown are its replays. Its slot is
      // held until then.
      auto retained = std::make_shared<FramePacket>();
      retained->frameIndex = packet.frameIndex;
      retained->renderPasses = std::move(packet.renderPasses);
      retained->ops = std::move(packet.ops);
      g_lastUniformShadowSize = packet.uniformShadow.size();
      auto uniforms = std::move(packet.uniformShadow);
      auto draws = std::move(packet.interpDraws);
      auto cameras = std::move(packet.interpCameras);
      packet = {};
      interp::push_frame(
          std::move(uniforms), std::move(draws), std::move(cameras),
          [retained](rhi::CommandList& replay) {
            g_replayedFrameIndex = retained->frameIndex;
            replay_frame(replay, *retained);
          },
          std::move(callback),
          [list, finished] {
            list->end();
            finished(g_device->submit(*list));
          });
      return;
    }
    uint64_t ticket;
    if (callback) {
      ticket = callback(*list);
    } else {
      list->end();
      ticket = g_device->submit(*list);
    }
    // What the frame's passes hold goes only now: until the submit the GPU hasn't
    // been told about it.
    packet = {};
    finished(ticket);
  });
}

uint32_t current_frame() noexcept { return g_frameIndex; }

static void expire_cached_bind_groups() {
  std::lock_guard lock{g_bindGroupCacheMutex};
  if (g_cachedBindGroups.empty() || g_frameIndex == UINT32_MAX || g_frameIndex % BindGroupCacheSweepPeriod != 0) {
    return;
  }

  const uint32_t now = std::min(g_frameIndex, g_replayedFrameIndex);
  for (auto it = g_cachedBindGroups.begin(); it != g_cachedBindGroups.end();) {
    if (it->second.lastUsedFrame + BindGroupCacheRetainFrames < now) {
      g_cachedBindGroups.erase(it++);
    } else {
      ++it;
    }
  }
}

static constexpr uint64_t VertexStagingOffset = 0;
static constexpr uint64_t UniformStagingOffset = VertexStagingOffset + VertexBufferSize;
static constexpr uint64_t IndexStagingOffset = UniformStagingOffset + UniformBufferSize;
static constexpr uint64_t StorageStagingOffset = IndexStagingOffset + IndexBufferSize;
static constexpr uint64_t TextureUploadStagingOffset = StorageStagingOffset + StorageBufferSize;
static_assert(TextureUploadStagingOffset % rhi::CopyOffsetAlignment == 0);

static constexpr uint32_t align_down_copy_offset(uint32_t value) noexcept { return value & ~3u; }

static void copy_staging_buffer_range(rhi::CommandList& list, const FramePacket& frame, uint32_t& copied,
                                      uint32_t highWater, uint64_t stagingOffset, rhi::Buffer& dst) {
  if (highWater <= copied) {
    return;
  }
  const uint32_t copyStart = align_down_copy_offset(copied);
  const uint32_t copyEnd = AURORA_ALIGN(highWater, 4);
  list.copy_buffer(*g_stagingBuffers[frame.stagingBuffer], stagingOffset + copyStart, dst, copyStart,
                   copyEnd - copyStart);
  copied = highWater;
}

static bool needs_staging_copy(const FramePacket& frame, const FrameOp& op) {
  const auto& highWater = op.highWater;
  if (highWater.verts > frame.copied.verts || highWater.uniforms > frame.copied.uniforms ||
      highWater.indices > frame.copied.indices || highWater.storage > frame.copied.storage) {
    return true;
  }
  if constexpr (UseTextureBuffer) {
    return op.textureUploads.size() > frame.copied.textureUploadCount;
  }
  return false;
}

static void copy_staging_to_high_water(rhi::CommandList& list, FramePacket& frame, const FrameOp& op) {
  if (!needs_staging_copy(frame, op)) {
    return;
  }
  const auto& highWater = op.highWater;
  copy_staging_buffer_range(list, frame, frame.copied.verts, highWater.verts, VertexStagingOffset, *g_vertexBuffer);
  copy_staging_buffer_range(list, frame, frame.copied.uniforms, highWater.uniforms, UniformStagingOffset,
                            *g_uniformBuffer);
  copy_staging_buffer_range(list, frame, frame.copied.indices, highWater.indices, IndexStagingOffset, *g_indexBuffer);
  copy_staging_buffer_range(list, frame, frame.copied.storage, highWater.storage, StorageStagingOffset,
                            *g_storageBuffer);

  if constexpr (UseTextureBuffer) {
    for (size_t i = frame.copied.textureUploadCount; i < op.textureUploads.size(); ++i) {
      const auto& item = *op.textureUploads[i];
      list.copy_buffer_to_texture(item.buffer ? *item.buffer : *g_stagingBuffers[frame.stagingBuffer],
                                  item.buffer ? item.offset : item.offset + TextureUploadStagingOffset,
                                  AURORA_ALIGN(item.bytesPerRow, rhi::CopyRowAlignment), *item.texture,
                                  {item.mip, 0, 0, item.size.width, item.size.height});
    }
    frame.copied.textureUpload = highWater.textureUpload;
    frame.copied.textureUploadCount = op.textureUploads.size();
  }
}

static void encode_op(rhi::CommandList& list, FramePacket& frame, const FrameOp& op) {
  copy_staging_to_high_water(list, frame, op);
  if (op.renderPass != nullptr) {
    render(list, frame, *op.renderPass);
  }
}

static void render_pass(rhi::CommandList& list, RenderPass& passInfo);

// The multisampled color the resolve target holds as it is now; empty once
// that color is drawn to. Copies in a row with nothing drawn between them
// resolve it once, and depth copies, which read the depth, not at all.
static std::weak_ptr<rhi::Texture> g_resolvedColor;

static void render(rhi::CommandList& list, FramePacket& frame, RenderPass& passInfo) {
  if (!passInfo.sealed) {
    return;
  }

  for (const auto& conv : passInfo.paletteConvs) {
    tex_palette_conv::run(list, conv);
  }
  if (!passInfo.observable && !passInfo.resolveTarget) {
    // Skip intermediate EFB render passes without observable output.
    return;
  }

  bool draws = passInfo.clearColor;
  for (const auto& cmd : passInfo.commands) {
    draws = draws || cmd.type == CommandType::Draw;
  }
  if (draws) {
    g_resolvedColor.reset();
  }
  const bool copy = passInfo.resolveTarget && (!g_replayingFrame || passInfo.copyReadInFrame);
  const bool resolved = passInfo.resolveTexture && g_resolvedColor.lock() == passInfo.colorTexture;
  // With ambient occlusion over the multisampled color, its pass resolves it.
  const bool resolve = passInfo.resolveTexture && !resolved &&
                       !(passInfo.ambientOcclusion && !passInfo.aoResolvedOnly) &&
                       (passInfo.observable || passInfo.aoResolvedOnly ||
                        (copy && !gx::is_depth_format(passInfo.resolveFormat)));
  // With nothing to draw, clear or resolve, only the copy.
  if (draws || passInfo.clearDepth || resolve || passInfo.ambientOcclusion) {
    rhi::RenderPassDesc passDesc;
    passDesc.color.target = passInfo.colorTexture.get();
    passDesc.color.resolve = resolve ? passInfo.resolveTexture.get() : nullptr;
    passDesc.color.clear = passInfo.clearColor;
    passDesc.color.clear_color = {passInfo.clearColorValue.x(), passInfo.clearColorValue.y(),
                                  passInfo.clearColorValue.z(), passInfo.clearColorValue.w()};
    passDesc.depth.target = passInfo.depthTexture.get();
    passDesc.depth.clear = passInfo.clearDepth;
    passDesc.depth.clear_depth = passInfo.clearDepthValue;
    list.begin_render_pass(passDesc);
    render_pass(list, passInfo);
    list.end_render_pass();
    if (resolve) {
      g_resolvedColor = passInfo.colorTexture;
    }
    if (passInfo.ambientOcclusion) {
      ao::apply(list, {
                          .color = passInfo.aoResolvedOnly ? passInfo.resolveTexture.get()
                                                           : passInfo.colorTexture.get(),
                          .colorView = passInfo.aoResolvedOnly ? passInfo.copySourceView.get()
                                                               : passInfo.colorView.get(),
                          .resolve = passInfo.aoResolvedOnly ? nullptr : passInfo.resolveTexture.get(),
                          .depth = passInfo.copySourceDepthView.get(),
                          .samples = passInfo.aoResolvedOnly ? 1 : passInfo.msaaSamples,
                          .depthSamples = passInfo.msaaSamples,
                          .size = passInfo.targetSize,
                          .rect = passInfo.aoRect,
                          .params = passInfo.aoParams,
                      });
      // Over the multisampled color it resolves it; over the resolve target
      // alone, the two differ.
      if (passInfo.aoResolvedOnly) {
        g_resolvedColor.reset();
      } else {
        g_resolvedColor = passInfo.colorTexture;
      }
  }
  }

  if (copy) {
    const auto& dstSize = passInfo.resolveTarget->size;
    const bool needsConversion = tex_copy_conv::needs_conversion(passInfo.resolveFormat);
    const bool needsScaling = dstSize.width != static_cast<uint32_t>(passInfo.resolveRect.width) ||
                              dstSize.height != static_cast<uint32_t>(passInfo.resolveRect.height);
    const bool isDepth = gx::is_depth_format(passInfo.resolveFormat);
    // A plain copy only where the target matches the EFB and keeps its alpha.
    const bool sameFormat = passInfo.resolveTarget->format == gpu::g_graphicsConfig.surfaceFormat;
    const tex_copy_conv::ConvRequest convReq{
        .fmt = passInfo.resolveFormat,
        .srcView = isDepth ? passInfo.copySourceDepthView : passInfo.copySourceView,
        .uniformRange = passInfo.resolveUniformRange,
        .dst = passInfo.resolveTarget,
        .sampleFilter = needsScaling ? tex_copy_conv::SampleFilter::Linear : tex_copy_conv::SampleFilter::Nearest,
        .multisampled = isDepth && passInfo.msaaSamples > 1,
    };
    if (needsConversion) {
      tex_copy_conv::run(list, convReq);
    } else if (needsScaling || passInfo.resolveOpaqueAlpha || !sameFormat) {
      tex_copy_conv::blit(list, convReq);
    } else {
      list.copy_texture(*passInfo.copySourceTexture,
                        {0, static_cast<uint32_t>(passInfo.resolveRect.x), static_cast<uint32_t>(passInfo.resolveRect.y),
                         static_cast<uint32_t>(passInfo.resolveRect.width),
                         static_cast<uint32_t>(passInfo.resolveRect.height)},
                        *passInfo.resolveTarget->texture, 0, 0);
    }
  }
}

static void replay_frame(rhi::CommandList& list, FramePacket& frame) {
  g_replayingFrame = true;
  for (const auto& op : frame.ops) {
    if (op.renderPass != nullptr) {
      render(list, frame, *op.renderPass);
    }
  }
  g_replayingFrame = false;
}

void interp::record_draw(const DrawRecord& record) { current_frame_packet().interpDraws.push_back(record); }

void interp::merge_draw(uint16_t used) {
  auto& draws = current_frame_packet().interpDraws;
  if (!draws.empty()) {
    draws.back().used |= used;
  }
}

uint16_t interp::record_camera(const ViewMatrix& view) {
  auto& cameras = current_frame_packet().interpCameras;
  cameras.push_back(view);
  return static_cast<uint16_t>(std::min<size_t>(cameras.size(), UINT16_MAX));
}

void gpu_synchronize() { render_worker::synchronize(); }

static void encode_command(rhi::CommandList& list, const RenderPass& passInfo, const Command& cmd) {
  switch (cmd.type) {
  case CommandType::SetViewport: {
    const auto& vp = cmd.data.setViewport;
    const float minDepth = gx::UseReversedZ ? 1.f - vp.zfar : vp.znear;
    const float maxDepth = gx::UseReversedZ ? 1.f - vp.znear : vp.zfar;
    // Games leave empty viewports, which Vulkan doesn't take and draw nothing anyway.
    g_viewportEmpty = vp.width <= 0.f || vp.height <= 0.f;
    if (g_viewportEmpty) {
      break;
    }
    list.set_viewport(vp.left, vp.top, vp.width, vp.height, minDepth, maxDepth);
  } break;
  case CommandType::SetScissor: {
    const auto& sc = cmd.data.setScissor;
    const auto& size = passInfo.targetSize;
    const auto x = std::clamp(static_cast<uint32_t>(sc.x), 0u, size.width);
    const auto y = std::clamp(static_cast<uint32_t>(sc.y), 0u, size.height);
    const auto w = std::clamp(static_cast<uint32_t>(sc.width), 0u, size.width - x);
    const auto h = std::clamp(static_cast<uint32_t>(sc.height), 0u, size.height - y);
    list.set_scissor(x, y, w, h);
  } break;
  case CommandType::Draw: {
    const auto& draw = cmd.data.draw;
    switch (draw.type) {
    case ShaderType::Clear:
      clear::render(draw.clear, list, passInfo.targetSize);
      // Its pipeline has no groups: GX's are set again after it.
      g_gxGroupsBound = false;
      break;
    case ShaderType::GX:
      if (!g_viewportEmpty) {
        gx::render(draw.gx, list);
      }
      break;
    case ShaderType::GXUber: // A pipeline's type; its draws are GX draws.
      break;
    }
  } break;
  }
}

static void render_pass(rhi::CommandList& list, RenderPass& passInfo) {
  g_currentPipeline = UINTPTR_MAX;
  g_gxGroupsBound = false;
  g_indexBufferBound = false;
  g_viewportEmpty = false;
  for (const auto& cmd : passInfo.commands) {
    encode_command(list, passInfo, cmd);
  }
}

bool bind_pipeline(PipelineRef ref, rhi::CommandList& list) {
  if (ref == g_currentPipeline) {
    return true;
  }
  auto* const pipeline = get_pipeline(ref);
  if (pipeline == nullptr) {
    ++g_stats.skippedDraws;
    return false;
  }
  list.set_pipeline(*pipeline);
  g_currentPipeline = ref;
  return true;
}

void bind_gx_groups(rhi::CommandList& list) {
  if (g_gxGroupsBound) {
    return;
  }
  list.set_bind_group(0, *g_staticBindGroup, nullptr, 0);
  list.set_bind_group(2, *gx::g_emptyTextureBindGroup, nullptr, 0);
  g_currentTextureBindGroup = 0;
  g_gxGroupsBound = true;
}

static Range push(ByteBuffer& target, const uint8_t* data, size_t length, size_t alignment) {
  if (alignment != 0) {
    const size_t begin = target.size();
    const size_t alignedBegin = AURORA_ALIGN(begin, alignment);
    if (alignedBegin > begin) {
      target.append_zeroes(alignedBegin - begin);
    }
  }
  const auto begin = target.size();
  if (length > 0) {
    target.append(data, length);
  }
  return {static_cast<uint32_t>(begin), static_cast<uint32_t>(length)};
}

static Range map(ByteBuffer& target, size_t length, size_t alignment) {
  if (alignment != 0) {
    const size_t begin = target.size();
    const size_t alignedBegin = AURORA_ALIGN(begin, alignment);
    if (alignedBegin > begin) {
      target.append_zeroes(alignedBegin - begin);
    }
  }
  auto begin = target.size();
  if (length > 0) {
    target.append_zeroes(length);
  }
  return {static_cast<uint32_t>(begin), static_cast<uint32_t>(length)};
}

Range push_verts(const uint8_t* data, size_t length, size_t alignment) {
  return push(current_frame_packet().verts, data, length, alignment);
}

ByteBuffer& index_staging(size_t alignment) {
  auto& indices = current_frame_packet().indices;
  if (alignment != 0) {
    const size_t begin = indices.size();
    const size_t alignedBegin = AURORA_ALIGN(begin, alignment);
    if (alignedBegin > begin) {
      indices.append_zeroes(alignedBegin - begin);
    }
  }
  return indices;
}

Range push_uniform(const uint8_t* data, size_t length) {
  auto& frame = current_frame_packet();
  const auto range = push(frame.uniforms, data, length, g_device->features().uniform_offset_alignment);
  if (interp::enabled()) {
    auto& shadow = frame.uniformShadow;
    shadow.append_zeroes(range.offset - shadow.size());
    shadow.append(data, length);
  }
  return range;
}

Range push_storage(const uint8_t* data, size_t length) {
  return push(current_frame_packet().storage, data, length, 16);
}

bool frame_has_room(size_t verts, size_t indices, size_t uniforms, size_t storage) noexcept {
  const auto& frame = current_frame_packet();
  return frame.verts.remaining() >= verts && frame.indices.remaining() >= indices &&
         frame.uniforms.remaining() >= uniforms && frame.storage.remaining() >= storage;
}

Range push_texture_data(const uint8_t* data, u32 bytesPerRow, u32 rowsPerImage) {
  const auto copyBytesPerRow = AURORA_ALIGN(bytesPerRow, rhi::CopyRowAlignment);
  const auto range = map(current_frame_packet().textureUpload, copyBytesPerRow * rowsPerImage, rhi::CopyOffsetAlignment);
  u8* dst = current_frame_packet().textureUpload.data() + range.offset;
  for (u32 i = 0; i < rowsPerImage; ++i) {
    memcpy(dst, data, bytesPerRow);
    data += bytesPerRow;
    dst += copyBytesPerRow;
  }
  return range;
}

BindGroupRef bind_group_ref(const TextureBinding* bindings, uint32_t count, rhi::BindGroupLayout* layout,
                            rhi::TextureView* emptyView, rhi::Sampler* emptySampler) {
  std::array<const void*, gx::MaxTextures * 2> key{};
  for (uint32_t i = 0; i < count; ++i) {
    key[i * 2] = bindings[i].texture.get();
    key[i * 2 + 1] = bindings[i].sampler;
  }
  const BindGroupRef id = xxh3_hash_s(key.data(), sizeof(key), 0);
  std::lock_guard lock{g_bindGroupCacheMutex};
  const auto it = g_cachedBindGroups.find(id);
  if (it != g_cachedBindGroups.end()) {
    it->second.lastUsedFrame = g_frameIndex;
    return id;
  }
  CachedBindGroup entry;
  entry.lastUsedFrame = g_frameIndex;
  std::array<rhi::BindingValue, gx::MaxTextures * 2> values{};
  for (uint32_t i = 0; i < count; ++i) {
    values[i * 2].binding = i * 2;
    values[i * 2].view = bindings[i].texture ? bindings[i].texture->sampleTextureView.get() : emptyView;
    values[i * 2 + 1].binding = i * 2 + 1;
    values[i * 2 + 1].sampler = bindings[i].sampler != nullptr ? bindings[i].sampler : emptySampler;
    entry.textures[i] = bindings[i].texture;
  }
  entry.bindGroup = g_device->create_bind_group({layout, values.data(), count * 2});
  g_cachedBindGroups.emplace(id, std::move(entry));
  return id;
}

void bind_texture_group(BindGroupRef id, rhi::CommandList& list) {
  if (id == g_currentTextureBindGroup) {
    return;
  }
  rhi::BindGroup* group;
  {
    std::lock_guard lock{g_bindGroupCacheMutex};
    const auto it = g_cachedBindGroups.find(id);
    CHECK(it != g_cachedBindGroups.end(), "bind_texture_group: failed to locate {:x}", id);
    group = it->second.bindGroup.get();
  }
  list.set_bind_group(2, *group, nullptr, 0);
  g_currentTextureBindGroup = id;
}

void bind_index_buffer(rhi::CommandList& list) {
  if (!g_indexBufferBound) {
    list.set_index_buffer_u16(*g_indexBuffer, 0);
    g_indexBufferBound = true;
  }
}

rhi::Sampler* sampler_ref(const rhi::SamplerDesc& descriptor) {
  const SamplerRef id = xxh3_hash_s(&descriptor, sizeof(descriptor), 0);
  std::lock_guard lock{g_samplerCacheMutex};
  auto it = g_cachedSamplers.find(id);
  if (it == g_cachedSamplers.end()) {
    it = g_cachedSamplers.try_emplace(id, g_device->create_sampler(descriptor)).first;
  }
  return it->second.get();
}


} // namespace aurora::gfx

const AuroraStats* aurora_get_stats() { return &aurora::gfx::g_stats; }

