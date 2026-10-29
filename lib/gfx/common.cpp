#include "common.hpp"

#include "clear.hpp"
#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
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

using webgpu::g_device;
using webgpu::g_instance;
using webgpu::g_queue;


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
} // namespace aurora::gfx

namespace aurora {
// For types that we can't ensure are safe to hash with has_unique_object_representations,
// we create specialized methods to handle them. Note that these are highly dependent on
// the structure definition, which could easily change with Dawn updates.
template <>
inline HashType xxh3_hash(const WGPUBindGroupDescriptor& input, HashType seed) {
  constexpr auto offset = offsetof(WGPUBindGroupDescriptor, layout); // skip nextInChain, label
  const auto hash = xxh3_hash_s(reinterpret_cast<const u8*>(&input) + offset,
                                sizeof(WGPUBindGroupDescriptor) - offset - sizeof(void*) /* skip entries */, seed);
  return xxh3_hash_s(input.entries, sizeof(WGPUBindGroupEntry) * input.entryCount, hash);
}
template <>
inline HashType xxh3_hash(const wgpu::SamplerDescriptor& input, HashType seed) {
  constexpr auto offset = offsetof(wgpu::SamplerDescriptor, addressModeU); // skip nextInChain, label
  return xxh3_hash_s(reinterpret_cast<const u8*>(&input) + offset,
                     sizeof(wgpu::SamplerDescriptor) - offset - 2 /* skip padding */, seed);
}
} // namespace aurora

namespace aurora::gfx {
namespace {
struct CachedBindGroup {
  wgpu::BindGroup bindGroup;
  uint32_t lastUsedFrame = 0;
};

constexpr uint32_t BindGroupCacheRetainFrames = 32;
constexpr uint32_t BindGroupCacheSweepPeriod = 16;
} // namespace

static absl::flat_hash_map<BindGroupRef, CachedBindGroup> g_cachedBindGroups;
static absl::flat_hash_map<SamplerRef, wgpu::Sampler> g_cachedSamplers;
static std::mutex g_bindGroupCacheMutex;
static std::mutex g_samplerCacheMutex;

wgpu::Buffer g_vertexBuffer;
wgpu::Buffer g_uniformBuffer;
wgpu::Buffer g_indexBuffer;
wgpu::Buffer g_storageBuffer;
enum class BufferMapState {
  Unmapped,
  Mapping,
  Mapped,
};
static std::array<wgpu::Buffer, StagingBufferCount> g_stagingBuffers;
static std::array<std::atomic<BufferMapState>, StagingBufferCount> s_mappingStates;
static wgpu::Limits g_cachedLimits;
static uint32_t g_frameIndex = UINT32_MAX;
static PipelineRef g_currentPipeline;
// Texture group bound at slot 2 (0: the empty one) and whether the index
// buffer is bound, so draws only re-set what changed.
static BindGroupRef g_currentTextureBindGroup;
static bool g_indexBufferBound;
wgpu::BindGroupLayout g_staticBindGroupLayout;
wgpu::BindGroup g_staticBindGroup;
wgpu::BindGroupLayout g_uniformBindGroupLayout;
wgpu::BindGroup g_uniformBindGroup;

// for imgui debug
AuroraStats g_stats{};
uint32_t g_drawCallCount = 0;
uint32_t g_mergedDrawCallCount = 0;
uint32_t g_droppedDrawCallCount = 0;

using CommandList = std::vector<Command>;

// A replayed pass is runs of GX draws in render bundles, separated by the
// commands bundles cannot hold.
struct ReplayStep {
  enum class Kind : uint8_t {
    Bundle,
    Command,       // encoded on the pass as recorded (viewport, scissor, non-GX draws)
    BlendConstant, // a GX draw's destination alpha
  };
  Kind kind = Kind::Bundle;
  wgpu::RenderBundle bundle;
  const Command* command = nullptr;
  float alpha = 0.f;
};

struct RenderPass {
  std::string label;
  wgpu::TextureView colorView;
  wgpu::TextureView resolveView; // MSAA resolve target; null if msaaSamples == 1
  wgpu::TextureView depthStencilView;
  wgpu::Texture copySourceTexture;
  wgpu::TextureView copySourceView;
  wgpu::TextureView copySourceDepthView;
  wgpu::Extent3D targetSize;
  uint32_t msaaSamples = 1;

  TextureHandle resolveTarget;
  GXTexFmt resolveFormat = GX_TF_RGBA8;
  bool resolveOpaqueAlpha = false;
  ClipRect resolveRect;
  Range resolveUniformRange;
  Vec4<float> clearColorValue{0.f, 0.f, 0.f, 0.f};
  float clearDepthValue = gx::UseReversedZ ? 0.f : 1.f;
  wgpu::LoadOp colorLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp colorStoreOp = wgpu::StoreOp::Store;
  wgpu::LoadOp depthLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp depthStoreOp = wgpu::StoreOp::Store;
  wgpu::LoadOp stencilLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp stencilStoreOp = wgpu::StoreOp::Undefined;
  uint32_t stencilClearValue = 0;
  CommandList commands;
  bool clearColor = true;
  bool clearDepth = true;
  bool hasDepth = true;
  bool hasStencil = false;
  bool observable = false;
  bool sealed = false;
  // Targets the EFB formats, so interpolated replays can run its commands
  // from render bundles.
  bool bundleable = true;
  std::vector<tex_palette_conv::ConvRequest> paletteConvs;
  // Built by the first replay of the frame.
  std::vector<ReplayStep> replaySteps;
  bool replayBuilt = false;
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
  wgpu::CommandEncoder encoder;
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

static std::atomic_bool g_processEventsQueued = false;

static void process_events() {
  if (g_instance) {
    g_instance.ProcessEvents();
  }
}

static void enqueue_process_events() {
  if (render_worker::is_worker_thread()) {
    process_events();
    return;
  }

  bool expected = false;
  if (!g_processEventsQueued.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                                     std::memory_order_acquire)) {
    return;
  }

  render_worker::enqueue_work([] {
    process_events();
    g_processEventsQueued.store(false, std::memory_order_release);
  });
}

static void wait_for_gpu_progress(std::chrono::nanoseconds sleepDuration) {
  if (render_worker::is_idle()) {
    enqueue_process_events();
  }
  std::this_thread::sleep_for(sleepDuration);
}

static void map_staging_buffer(size_t slot, bool releaseSlotOnCompletion = false) {
  auto expected = BufferMapState::Unmapped;
  if (!s_mappingStates[slot].compare_exchange_strong(expected, BufferMapState::Mapping, std::memory_order_acq_rel,
                                                     std::memory_order_acquire)) {
    return;
  }

  g_stagingBuffers[slot].MapAsync(
      wgpu::MapMode::Write, 0, StagingBufferSize, wgpu::CallbackMode::AllowSpontaneous,
      [slot, releaseSlotOnCompletion](wgpu::MapAsyncStatus status, wgpu::StringView message) {
        if (status == wgpu::MapAsyncStatus::CallbackCancelled || status == wgpu::MapAsyncStatus::Aborted) {
          Log.warn("Buffer mapping {}: {}", magic_enum::enum_name(status), message);
          s_mappingStates[slot].store(BufferMapState::Unmapped, std::memory_order_release);
          if (releaseSlotOnCompletion) {
            g_stagingSlots.release(slot);
          }
          return;
        }
        ASSERT(status == wgpu::MapAsyncStatus::Success, "Buffer mapping failed: {} {}", magic_enum::enum_name(status),
               message);
        s_mappingStates[slot].store(BufferMapState::Mapped, std::memory_order_release);
        if (releaseSlotOnCompletion) {
          g_stagingSlots.release(slot);
        }
      });
}

static void set_efb_targets(RenderPass& pass) {
  pass.colorView = webgpu::g_frameBuffer.view;
  pass.resolveView = webgpu::g_graphicsConfig.msaaSamples > 1 ? webgpu::g_frameBufferResolved.view : nullptr;
  pass.depthStencilView = webgpu::g_depthBuffer.view;
  pass.copySourceTexture =
      webgpu::g_graphicsConfig.msaaSamples > 1 ? webgpu::g_frameBufferResolved.texture : webgpu::g_frameBuffer.texture;
  pass.copySourceView =
      webgpu::g_graphicsConfig.msaaSamples > 1 ? webgpu::g_frameBufferResolved.view : webgpu::g_frameBuffer.view;
  pass.copySourceDepthView = webgpu::g_depthBuffer.view;
  pass.targetSize = webgpu::g_frameBuffer.size;
  pass.msaaSamples = webgpu::g_graphicsConfig.msaaSamples;
  pass.hasDepth = true;
  pass.hasStencil = false;
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

static void encode_op(wgpu::CommandEncoder& cmd, FramePacket& frame, const FrameOp& op);
static void render(wgpu::CommandEncoder& cmd, FramePacket& frame, RenderPass& passInfo, uint32_t passIndex);
static void render_pass(const wgpu::RenderPassEncoder& pass, FramePacket& frame, RenderPass& passInfo);
static void expire_cached_bind_groups();
static void push_command(CommandType type, const Command::Data& data);
static void replay_frame(wgpu::CommandEncoder& cmd, FramePacket& frame);

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
    encode_op(packet.encoder, packet, op);
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
                               wgpu::TexelCopyTextureInfo tex, wgpu::Extent3D size) {
  const auto copyBytesPerRow = AURORA_ALIGN(bytesPerRow, 256);
  auto& frame = current_frame_packet();
  if (frame.textureUpload.size() + copyBytesPerRow * rowsPerImage <= TextureUploadSize) {
    const auto range = push_texture_data(data, bytesPerRow, rowsPerImage);
    const wgpu::TexelCopyBufferLayout layout{
        .offset = range.offset,
        .bytesPerRow = bytesPerRow,
        .rowsPerImage = rowsPerImage,
    };
    queue_texture_upload(TextureUpload{layout, std::move(tex), size});
    return;
  }

  const uint64_t uploadSize = copyBytesPerRow * rowsPerImage;
  const wgpu::BufferDescriptor descriptor{
      .label = "Overflow Texture Upload Buffer",
      .usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc,
      .size = uploadSize,
      .mappedAtCreation = true,
  };
  auto buffer = g_device.CreateBuffer(&descriptor);
  auto* dst = static_cast<uint8_t*>(buffer.GetMappedRange(0, uploadSize));
  for (uint32_t row = 0; row < rowsPerImage; ++row) {
    memcpy(dst, data, bytesPerRow);
    data += bytesPerRow;
    dst += copyBytesPerRow;
  }
  buffer.Unmap();

  const wgpu::TexelCopyBufferLayout layout{
      .offset = 0,
      .bytesPerRow = bytesPerRow,
      .rowsPerImage = rowsPerImage,
  };
  queue_texture_upload(TextureUpload{layout, std::move(tex), size, std::move(buffer)});
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
  return find_pipeline(ShaderType::Clear, config, [=] { return create_pipeline(config); });
}

void resolve_pass(TextureHandle texture, ClipRect rect, bool clearColor, bool clearAlpha, bool clearDepth,
                  Vec4<float> clearColorValue, float clearDepthValue, GXTexFmt resolveFormat, bool opaqueAlpha) {
  // Resolve current render pass
  auto& prevPass = current_render_passes()[g_currentRenderPass];
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
      .label = "EFB",
      .colorView = prevPass.colorView,
      .resolveView = prevPass.resolveView,
      .depthStencilView = prevPass.depthStencilView,
      .copySourceTexture = prevPass.copySourceTexture,
      .copySourceView = prevPass.copySourceView,
      .copySourceDepthView = prevPass.copySourceDepthView,
      .targetSize = prevPass.targetSize,
      .msaaSamples = msaaSamples,
      .clearColorValue = clearColorValue,
      .clearDepthValue = clearDepthValue,
      .clearColor = fullColorClear,
      .clearDepth = clearDepth && fullTarget,
      .hasDepth = prevPass.hasDepth,
      .hasStencil = prevPass.hasStencil,
      .bundleable = prevPass.bundleable,
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
        .color =
            wgpu::Color{
                .r = clearColorValue.x(),
                .g = clearColorValue.y(),
                .b = clearColorValue.z(),
                .a = clearColorValue.w(),
            },
        .depth = clearDepthValue,
        .rect = clearRect,
    });
  }
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_cachedScissor});
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

void initialize() {
  g_frameIndex = 0;
  g_processEventsQueued.store(false, std::memory_order_release);
  render_worker::initialize();
  // This appears to take a while and blocks the render thread for periods of time
  // render_worker::set_event_pump([] {
  //   if (g_instance) {
  //     g_instance.ProcessEvents();
  //   }
  // });
  tex_copy_conv::initialize();
  tex_palette_conv::initialize();
  texture_replacement::initialize();

  // For uniform & storage buffer offset alignments
  g_device.GetLimits(&g_cachedLimits);

  const auto createBuffer = [](wgpu::Buffer& out, wgpu::BufferUsage usage, uint64_t size, const char* label) {
    if (size <= 0) {
      return;
    }
    const wgpu::BufferDescriptor descriptor{
        .label = label,
        .usage = usage,
        .size = size,
    };
    out = g_device.CreateBuffer(&descriptor);
  };
  createBuffer(g_uniformBuffer, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, UniformBufferSize,
               "Shared Uniform Buffer");
  createBuffer(g_vertexBuffer, wgpu::BufferUsage::Storage | wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst,
               VertexBufferSize, "Shared Vertex Buffer");
  createBuffer(g_indexBuffer, wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst, IndexBufferSize,
               "Shared Index Buffer");
  createBuffer(g_storageBuffer, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst, StorageBufferSize,
               "Shared Storage Buffer");
  for (size_t i = 0; i < g_stagingBuffers.size(); ++i) {
    const auto label = fmt::format("Staging Buffer {}", i);
    createBuffer(g_stagingBuffers[i], wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc, StagingBufferSize,
                 label.c_str());
  }
  for (auto& state : s_mappingStates) {
    state.store(BufferMapState::Unmapped, std::memory_order_release);
  }
  for (size_t slot = 0; slot < g_stagingBuffers.size(); ++slot) {
    map_staging_buffer(slot);
  }

  {
    constexpr std::array layoutEntries{
        // Vertex data buffer
        wgpu::BindGroupLayoutEntry{
            .binding = 0,
            .visibility = wgpu::ShaderStage::Vertex,
            .buffer =
                wgpu::BufferBindingLayout{
                    .type = wgpu::BufferBindingType::ReadOnlyStorage,
                },
        },
        // Storage data buffer (the fragment stage reads the fog range table)
        wgpu::BindGroupLayoutEntry{
            .binding = 1,
            .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
            .buffer =
                wgpu::BufferBindingLayout{
                    .type = wgpu::BufferBindingType::ReadOnlyStorage,
                },
        },
    };
    const wgpu::BindGroupLayoutDescriptor layoutDesc{
        .label = "Static bind group layout",
        .entryCount = layoutEntries.size(),
        .entries = layoutEntries.data(),
    };
    g_staticBindGroupLayout = g_device.CreateBindGroupLayout(&layoutDesc);
    const std::array entries{
        wgpu::BindGroupEntry{
            .binding = 0,
            .buffer = g_vertexBuffer,
        },
        wgpu::BindGroupEntry{
            .binding = 1,
            .buffer = g_storageBuffer,
        },
    };
    const wgpu::BindGroupDescriptor bindGroupDescriptor{
        .label = "Static bind group",
        .layout = g_staticBindGroupLayout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    g_staticBindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
  }

  {
    constexpr std::array layoutEntries{
        // Uniform buffer (dynamic offset)
        wgpu::BindGroupLayoutEntry{
            .binding = 0,
            .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
            .buffer =
                wgpu::BufferBindingLayout{
                    .type = wgpu::BufferBindingType::Uniform,
                    .hasDynamicOffset = true,
                },
        },
    };
    const wgpu::BindGroupLayoutDescriptor layoutDesc{
        .label = "Uniform bind group layout",
        .entryCount = layoutEntries.size(),
        .entries = layoutEntries.data(),
    };
    g_uniformBindGroupLayout = g_device.CreateBindGroupLayout(&layoutDesc);
    const std::array entries{
        wgpu::BindGroupEntry{
            .binding = 0,
            .buffer = g_uniformBuffer,
            .size = gx::MaxUniformSize,
        },
    };
    const wgpu::BindGroupDescriptor bindGroupDescriptor{
        .label = "Uniform bind group",
        .layout = g_uniformBindGroupLayout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    g_uniformBindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
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
  g_processEventsQueued.store(false, std::memory_order_release);
  shutdown_pipeline_cache();
  tex_copy_conv::shutdown();
  tex_palette_conv::shutdown();
  texture_replacement::shutdown();
  gx::shutdown();

  {
    std::lock_guard lock{g_bindGroupCacheMutex};
    g_cachedBindGroups.clear();
  }
  {
    std::lock_guard lock{g_samplerCacheMutex};
    g_cachedSamplers.clear();
  }
  g_vertexBuffer = {};
  g_uniformBuffer = {};
  g_indexBuffer = {};
  g_storageBuffer = {};
  g_stagingBuffers.fill({});
  for (auto& packet : g_framePackets) {
    packet = {};
  }
  g_recordingFrame = nullptr;
  g_currentRenderPass = UINT32_MAX;
  g_staticBindGroup = {};
  g_staticBindGroupLayout = {};
  g_uniformBindGroup = {};
  g_uniformBindGroupLayout = {};
  g_frameIndex = UINT32_MAX;
  g_frameSlots.reset();
  g_stagingSlots.reset();
  for (auto& state : s_mappingStates) {
    state.store(BufferMapState::Unmapped, std::memory_order_release);
  }
}

static bool wait_for_staging_buffer(size_t slot) {
  map_staging_buffer(slot);
  while (true) {
    const auto mappingState = s_mappingStates[slot].load(std::memory_order_acquire);
    if (mappingState == BufferMapState::Mapped) {
      return true;
    }
    if (mappingState == BufferMapState::Unmapped) {
      return false;
    }
    wait_for_gpu_progress(std::chrono::milliseconds{1});
  }
}

static size_t acquire_frame_slot() {
  while (true) {
    if (const auto slot = g_frameSlots.try_acquire()) {
      return *slot;
    }
    wait_for_gpu_progress(std::chrono::microseconds{100});
  }
}

static std::optional<size_t> acquire_mapped_staging_buffer() {
  while (true) {
    if (auto slot = g_stagingSlots.try_acquire()) {
      if (wait_for_staging_buffer(*slot)) {
        return *slot;
      }
      g_stagingSlots.release(*slot);
      return std::nullopt;
    }
    wait_for_gpu_progress(std::chrono::microseconds{100});
  }
}

bool begin_frame() {
  const size_t frameSlot = acquire_frame_slot();
  const auto stagingSlot = acquire_mapped_staging_buffer();
  if (!stagingSlot) {
    g_frameSlots.release(frameSlot);
    return false;
  }

  auto& frame = g_framePackets[frameSlot];
  frame = {};
  frame.frameId = g_nextFrameId++;
  frame.frameIndex = g_frameIndex;
  frame.stagingBuffer = *stagingSlot;
  if (interp::enabled()) {
    frame.uniformShadow.reserve_extra(g_lastUniformShadowSize);
  }
  g_recordingFrame = &frame;
  g_recordingFrameSlot = frameSlot;

  size_t bufferOffset = 0;
  const auto& stagingBuf = g_stagingBuffers[*stagingSlot];
  const auto mapBuffer = [&](ByteBuffer& buf, uint64_t size) {
    if (size <= 0) {
      return;
    }
    buf = ByteBuffer{static_cast<u8*>(stagingBuf.GetMappedRange(bufferOffset, size)), static_cast<size_t>(size)};
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
  pass.label = "EFB";
  set_efb_targets(pass);
  pass.clearColorValue = gx::g_gxState.clearColor;
  pass.clearDepthValue = gx::clear_depth_value();
  g_currentRenderPass = 0;
  // Refresh render viewport/scissor from logical in case FB size changed
  g_cachedViewport = gx::map_logical_viewport(gx::g_gxState.logicalViewport);
  g_cachedScissor = gx::map_logical_scissor(gx::g_gxState.logicalScissor);
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_cachedScissor});
  render_worker::enqueue_begin_frame(frame.frameId, [frameSlot] {
    static constexpr wgpu::CommandEncoderDescriptor EncoderDescriptor{.label = "Redraw encoder"};
    g_framePackets[frameSlot].encoder = g_device.CreateCommandEncoder(&EncoderDescriptor);
  });
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
    g_stagingBuffers[stagingSlot].Unmap();
    s_mappingStates[stagingSlot].store(BufferMapState::Unmapped, std::memory_order_release);
    auto encoder = std::move(packet.encoder);
    const auto stats = packet.stats;
    if (interp::enabled() && callback) {
      // The frame is submitted, for its copies and side effects, when
      // interpolation reaches it; what is shown are its replays. Its slot is
      // held until then.
      auto retained = std::make_shared<FramePacket>();
      retained->renderPasses = std::move(packet.renderPasses);
      retained->ops = std::move(packet.ops);
      g_lastUniformShadowSize = packet.uniformShadow.size();
      auto uniforms = std::move(packet.uniformShadow);
      auto draws = std::move(packet.interpDraws);
      auto cameras = std::move(packet.interpCameras);
      packet = {};
      interp::push_frame(
          std::move(uniforms), std::move(draws), std::move(cameras), [retained](wgpu::CommandEncoder& cmd) { replay_frame(cmd, *retained); },
          std::move(callback), [frameSlot, stagingSlot, encoder = std::move(encoder), stats]() mutable {
            const wgpu::CommandBufferDescriptor cmdBufDescriptor{.label = "Frame command buffer"};
            const auto buffer = encoder.Finish(&cmdBufDescriptor);
            webgpu::g_queue.Submit(1, &buffer);
            encoder = {};
            publish_frame_stats(stats);
            g_frameSlots.release(frameSlot);
            expire_cached_bind_groups();
            map_staging_buffer(stagingSlot, true);
            process_events();
          });
      return;
    }
    packet = {};
    publish_frame_stats(stats);
    if (callback) {
      callback(encoder);
    }
    g_frameSlots.release(frameSlot);
    expire_cached_bind_groups();
    map_staging_buffer(stagingSlot, true);
    process_events();
  });
}

uint32_t current_frame() noexcept { return g_frameIndex; }

static void expire_cached_bind_groups() {
  std::lock_guard lock{g_bindGroupCacheMutex};
  if (g_cachedBindGroups.empty() || g_frameIndex == UINT32_MAX || g_frameIndex % BindGroupCacheSweepPeriod != 0) {
    return;
  }

  for (auto it = g_cachedBindGroups.begin(); it != g_cachedBindGroups.end();) {
    if (g_frameIndex - it->second.lastUsedFrame > BindGroupCacheRetainFrames) {
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

static constexpr uint32_t align_down_copy_offset(uint32_t value) noexcept { return value & ~3u; }

static void copy_staging_buffer_range(wgpu::CommandEncoder& cmd, const FramePacket& frame, uint32_t& copied,
                                      uint32_t highWater, uint64_t stagingOffset, const wgpu::Buffer& dst) {
  if (highWater <= copied) {
    return;
  }
  const uint32_t copyStart = align_down_copy_offset(copied);
  const uint32_t copyEnd = AURORA_ALIGN(highWater, 4);
  cmd.CopyBufferToBuffer(g_stagingBuffers[frame.stagingBuffer], stagingOffset + copyStart, dst, copyStart,
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

static void copy_staging_to_high_water(wgpu::CommandEncoder& cmd, FramePacket& frame, const FrameOp& op) {
  if (!needs_staging_copy(frame, op)) {
    return;
  }
  const auto& highWater = op.highWater;
  copy_staging_buffer_range(cmd, frame, frame.copied.verts, highWater.verts, VertexStagingOffset, g_vertexBuffer);
  copy_staging_buffer_range(cmd, frame, frame.copied.uniforms, highWater.uniforms, UniformStagingOffset,
                            g_uniformBuffer);
  copy_staging_buffer_range(cmd, frame, frame.copied.indices, highWater.indices, IndexStagingOffset, g_indexBuffer);
  copy_staging_buffer_range(cmd, frame, frame.copied.storage, highWater.storage, StorageStagingOffset, g_storageBuffer);

  if constexpr (UseTextureBuffer) {
    for (size_t i = frame.copied.textureUploadCount; i < op.textureUploads.size(); ++i) {
      const auto& item = *op.textureUploads[i];
      const wgpu::TexelCopyBufferInfo buf{
          .layout =
              wgpu::TexelCopyBufferLayout{
                  .offset = item.buffer ? item.layout.offset : item.layout.offset + TextureUploadStagingOffset,
                  .bytesPerRow = AURORA_ALIGN(item.layout.bytesPerRow, 256),
                  .rowsPerImage = item.layout.rowsPerImage,
              },
          .buffer = item.buffer ? item.buffer : g_stagingBuffers[frame.stagingBuffer],
      };
      cmd.CopyBufferToTexture(&buf, &item.tex, &item.size);
    }
    frame.copied.textureUpload = highWater.textureUpload;
    frame.copied.textureUploadCount = op.textureUploads.size();
  }
}

static void encode_op(wgpu::CommandEncoder& cmd, FramePacket& frame, const FrameOp& op) {
  copy_staging_to_high_water(cmd, frame, op);
  if (op.renderPass != nullptr) {
    render(cmd, frame, *op.renderPass, op.index);
  }
}

static void render(wgpu::CommandEncoder& cmd, FramePacket& frame, RenderPass& passInfo, uint32_t passIndex) {
  if (!passInfo.sealed) {
    return;
  }

  for (const auto& conv : passInfo.paletteConvs) {
    tex_palette_conv::run(cmd, conv);
  }
  if (!passInfo.observable && !passInfo.resolveTarget) {
    // Skip intermediate EFB render passes without observable output.
    return;
  }

  const std::array attachments{
      wgpu::RenderPassColorAttachment{
          .view = passInfo.colorView,
          .resolveTarget = passInfo.resolveView,
          .loadOp = passInfo.colorLoadOp != wgpu::LoadOp::Undefined
                        ? passInfo.colorLoadOp
                        : (passInfo.clearColor ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load),
          .storeOp = passInfo.colorStoreOp,
          .clearValue =
              {
                  .r = passInfo.clearColorValue.x(),
                  .g = passInfo.clearColorValue.y(),
                  .b = passInfo.clearColorValue.z(),
                  .a = passInfo.clearColorValue.w(),
              },
      },
  };
  wgpu::RenderPassDepthStencilAttachment depthStencilAttachment{};
  const wgpu::RenderPassDepthStencilAttachment* depthStencilAttachmentPtr = nullptr;
  if (passInfo.depthStencilView) {
    depthStencilAttachment = {
        .view = passInfo.depthStencilView,
        .depthLoadOp = passInfo.hasDepth ? (passInfo.depthLoadOp != wgpu::LoadOp::Undefined
                                                ? passInfo.depthLoadOp
                                                : (passInfo.clearDepth ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load))
                                         : wgpu::LoadOp::Undefined,
        .depthStoreOp = passInfo.hasDepth ? passInfo.depthStoreOp : wgpu::StoreOp::Undefined,
        .depthClearValue = passInfo.clearDepthValue,
        .stencilLoadOp = passInfo.hasStencil ? passInfo.stencilLoadOp : wgpu::LoadOp::Undefined,
        .stencilStoreOp = passInfo.hasStencil ? passInfo.stencilStoreOp : wgpu::StoreOp::Undefined,
        .stencilClearValue = passInfo.stencilClearValue,
    };
    depthStencilAttachmentPtr = &depthStencilAttachment;
  }
  const auto label = passInfo.label.empty() ? fmt::format("Render pass {}", passIndex)
                                            : fmt::format("{} {}", passInfo.label, passIndex);
  const wgpu::RenderPassDescriptor renderPassDescriptor{
      .label = label.c_str(),
      .colorAttachmentCount = attachments.size(),
      .colorAttachments = attachments.data(),
      .depthStencilAttachment = depthStencilAttachmentPtr,
  };

  auto pass = cmd.BeginRenderPass(&renderPassDescriptor);
  render_pass(pass, frame, passInfo);
  pass.End();

  if (passInfo.resolveTarget) {
    const auto& dstSize = passInfo.resolveTarget->size;
    const bool needsConversion = tex_copy_conv::needs_conversion(passInfo.resolveFormat);
    const bool needsScaling = dstSize.width != static_cast<uint32_t>(passInfo.resolveRect.width) ||
                              dstSize.height != static_cast<uint32_t>(passInfo.resolveRect.height);
    const bool isDepth = gx::is_depth_format(passInfo.resolveFormat);
    // A plain copy only where the target matches the EFB and keeps its alpha.
    const bool sameFormat = passInfo.resolveTarget->format == webgpu::g_graphicsConfig.surfaceConfiguration.format;
    const tex_copy_conv::ConvRequest convReq{
        .fmt = passInfo.resolveFormat,
        .srcView = isDepth ? passInfo.copySourceDepthView : passInfo.copySourceView,
        .uniformRange = passInfo.resolveUniformRange,
        .dst = passInfo.resolveTarget,
        .sampleFilter = needsScaling ? tex_copy_conv::SampleFilter::Linear : tex_copy_conv::SampleFilter::Nearest,
        .multisampled = isDepth && passInfo.msaaSamples > 1,
    };
    if (needsConversion) {
      tex_copy_conv::run(cmd, convReq);
    } else if (needsScaling || passInfo.resolveOpaqueAlpha || !sameFormat) {
      tex_copy_conv::blit(cmd, convReq);
    } else {
      const wgpu::TexelCopyTextureInfo src{
          .texture = passInfo.copySourceTexture,
          .origin =
              wgpu::Origin3D{
                  .x = static_cast<uint32_t>(passInfo.resolveRect.x),
                  .y = static_cast<uint32_t>(passInfo.resolveRect.y),
              },
      };
      const wgpu::TexelCopyTextureInfo dst{
          .texture = passInfo.resolveTarget->texture,
      };
      const wgpu::Extent3D size{
          .width = static_cast<uint32_t>(passInfo.resolveRect.width),
          .height = static_cast<uint32_t>(passInfo.resolveRect.height),
          .depthOrArrayLayers = 1,
      };
      cmd.CopyTextureToTexture(&src, &dst, &size);
    }
  }
}

static void replay_frame(wgpu::CommandEncoder& cmd, FramePacket& frame) {
  for (const auto& op : frame.ops) {
    if (op.renderPass != nullptr) {
      render(cmd, frame, *op.renderPass, op.index);
    }
  }
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

static void encode_command(const wgpu::RenderPassEncoder& pass, const RenderPass& passInfo, const Command& cmd) {
  switch (cmd.type) {
  case CommandType::SetViewport: {
    const auto& vp = cmd.data.setViewport;
    const float minDepth = gx::UseReversedZ ? 1.f - vp.zfar : vp.znear;
    const float maxDepth = gx::UseReversedZ ? 1.f - vp.znear : vp.zfar;
    pass.SetViewport(vp.left, vp.top, vp.width, vp.height, minDepth, maxDepth);
  } break;
  case CommandType::SetScissor: {
    const auto& sc = cmd.data.setScissor;
    const auto& size = passInfo.targetSize;
    const auto x = std::clamp(static_cast<uint32_t>(sc.x), 0u, size.width);
    const auto y = std::clamp(static_cast<uint32_t>(sc.y), 0u, size.height);
    const auto w = std::clamp(static_cast<uint32_t>(sc.width), 0u, size.width - x);
    const auto h = std::clamp(static_cast<uint32_t>(sc.height), 0u, size.height - y);
    pass.SetScissorRect(x, y, w, h);
  } break;
  case CommandType::Draw: {
    const auto& draw = cmd.data.draw;
    switch (draw.type) {
    case ShaderType::Clear:
      clear::render(draw.clear, pass, passInfo.targetSize);
      break;
    case ShaderType::GX:
      gx::render(draw.gx, pass);
      break;
    }
  } break;
  }
}

static void bind_pass_groups(const wgpu::RenderPassEncoder& pass) {
  g_currentPipeline = UINTPTR_MAX;
  g_currentTextureBindGroup = 0;
  g_indexBufferBound = false;
  pass.SetBindGroup(0, g_staticBindGroup);
  pass.SetBindGroup(2, gx::g_emptyTextureBindGroup);
}

// Splits the pass's commands into replay steps, recording its GX draws into
// render bundles once per frame.
static void build_replay_steps(RenderPass& passInfo) {
  const wgpu::TextureFormat colorFormat = webgpu::g_graphicsConfig.surfaceConfiguration.format;
  const wgpu::RenderBundleEncoderDescriptor descriptor{
      .label = "Replay bundle",
      .colorFormatCount = 1,
      .colorFormats = &colorFormat,
      .depthStencilFormat =
          passInfo.depthStencilView ? webgpu::g_graphicsConfig.depthFormat : wgpu::TextureFormat::Undefined,
      .sampleCount = passInfo.msaaSamples,
  };
  auto& steps = passInfo.replaySteps;
  wgpu::RenderBundleEncoder bundle;
  const auto flush = [&] {
    if (bundle) {
      steps.push_back({.kind = ReplayStep::Kind::Bundle, .bundle = bundle.Finish()});
      bundle = {};
    }
  };
  for (const auto& cmd : passInfo.commands) {
    if (cmd.type != CommandType::Draw || cmd.data.draw.type != ShaderType::GX) {
      flush();
      steps.push_back({.kind = ReplayStep::Kind::Command, .command = &cmd});
      continue;
    }
    const auto& draw = cmd.data.draw.gx;
    if (draw.dstAlpha != UINT32_MAX) {
      flush();
      steps.push_back({.kind = ReplayStep::Kind::BlendConstant, .alpha = draw.dstAlpha / 255.f});
    }
    if (!bundle) {
      bundle = g_device.CreateRenderBundleEncoder(&descriptor);
      g_currentPipeline = UINTPTR_MAX;
      g_currentTextureBindGroup = 0;
      g_indexBufferBound = false;
      bundle.SetBindGroup(0, g_staticBindGroup);
      bundle.SetBindGroup(2, gx::g_emptyTextureBindGroup);
    }
    gx::render(draw, bundle);
  }
  flush();
  passInfo.replayBuilt = true;
}

static void replay_pass(const wgpu::RenderPassEncoder& pass, RenderPass& passInfo) {
  if (!passInfo.replayBuilt) {
    build_replay_steps(passInfo);
  }
  bool bound = false;
  for (const auto& step : passInfo.replaySteps) {
    switch (step.kind) {
    case ReplayStep::Kind::Bundle:
      pass.ExecuteBundles(1, &step.bundle);
      bound = false; // bundles leave the pass's bindings unset
      break;
    case ReplayStep::Kind::Command:
      if (step.command->type == CommandType::Draw && !bound) {
        bind_pass_groups(pass);
        bound = true;
      }
      encode_command(pass, passInfo, *step.command);
      break;
    case ReplayStep::Kind::BlendConstant: {
      const wgpu::Color color{0.f, 0.f, 0.f, step.alpha};
      pass.SetBlendConstant(&color);
    } break;
    }
  }
}

static void render_pass(const wgpu::RenderPassEncoder& pass, FramePacket& frame, RenderPass& passInfo) {
  if (interp::replaying() && passInfo.bundleable) {
    replay_pass(pass, passInfo);
    return;
  }

  // Bind static bind group for the whole pass
  bind_pass_groups(pass);

  for (const auto& cmd : passInfo.commands) {
    encode_command(pass, passInfo, cmd);
  }

}

void render_pass(const wgpu::RenderPassEncoder& pass, u32 idx) {
  auto& frame = current_frame_packet();
  render_pass(pass, frame, frame.renderPasses[idx]);
}

template <typename Encoder>
static bool bind_pipeline_on(PipelineRef ref, const Encoder& pass) {
  if (ref == g_currentPipeline) {
    return true;
  }
  wgpu::RenderPipeline pipeline;
  if (!get_pipeline(ref, pipeline)) {
    ++g_stats.skippedDraws;
    return false;
  }
  pass.SetPipeline(pipeline);
  g_currentPipeline = ref;
  return true;
}

bool bind_pipeline(PipelineRef ref, const wgpu::RenderPassEncoder& pass) { return bind_pipeline_on(ref, pass); }

bool bind_pipeline(PipelineRef ref, const wgpu::RenderBundleEncoder& bundle) { return bind_pipeline_on(ref, bundle); }

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
  const auto range = push(frame.uniforms, data, length, g_cachedLimits.minUniformBufferOffsetAlignment);
  if (interp::enabled()) {
    auto& shadow = frame.uniformShadow;
    shadow.append_zeroes(range.offset - shadow.size());
    shadow.append(data, length);
  }
  return range;
}

Range push_storage(const uint8_t* data, size_t length) {
  return push(current_frame_packet().storage, data, length, g_cachedLimits.minStorageBufferOffsetAlignment);
}

bool frame_has_room(size_t verts, size_t indices, size_t uniforms, size_t storage) noexcept {
  const auto& frame = current_frame_packet();
  return frame.verts.remaining() >= verts && frame.indices.remaining() >= indices &&
         frame.uniforms.remaining() >= uniforms && frame.storage.remaining() >= storage;
}

Range push_texture_data(const uint8_t* data, u32 bytesPerRow, u32 rowsPerImage) {
  // For CopyBufferToTexture, we need an alignment of 256 per row (see Dawn kTextureBytesPerRowAlignment)
  const auto copyBytesPerRow = AURORA_ALIGN(bytesPerRow, 256);
  const auto range = map(current_frame_packet().textureUpload, copyBytesPerRow * rowsPerImage, 0);
  u8* dst = current_frame_packet().textureUpload.data() + range.offset;
  for (u32 i = 0; i < rowsPerImage; ++i) {
    memcpy(dst, data, bytesPerRow);
    data += bytesPerRow;
    dst += copyBytesPerRow;
  }
  return range;
}

BindGroupRef bind_group_ref(const WGPUBindGroupDescriptor& descriptor) {
  const auto id = xxh3_hash(descriptor);
  std::lock_guard lock{g_bindGroupCacheMutex};
  const auto it = g_cachedBindGroups.find(id);
  if (it == g_cachedBindGroups.end()) {
    auto bg = wgpu::BindGroup::Acquire(wgpuDeviceCreateBindGroup(g_device.Get(), &descriptor));
    g_cachedBindGroups.emplace(id, CachedBindGroup{
                                       .bindGroup = std::move(bg),
                                       .lastUsedFrame = g_frameIndex,
                                   });
  } else {
    it->second.lastUsedFrame = g_frameIndex;
  }
  return id;
}

template <typename Encoder>
static void bind_texture_group_on(BindGroupRef id, const Encoder& pass) {
  if (id == g_currentTextureBindGroup) {
    return;
  }
  wgpu::BindGroup group;
  {
    std::lock_guard lock{g_bindGroupCacheMutex};
    const auto it = g_cachedBindGroups.find(id);
    CHECK(it != g_cachedBindGroups.end(), "bind_texture_group: failed to locate {:x}", id);
    group = it->second.bindGroup;
  }
  pass.SetBindGroup(2, group);
  g_currentTextureBindGroup = id;
}

void bind_texture_group(BindGroupRef id, const wgpu::RenderPassEncoder& pass) { bind_texture_group_on(id, pass); }

void bind_texture_group(BindGroupRef id, const wgpu::RenderBundleEncoder& bundle) {
  bind_texture_group_on(id, bundle);
}

template <typename Encoder>
static void bind_index_buffer_on(const Encoder& pass) {
  if (!g_indexBufferBound) {
    pass.SetIndexBuffer(g_indexBuffer, wgpu::IndexFormat::Uint16);
    g_indexBufferBound = true;
  }
}

void bind_index_buffer(const wgpu::RenderPassEncoder& pass) { bind_index_buffer_on(pass); }

void bind_index_buffer(const wgpu::RenderBundleEncoder& bundle) { bind_index_buffer_on(bundle); }

wgpu::Sampler sampler_ref(const wgpu::SamplerDescriptor& descriptor) {
  const auto id = xxh3_hash(descriptor);
  std::lock_guard lock{g_samplerCacheMutex};
  auto it = g_cachedSamplers.find(id);
  if (it == g_cachedSamplers.end()) {
    it = g_cachedSamplers.try_emplace(id, g_device.CreateSampler(&descriptor)).first;
  }
  return it->second;
}

uint32_t align_uniform(uint32_t value) { return AURORA_ALIGN(value, g_cachedLimits.minUniformBufferOffsetAlignment); }

} // namespace aurora::gfx

const AuroraStats* aurora_get_stats() { return &aurora::gfx::g_stats; }

