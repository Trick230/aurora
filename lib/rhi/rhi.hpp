#pragma once

// A thin layer over D3D12 and Vulkan: the backend is picked at run time and both
// offer the same features. It checks nothing a correct caller wouldn't need.
//
// States are the layer's business. A texture rests readable by shaders, and a
// buffer readable by everything; a pass or a copy takes it from there and puts it
// back, so nothing needs a barrier inside a render pass. Commands must be
// submitted in the order they were recorded.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aurora::rhi {

enum class Backend { D3D12, Vulkan };

enum class Format {
  Undefined,
  R8Unorm,
  RG8Unorm,
  RGBA8Unorm,
  RGBA8Srgb,
  BGRA8Unorm,
  BGRA8Srgb,
  R16Sint,
  R32Float,
  Depth32Float,
  BC1,
  BC2,
  BC3,
  BC4,
  BC5,
  BC6H,
  BC7,
};

enum class MemoryKind {
  Device,   // fastest for the GPU, not mappable
  Upload,   // CPU writes, GPU reads
  Readback, // GPU writes, CPU reads
};

struct BufferDesc {
  uint64_t size = 0;
  MemoryKind memory = MemoryKind::Device;
};

class Buffer {
public:
  virtual ~Buffer() = default;
  // Upload and Readback buffers only; stays valid until the buffer goes away.
  virtual void* mapped() = 0;
};

enum TextureUsage : uint32_t {
  TextureSampled = 1,
  TextureColorTarget = 2,
  TextureDepthTarget = 4,
};

struct TextureDesc {
  uint32_t width = 1;
  uint32_t height = 1;
  uint32_t mips = 1;
  uint32_t samples = 1;
  Format format = Format::RGBA8Unorm;
  uint32_t usage = TextureSampled;
};

enum class Swizzle { Identity, R, G, B, A, Zero, One };

// How a shader sees a texture.
struct TextureViewDesc {
  uint32_t base_mip = 0;
  uint32_t mips = 0; // 0: to the last
  Swizzle r = Swizzle::Identity, g = Swizzle::Identity, b = Swizzle::Identity, a = Swizzle::Identity;
};

class TextureView {
public:
  virtual ~TextureView() = default;
};

class Texture {
public:
  virtual ~Texture() = default;
  virtual std::unique_ptr<TextureView> create_view(const TextureViewDesc& desc) = 0;
};

enum class Filter { Nearest, Linear };
enum class AddressMode { Repeat, MirrorRepeat, ClampToEdge };

struct SamplerDesc {
  Filter min = Filter::Linear, mag = Filter::Linear, mip = Filter::Linear;
  AddressMode u = AddressMode::Repeat, v = AddressMode::Repeat;
  float lod_bias = 0.0f;
  uint32_t anisotropy = 1;
};

class Sampler {
public:
  virtual ~Sampler() = default;
};

// In HLSL, a group is a register's space and an entry's binding its index: b for
// uniforms, t for storage buffers and textures, s for samplers.
enum class BindingKind {
  Uniform,        // cbuffer; a dynamic offset moves it at each set_bind_group
  StorageRead,    // ByteAddressBuffer or StructuredBuffer
  Texture,
  Sampler,
};

struct BindingLayout {
  uint32_t binding = 0;
  BindingKind kind = BindingKind::Texture;
  bool dynamic_offset = false;
};

class BindGroupLayout {
public:
  virtual ~BindGroupLayout() = default;
};

struct BindGroupLayoutDesc {
  const BindingLayout* entries = nullptr;
  uint32_t count = 0;
};

struct BindingValue {
  uint32_t binding = 0;
  Buffer* buffer = nullptr;
  uint64_t offset = 0;
  uint64_t size = 0;
  TextureView* view = nullptr;
  Sampler* sampler = nullptr;
};

class BindGroup {
public:
  virtual ~BindGroup() = default;
};

struct BindGroupDesc {
  BindGroupLayout* layout = nullptr;
  const BindingValue* entries = nullptr;
  uint32_t count = 0;
};

class PipelineLayout {
public:
  virtual ~PipelineLayout() = default;
};

struct PipelineLayoutDesc {
  BindGroupLayout* const* groups = nullptr;
  uint32_t count = 0;
};

enum class ShaderStage { Vertex, Fragment };

// What the backend's API takes: DXIL or SPIR-V.
using ShaderCode = std::vector<uint8_t>;

// The same HLSL for every backend.
bool compile_shader(Backend backend, ShaderStage stage, const std::string& source, const std::string& entry,
                    ShaderCode* code, std::string* error);

enum class BlendFactor {
  Zero,
  One,
  Src,
  OneMinusSrc,
  SrcAlpha,
  OneMinusSrcAlpha,
  Dst,
  OneMinusDst,
  DstAlpha,
  OneMinusDstAlpha,
  Constant,
  Src1Alpha,
  OneMinusSrc1Alpha,
};
enum class BlendOp { Add, ReverseSubtract };
enum class Compare { Never, Less, Equal, LessEqual, Greater, NotEqual, GreaterEqual, Always };
enum class CullMode { None, Front, Back };

struct BlendComponent {
  BlendFactor src = BlendFactor::One;
  BlendFactor dst = BlendFactor::Zero;
  BlendOp op = BlendOp::Add;
};

enum ColorWrite : uint8_t { WriteR = 1, WriteG = 2, WriteB = 4, WriteA = 8, WriteAll = 15 };

struct RenderPipelineDesc {
  PipelineLayout* layout = nullptr;
  const ShaderCode* vertex = nullptr;
  const char* vertex_entry = "main";
  const ShaderCode* fragment = nullptr;
  const char* fragment_entry = "main";
  Format color_format = Format::RGBA8Unorm; // Undefined: no color target
  uint8_t color_write = WriteAll;
  bool blend = false;
  BlendComponent blend_color, blend_alpha;
  Format depth_format = Format::Undefined;
  bool depth_write = false;
  Compare depth_compare = Compare::Always;
  CullMode cull = CullMode::None;
  uint32_t samples = 1;
  bool alpha_to_coverage = false;
};

class RenderPipeline {
public:
  virtual ~RenderPipeline() = default;
};

// Rows in copies between buffers and textures are this far apart at least, and
// start at offsets aligned to this: the strictest of what the backends require.
constexpr uint32_t CopyRowAlignment = 256;
constexpr uint32_t CopyOffsetAlignment = 512;

struct ClearColor {
  float r, g, b, a;
};

struct ColorAttachment {
  Texture* target = nullptr;
  Texture* resolve = nullptr; // written at the end of the pass, from a multisampled target
  bool clear = false;         // otherwise the target's contents are kept
  ClearColor clear_color{0.0f, 0.0f, 0.0f, 0.0f};
  bool store = true;
};

struct DepthAttachment {
  Texture* target = nullptr;
  bool clear = false;
  float clear_depth = 1.0f;
  bool store = true;
};

struct RenderPassDesc {
  ColorAttachment color;
  DepthAttachment depth;
};

struct TextureRegion {
  uint32_t mip = 0;
  uint32_t x = 0, y = 0;
  uint32_t width = 0, height = 0;
};

class CommandList {
public:
  virtual ~CommandList() = default;
  virtual void begin() = 0;
  virtual void end() = 0;

  // Sets the viewport and scissor to the whole target. Depth runs from 0 at the
  // near plane, and clip space's y points up, on every backend.
  virtual void begin_render_pass(const RenderPassDesc& desc) = 0;
  virtual void end_render_pass() = 0;
  virtual void set_pipeline(RenderPipeline& pipeline) = 0;
  // After a pipeline with another layout, the groups have to be set again.
  virtual void set_bind_group(uint32_t index, BindGroup& group, const uint32_t* dynamic_offsets,
                              uint32_t dynamic_count) = 0;
  virtual void set_index_buffer_u16(Buffer& buffer, uint64_t offset) = 0;
  virtual void set_viewport(float x, float y, float width, float height, float min_depth, float max_depth) = 0;
  virtual void set_scissor(uint32_t x, uint32_t y, uint32_t width, uint32_t height) = 0;
  virtual void set_blend_constant(float r, float g, float b, float a) = 0;
  virtual void draw(uint32_t vertices, uint32_t first_vertex) = 0;
  virtual void draw_indexed(uint32_t indices, uint32_t first_index, int32_t base_vertex) = 0;

  // Outside render passes. A zero width or height in a region means the mip's.
  virtual void copy_buffer(Buffer& source, uint64_t source_offset, Buffer& destination, uint64_t destination_offset,
                           uint64_t size) = 0;
  virtual void copy_buffer_to_texture(Buffer& source, uint64_t offset, uint32_t bytes_per_row, Texture& destination,
                                      const TextureRegion& region) = 0;
  virtual void copy_texture_to_buffer(Texture& source, const TextureRegion& region, Buffer& destination, uint64_t offset,
                                      uint32_t bytes_per_row) = 0;
  virtual void copy_texture(Texture& source, const TextureRegion& source_region, Texture& destination,
                            uint32_t destination_x, uint32_t destination_y) = 0;
};

struct DeviceFeatures {
  uint32_t max_samples = 1; // the most both color and depth targets can have
  uint32_t uniform_offset_alignment = 256;
};

class Device {
public:
  virtual ~Device() = default;
  virtual Backend backend() const = 0;
  virtual const std::string& adapter_name() const = 0;
  virtual const DeviceFeatures& features() const = 0;
  virtual std::unique_ptr<Buffer> create_buffer(const BufferDesc& desc) = 0;
  virtual std::unique_ptr<Texture> create_texture(const TextureDesc& desc) = 0;
  virtual std::unique_ptr<Sampler> create_sampler(const SamplerDesc& desc) = 0;
  virtual std::unique_ptr<BindGroupLayout> create_bind_group_layout(const BindGroupLayoutDesc& desc) = 0;
  virtual std::unique_ptr<BindGroup> create_bind_group(const BindGroupDesc& desc) = 0;
  virtual std::unique_ptr<PipelineLayout> create_pipeline_layout(const PipelineLayoutDesc& desc) = 0;
  virtual std::unique_ptr<RenderPipeline> create_render_pipeline(const RenderPipelineDesc& desc) = 0;
  virtual std::unique_ptr<CommandList> create_command_list() = 0;
  virtual void submit(CommandList& list) = 0;
  virtual void wait_idle() = 0;
};

struct DeviceOptions {
  // The API's validation layers, where they are installed. Their messages go to
  // stderr, and validation_errors() counts them.
  bool validation = false;
};

// nullptr, with the reason in `error`, if this machine can't run the backend.
std::unique_ptr<Device> create_device(Backend backend, const DeviceOptions& options, std::string* error);

// Messages from the validation layers so far, on any device; -1 if none are active.
int64_t validation_errors();

} // namespace aurora::rhi
