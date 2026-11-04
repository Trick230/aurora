#pragma once

// A thin layer over D3D12 and Vulkan: the backend is picked at run time and both
// offer the same features. It checks nothing a correct caller wouldn't need.
//
// Resource states are the layer's business, as in WebGPU: callers say what they
// do with a resource, and the command list emits the barriers.

#include <cstdint>
#include <memory>
#include <string>

namespace aurora::rhi {

enum class Backend { D3D12, Vulkan };

enum class Format { RGBA8Unorm };

enum class MemoryKind {
  Device,   // fastest for the GPU, not mappable
  Upload,   // CPU writes, GPU reads
  Readback, // GPU writes, CPU reads
};

struct BufferDesc {
  uint64_t size = 0;
  MemoryKind memory = MemoryKind::Device;
};

struct TextureDesc {
  uint32_t width = 1;
  uint32_t height = 1;
  Format format = Format::RGBA8Unorm;
};

class Buffer {
public:
  virtual ~Buffer() = default;
  // Upload and Readback buffers only; stays valid until the buffer goes away.
  virtual void* mapped() = 0;
};

class Texture {
public:
  virtual ~Texture() = default;
};

struct ClearColor {
  float r, g, b, a;
};

// Rows in copies between buffers and textures are this far apart at least: the
// strictest of what the backends require.
constexpr uint32_t CopyRowAlignment = 256;

class CommandList {
public:
  virtual ~CommandList() = default;
  virtual void begin() = 0;
  virtual void end() = 0;
  virtual void begin_render_pass(Texture& target, const ClearColor& clear) = 0;
  virtual void end_render_pass() = 0;
  virtual void copy_texture_to_buffer(Texture& source, Buffer& destination, uint32_t bytes_per_row) = 0;
};

class Device {
public:
  virtual ~Device() = default;
  virtual Backend backend() const = 0;
  virtual const std::string& adapter_name() const = 0;
  virtual std::unique_ptr<Buffer> create_buffer(const BufferDesc& desc) = 0;
  virtual std::unique_ptr<Texture> create_texture(const TextureDesc& desc) = 0;
  virtual std::unique_ptr<CommandList> create_command_list() = 0;
  virtual void submit(CommandList& list) = 0;
  virtual void wait_idle() = 0;
};

// nullptr, with the reason in `error`, if this machine can't run the backend.
std::unique_ptr<Device> create_device(Backend backend, std::string* error);

} // namespace aurora::rhi
