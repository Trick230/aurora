#include "rhi_internal.hpp"

#include <D3D12MemAlloc.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace aurora::rhi {
namespace {

using Microsoft::WRL::ComPtr;

DXGI_FORMAT to_dxgi(Format format) {
  switch (format) {
  case Format::Undefined: return DXGI_FORMAT_UNKNOWN;
  case Format::R8Unorm: return DXGI_FORMAT_R8_UNORM;
  case Format::RG8Unorm: return DXGI_FORMAT_R8G8_UNORM;
  case Format::RGBA8Unorm: return DXGI_FORMAT_R8G8B8A8_UNORM;
  case Format::RGBA8Srgb: return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
  case Format::BGRA8Unorm: return DXGI_FORMAT_B8G8R8A8_UNORM;
  case Format::BGRA8Srgb: return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
  case Format::R16Sint: return DXGI_FORMAT_R16_SINT;
  case Format::R32Float: return DXGI_FORMAT_R32_FLOAT;
  case Format::Depth32Float: return DXGI_FORMAT_D32_FLOAT;
  case Format::BC1: return DXGI_FORMAT_BC1_UNORM;
  case Format::BC2: return DXGI_FORMAT_BC2_UNORM;
  case Format::BC3: return DXGI_FORMAT_BC3_UNORM;
  case Format::BC4: return DXGI_FORMAT_BC4_UNORM;
  case Format::BC5: return DXGI_FORMAT_BC5_UNORM;
  case Format::BC6H: return DXGI_FORMAT_BC6H_UF16;
  case Format::BC7: return DXGI_FORMAT_BC7_UNORM;
  }
  return DXGI_FORMAT_UNKNOWN;
}

D3D12_FILTER to_filter(const SamplerDesc& desc) {
  if (desc.anisotropy > 1)
    return D3D12_FILTER_ANISOTROPIC;
  const auto type = [](Filter filter) {
    return filter == Filter::Linear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT;
  };
  return D3D12_ENCODE_BASIC_FILTER(type(desc.min), type(desc.mag), type(desc.mip), D3D12_FILTER_REDUCTION_TYPE_STANDARD);
}

D3D12_TEXTURE_ADDRESS_MODE to_address(AddressMode mode) {
  switch (mode) {
  case AddressMode::Repeat: return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
  case AddressMode::MirrorRepeat: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
  case AddressMode::ClampToEdge: return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
  }
  return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
}

D3D12_BLEND to_blend(BlendFactor factor, bool alpha) {
  switch (factor) {
  case BlendFactor::Zero: return D3D12_BLEND_ZERO;
  case BlendFactor::One: return D3D12_BLEND_ONE;
  case BlendFactor::Src: return alpha ? D3D12_BLEND_SRC_ALPHA : D3D12_BLEND_SRC_COLOR;
  case BlendFactor::OneMinusSrc: return alpha ? D3D12_BLEND_INV_SRC_ALPHA : D3D12_BLEND_INV_SRC_COLOR;
  case BlendFactor::SrcAlpha: return D3D12_BLEND_SRC_ALPHA;
  case BlendFactor::OneMinusSrcAlpha: return D3D12_BLEND_INV_SRC_ALPHA;
  case BlendFactor::Dst: return alpha ? D3D12_BLEND_DEST_ALPHA : D3D12_BLEND_DEST_COLOR;
  case BlendFactor::OneMinusDst: return alpha ? D3D12_BLEND_INV_DEST_ALPHA : D3D12_BLEND_INV_DEST_COLOR;
  case BlendFactor::DstAlpha: return D3D12_BLEND_DEST_ALPHA;
  case BlendFactor::OneMinusDstAlpha: return D3D12_BLEND_INV_DEST_ALPHA;
  case BlendFactor::Constant: return D3D12_BLEND_BLEND_FACTOR;
  case BlendFactor::Src1Alpha: return D3D12_BLEND_SRC1_ALPHA;
  case BlendFactor::OneMinusSrc1Alpha: return D3D12_BLEND_INV_SRC1_ALPHA;
  }
  return D3D12_BLEND_ONE;
}

D3D12_COMPARISON_FUNC to_compare(Compare compare) {
  switch (compare) {
  case Compare::Never: return D3D12_COMPARISON_FUNC_NEVER;
  case Compare::Less: return D3D12_COMPARISON_FUNC_LESS;
  case Compare::Equal: return D3D12_COMPARISON_FUNC_EQUAL;
  case Compare::LessEqual: return D3D12_COMPARISON_FUNC_LESS_EQUAL;
  case Compare::Greater: return D3D12_COMPARISON_FUNC_GREATER;
  case Compare::NotEqual: return D3D12_COMPARISON_FUNC_NOT_EQUAL;
  case Compare::GreaterEqual: return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
  case Compare::Always: return D3D12_COMPARISON_FUNC_ALWAYS;
  }
  return D3D12_COMPARISON_FUNC_ALWAYS;
}

class D3D12Device;

// A range of a heap's descriptors, reused once a group goes away.
class DescriptorHeap {
public:
  ComPtr<ID3D12DescriptorHeap> heap;
  UINT increment = 0, capacity = 0, used = 0;
  std::map<UINT, std::vector<UINT>> free_ranges;

  bool init(ID3D12Device* device, D3D12_DESCRIPTOR_HEAP_TYPE type, UINT count, bool shader_visible) {
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = type;
    desc.NumDescriptors = count;
    desc.Flags = shader_visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    capacity = count;
    increment = device->GetDescriptorHandleIncrementSize(type);
    return SUCCEEDED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap)));
  }
  // UINT_MAX if the heap is full.
  UINT allocate(UINT count) {
    auto& ranges = free_ranges[count];
    if (!ranges.empty()) {
      const UINT at = ranges.back();
      ranges.pop_back();
      return at;
    }
    if (used + count > capacity)
      return UINT_MAX;
    const UINT at = used;
    used += count;
    return at;
  }
  void release(UINT at, UINT count) { free_ranges[count].push_back(at); }
  D3D12_CPU_DESCRIPTOR_HANDLE cpu(UINT at) const {
    D3D12_CPU_DESCRIPTOR_HANDLE handle = heap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += SIZE_T{at} * increment;
    return handle;
  }
  D3D12_GPU_DESCRIPTOR_HANDLE gpu(UINT at) const {
    D3D12_GPU_DESCRIPTOR_HANDLE handle = heap->GetGPUDescriptorHandleForHeapStart();
    handle.ptr += UINT64{at} * increment;
    return handle;
  }
};

class D3D12Buffer final : public Buffer {
public:
  D3D12Device* owner = nullptr;
  ComPtr<D3D12MA::Allocation> allocation;
  ID3D12Resource* resource = nullptr;
  D3D12_GPU_VIRTUAL_ADDRESS address = 0;
  uint64_t size = 0;
  void* pointer = nullptr;
  D3D12_RESOURCE_STATES rest = D3D12_RESOURCE_STATE_GENERIC_READ;
  ~D3D12Buffer() override;
  void* mapped() override { return pointer; }
};

class D3D12Texture final : public Texture {
public:
  D3D12Device* owner = nullptr;
  ComPtr<D3D12MA::Allocation> allocation; // none for a swapchain's
  ComPtr<ID3D12Resource> resource;
  UINT rtv = UINT_MAX, dsv = UINT_MAX;
  D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
  D3D12_RESOURCE_STATES rest = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
  DXGI_FORMAT copy_format = DXGI_FORMAT_UNKNOWN;
  Format format = Format::Undefined;
  uint32_t width = 0, height = 0, mips = 1, samples = 1;
  ~D3D12Texture() override;
  std::unique_ptr<TextureView> create_view(const TextureViewDesc& desc) override;
};

class D3D12TextureView final : public TextureView {
public:
  ComPtr<ID3D12Resource> resource;
  D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
};

class D3D12Sampler final : public Sampler {
public:
  D3D12_SAMPLER_DESC desc{};
};

class D3D12BindGroupLayout final : public BindGroupLayout {
public:
  std::vector<BindingLayout> entries;
  std::vector<uint32_t> srvs, samplers; // bindings, in table order
  std::vector<uint32_t> uniforms;       // bindings, in root parameter order
};

class D3D12BindGroup final : public BindGroup {
public:
  D3D12Device* owner = nullptr;
  UINT srv_table = UINT_MAX, srv_count = 0;
  bool has_sampler_table = false;
  D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu{}, sampler_gpu{};
  struct Uniform {
    D3D12_GPU_VIRTUAL_ADDRESS address;
    bool dynamic;
  };
  std::vector<Uniform> uniforms;
  ~D3D12BindGroup() override;
};

class D3D12PipelineLayout final : public PipelineLayout {
public:
  ComPtr<ID3D12RootSignature> root;
  struct Group {
    UINT first_uniform = 0, srv_table = UINT_MAX, sampler_table = UINT_MAX;
  };
  std::vector<Group> groups;
  UINT draw_constant = UINT_MAX;
};

class D3D12RenderPipeline final : public RenderPipeline {
public:
  ComPtr<ID3D12RootSignature> root;
  ComPtr<ID3D12PipelineState> state;
  const D3D12PipelineLayout* layout = nullptr;
};

class D3D12CommandList final : public CommandList {
public:
  D3D12Device* owner = nullptr;
  ComPtr<ID3D12CommandAllocator> allocator;
  ComPtr<ID3D12GraphicsCommandList> list;
  const D3D12PipelineLayout* layout = nullptr;
  ID3D12RootSignature* bound_root = nullptr;
  // The pass in progress, for what its end has to do.
  D3D12Texture* pass_color = nullptr;
  D3D12Texture* pass_resolve = nullptr;
  D3D12Texture* pass_depth = nullptr;

  ~D3D12CommandList() override;
  void begin() override;
  void end() override { list->Close(); }

  void transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    if (from == to)
      return;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = from;
    barrier.Transition.StateAfter = to;
    list->ResourceBarrier(1, &barrier);
  }
  void transition(D3D12Texture& texture, D3D12_RESOURCE_STATES state) {
    transition(texture.resource.Get(), texture.state, state);
    texture.state = state;
  }

  void begin_render_pass(const RenderPassDesc& desc) override;
  void end_render_pass() override;
  void set_pipeline(RenderPipeline& pipeline) override {
    auto& native = static_cast<D3D12RenderPipeline&>(pipeline);
    if (bound_root != native.root.Get()) {
      list->SetGraphicsRootSignature(native.root.Get());
      bound_root = native.root.Get();
    }
    layout = native.layout;
    list->SetPipelineState(native.state.Get());
  }
  void set_bind_group(uint32_t index, BindGroup& group, const uint32_t* dynamic_offsets,
                      uint32_t dynamic_count) override {
    (void)dynamic_count;
    auto& native = static_cast<D3D12BindGroup&>(group);
    const auto& slots = layout->groups[index];
    UINT parameter = slots.first_uniform;
    uint32_t next_dynamic = 0;
    for (const auto& uniform : native.uniforms) {
      const uint64_t offset = uniform.dynamic ? dynamic_offsets[next_dynamic++] : 0;
      list->SetGraphicsRootConstantBufferView(parameter++, uniform.address + offset);
    }
    if (slots.srv_table != UINT_MAX && native.srv_count)
      list->SetGraphicsRootDescriptorTable(slots.srv_table, native.srv_gpu);
    if (slots.sampler_table != UINT_MAX && native.has_sampler_table)
      list->SetGraphicsRootDescriptorTable(slots.sampler_table, native.sampler_gpu);
  }
  void set_index_buffer_u16(Buffer& buffer, uint64_t offset) override {
    auto& native = static_cast<D3D12Buffer&>(buffer);
    D3D12_INDEX_BUFFER_VIEW view{};
    view.BufferLocation = native.address + offset;
    view.SizeInBytes = static_cast<UINT>(native.size - offset);
    view.Format = DXGI_FORMAT_R16_UINT;
    list->IASetIndexBuffer(&view);
  }
  void set_viewport(float x, float y, float width, float height, float min_depth, float max_depth) override {
    const D3D12_VIEWPORT viewport{x, y, width, height, min_depth, max_depth};
    list->RSSetViewports(1, &viewport);
  }
  void set_scissor(uint32_t x, uint32_t y, uint32_t width, uint32_t height) override {
    const D3D12_RECT rect{static_cast<LONG>(x), static_cast<LONG>(y), static_cast<LONG>(x + width),
                          static_cast<LONG>(y + height)};
    list->RSSetScissorRects(1, &rect);
  }
  void set_blend_constant(float r, float g, float b, float a) override {
    const FLOAT factor[4] = {r, g, b, a};
    list->OMSetBlendFactor(factor);
  }
  void set_draw_constant(uint32_t value) override {
    list->SetGraphicsRoot32BitConstant(layout->draw_constant, value, 0);
  }
  void draw(uint32_t vertices, uint32_t first_vertex, uint32_t instances) override {
    list->DrawInstanced(vertices, instances, first_vertex, 0);
  }
  void draw_indexed(uint32_t indices, uint32_t first_index, int32_t base_vertex, uint32_t instances) override {
    list->DrawIndexedInstanced(indices, instances, first_index, base_vertex, 0);
  }

  void copy_buffer(Buffer& source, uint64_t source_offset, Buffer& destination, uint64_t destination_offset,
                   uint64_t size) override {
    auto& to = static_cast<D3D12Buffer&>(destination);
    transition(to.resource, to.rest, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(to.resource, destination_offset, static_cast<D3D12Buffer&>(source).resource, source_offset,
                           size);
    transition(to.resource, D3D12_RESOURCE_STATE_COPY_DEST, to.rest);
  }

  // Compressed formats copy whole blocks, even for a mip smaller than one.
  static uint32_t copy_extent(const D3D12Texture& texture, uint32_t size) {
    const bool compressed = (texture.copy_format >= DXGI_FORMAT_BC1_TYPELESS && texture.copy_format <= DXGI_FORMAT_BC5_SNORM) ||
                            (texture.copy_format >= DXGI_FORMAT_BC6H_TYPELESS && texture.copy_format <= DXGI_FORMAT_BC7_UNORM_SRGB);
    return compressed ? (size + 3) & ~3u : size;
  }

  static D3D12_TEXTURE_COPY_LOCATION footprint(D3D12Buffer& buffer, const D3D12Texture& texture, uint64_t offset,
                                               uint32_t bytes_per_row, uint32_t width, uint32_t height) {
    D3D12_TEXTURE_COPY_LOCATION location{};
    location.pResource = buffer.resource;
    location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    location.PlacedFootprint.Offset = offset;
    location.PlacedFootprint.Footprint.Format = texture.copy_format;
    location.PlacedFootprint.Footprint.Width = width;
    location.PlacedFootprint.Footprint.Height = height;
    location.PlacedFootprint.Footprint.Depth = 1;
    location.PlacedFootprint.Footprint.RowPitch = bytes_per_row;
    return location;
  }
  static D3D12_TEXTURE_COPY_LOCATION subresource(const D3D12Texture& texture, uint32_t mip) {
    D3D12_TEXTURE_COPY_LOCATION location{};
    location.pResource = texture.resource.Get();
    location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    location.SubresourceIndex = mip;
    return location;
  }

  void copy_buffer_to_texture(Buffer& source, uint64_t offset, uint32_t bytes_per_row, Texture& destination,
                              const TextureRegion& region) override {
    auto& texture = static_cast<D3D12Texture&>(destination);
    const uint32_t width = copy_extent(texture, region.width ? region.width : MipExtent(texture.width, region.mip));
    const uint32_t height = copy_extent(texture, region.height ? region.height : MipExtent(texture.height, region.mip));
    transition(texture, D3D12_RESOURCE_STATE_COPY_DEST);
    const auto to = subresource(texture, region.mip);
    const auto from = footprint(static_cast<D3D12Buffer&>(source), texture, offset, bytes_per_row, width, height);
    list->CopyTextureRegion(&to, region.x, region.y, 0, &from, nullptr);
    transition(texture, texture.rest);
  }

  void copy_texture_to_buffer(Texture& source, const TextureRegion& region, Buffer& destination, uint64_t offset,
                              uint32_t bytes_per_row) override {
    auto& texture = static_cast<D3D12Texture&>(source);
    const uint32_t width = copy_extent(texture, region.width ? region.width : MipExtent(texture.width, region.mip));
    const uint32_t height = copy_extent(texture, region.height ? region.height : MipExtent(texture.height, region.mip));
    auto& buffer = static_cast<D3D12Buffer&>(destination);
    transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(buffer.resource, buffer.rest, D3D12_RESOURCE_STATE_COPY_DEST);
    const auto from = subresource(texture, region.mip);
    const auto to = footprint(buffer, texture, offset, bytes_per_row, width, height);
    const D3D12_BOX box{region.x, region.y, 0, region.x + width, region.y + height, 1};
    list->CopyTextureRegion(&to, 0, 0, 0, &from, &box);
    transition(buffer.resource, D3D12_RESOURCE_STATE_COPY_DEST, buffer.rest);
    transition(texture, texture.rest);
  }

  void copy_texture(Texture& source, const TextureRegion& region, Texture& destination, uint32_t x,
                    uint32_t y) override {
    auto& from_texture = static_cast<D3D12Texture&>(source);
    auto& to_texture = static_cast<D3D12Texture&>(destination);
    const uint32_t width = region.width ? region.width : MipExtent(from_texture.width, region.mip);
    const uint32_t height = region.height ? region.height : MipExtent(from_texture.height, region.mip);
    transition(from_texture, D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(to_texture, D3D12_RESOURCE_STATE_COPY_DEST);
    const auto from = subresource(from_texture, region.mip);
    const auto to = subresource(to_texture, 0);
    const D3D12_BOX box{region.x, region.y, 0, region.x + width, region.y + height, 1};
    list->CopyTextureRegion(&to, x, y, 0, &from, &box);
    transition(to_texture, to_texture.rest);
    transition(from_texture, from_texture.rest);
  }
};

class D3D12Device final : public Device {
public:
  ComPtr<IDXGIFactory6> factory;
  ComPtr<ID3D12Device> device;
  ComPtr<D3D12MA::Allocator> allocator;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<ID3D12Fence> fence;
  ComPtr<ID3D12InfoQueue> info_queue;
  HANDLE fence_event = nullptr;
  UINT64 fence_value = 0;
  DescriptorHeap rtv_heap, dsv_heap, srv_heap, sampler_heap;
  // Tables of samplers are few and shared: the heap holds only 2048 samplers.
  std::map<std::vector<std::array<uint32_t, 16>>, UINT> sampler_tables;
  RetireQueue retired;
  std::mutex mutex;
  std::string name;
  DeviceFeatures device_features;

  ~D3D12Device() override {
    wait_idle();
    collect();
    if (fence_event)
      CloseHandle(fence_event);
  }

  Backend backend() const override { return Backend::D3D12; }
  const std::string& adapter_name() const override { return name; }
  const DeviceFeatures& features() const override { return device_features; }

  void retire(std::function<void()> release) {
    std::lock_guard lock{mutex};
    retired.retire(fence_value + 1, std::move(release));
  }
  void collect() { retired.collect(fence ? fence->GetCompletedValue() : 0); }
  void drain_messages() {
    if (!info_queue)
      return;
    const UINT64 count = info_queue->GetNumStoredMessages();
    for (UINT64 i = 0; i < count; ++i) {
      SIZE_T size = 0;
      info_queue->GetMessage(i, nullptr, &size);
      std::vector<char> storage(size);
      auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
      if (SUCCEEDED(info_queue->GetMessage(i, message, &size)) && message->Severity <= D3D12_MESSAGE_SEVERITY_WARNING) {
        std::fprintf(stderr, "D3D12: %s\n", message->pDescription);
        g_validation_errors.fetch_add(1);
      }
    }
    info_queue->ClearStoredMessages();
  }

  std::unique_ptr<Buffer> create_buffer(const BufferDesc& desc) override {
    D3D12MA::ALLOCATION_DESC heap{};
    heap.HeapType = desc.memory == MemoryKind::Upload     ? D3D12_HEAP_TYPE_UPLOAD
                    : desc.memory == MemoryKind::Readback ? D3D12_HEAP_TYPE_READBACK
                                                          : D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC info{};
    info.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    info.Width = desc.size;
    info.Height = 1;
    info.DepthOrArraySize = 1;
    info.MipLevels = 1;
    info.SampleDesc.Count = 1;
    info.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    auto buffer = std::make_unique<D3D12Buffer>();
    buffer->owner = this;
    buffer->size = desc.size;
    buffer->rest = desc.memory == MemoryKind::Readback ? D3D12_RESOURCE_STATE_COPY_DEST
                                                       : D3D12_RESOURCE_STATE_GENERIC_READ;
    // Buffers in the default heap are always created in the common state, which
    // reads promote from without a barrier.
    const D3D12_RESOURCE_STATES initial = desc.memory == MemoryKind::Device ? D3D12_RESOURCE_STATE_COMMON : buffer->rest;
    if (FAILED(allocator->CreateResource(&heap, &info, initial, nullptr, &buffer->allocation, IID_NULL, nullptr)))
      return nullptr;
    buffer->resource = buffer->allocation->GetResource();
    buffer->address = buffer->resource->GetGPUVirtualAddress();
    if (desc.memory != MemoryKind::Device && FAILED(buffer->resource->Map(0, nullptr, &buffer->pointer)))
      return nullptr;
    return buffer;
  }

  std::unique_ptr<Texture> create_texture(const TextureDesc& desc) override {
    D3D12MA::ALLOCATION_DESC heap{};
    heap.HeapType = D3D12_HEAP_TYPE_DEFAULT;
    // Targets get memory of their own: it starts zeroed, where a placed target would
    // have to be cleared or discarded before anything else could touch it.
    if (desc.usage & (TextureColorTarget | TextureDepthTarget))
      heap.Flags = D3D12MA::ALLOCATION_FLAG_COMMITTED;
    const bool depth = desc.format == Format::Depth32Float;
    D3D12_RESOURCE_DESC info{};
    info.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    info.Width = desc.width;
    info.Height = desc.height;
    info.DepthOrArraySize = 1;
    info.MipLevels = static_cast<UINT16>(desc.mips);
    // A depth texture a shader reads needs a format that can be viewed both ways.
    info.Format = depth ? DXGI_FORMAT_R32_TYPELESS : to_dxgi(desc.format);
    info.SampleDesc.Count = desc.samples;
    if (desc.usage & TextureColorTarget)
      info.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    if (desc.usage & TextureDepthTarget)
      info.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    auto texture = std::make_unique<D3D12Texture>();
    texture->owner = this;
    texture->format = desc.format;
    texture->copy_format = depth ? DXGI_FORMAT_R32_FLOAT : info.Format;
    texture->width = desc.width;
    texture->height = desc.height;
    texture->mips = desc.mips;
    texture->samples = desc.samples;
    texture->rest = (desc.usage & TextureSampled) ? D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                    : depth                       ? D3D12_RESOURCE_STATE_DEPTH_WRITE
                                                  : D3D12_RESOURCE_STATE_RENDER_TARGET;
    texture->state = texture->rest;
    if (FAILED(allocator->CreateResource(&heap, &info, texture->state, nullptr, &texture->allocation, IID_NULL, nullptr)))
      return nullptr;
    texture->resource = texture->allocation->GetResource();
    std::lock_guard lock{mutex};
    if (desc.usage & TextureColorTarget) {
      texture->rtv = rtv_heap.allocate(1);
      if (texture->rtv == UINT_MAX)
        return nullptr;
      device->CreateRenderTargetView(texture->resource.Get(), nullptr, rtv_heap.cpu(texture->rtv));
    }
    if (desc.usage & TextureDepthTarget) {
      texture->dsv = dsv_heap.allocate(1);
      if (texture->dsv == UINT_MAX)
        return nullptr;
      D3D12_DEPTH_STENCIL_VIEW_DESC view{};
      view.Format = DXGI_FORMAT_D32_FLOAT;
      view.ViewDimension = desc.samples > 1 ? D3D12_DSV_DIMENSION_TEXTURE2DMS : D3D12_DSV_DIMENSION_TEXTURE2D;
      device->CreateDepthStencilView(texture->resource.Get(), &view, dsv_heap.cpu(texture->dsv));
    }
    return texture;
  }

  std::unique_ptr<Sampler> create_sampler(const SamplerDesc& desc) override {
    auto sampler = std::make_unique<D3D12Sampler>();
    auto& native = sampler->desc;
    native.Filter = to_filter(desc);
    native.AddressU = to_address(desc.u);
    native.AddressV = to_address(desc.v);
    native.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    native.MipLODBias = desc.lod_bias;
    native.MaxAnisotropy = desc.anisotropy;
    native.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    native.MinLOD = desc.min_lod;
    native.MaxLOD = desc.max_lod;
    return sampler;
  }

  std::unique_ptr<BindGroupLayout> create_bind_group_layout(const BindGroupLayoutDesc& desc) override {
    auto layout = std::make_unique<D3D12BindGroupLayout>();
    layout->entries.assign(desc.entries, desc.entries + desc.count);
    std::sort(layout->entries.begin(), layout->entries.end(),
              [](const BindingLayout& a, const BindingLayout& b) { return a.binding < b.binding; });
    for (const auto& entry : layout->entries) {
      if (entry.kind == BindingKind::Uniform)
        layout->uniforms.push_back(entry.binding);
      else if (entry.kind == BindingKind::Sampler)
        layout->samplers.push_back(entry.binding);
      else
        layout->srvs.push_back(entry.binding);
    }
    return layout;
  }

  std::unique_ptr<BindGroup> create_bind_group(const BindGroupDesc& desc) override {
    const auto& layout = *static_cast<D3D12BindGroupLayout*>(desc.layout);
    const auto find = [&](uint32_t binding) -> const BindingValue* {
      for (uint32_t i = 0; i < desc.count; ++i) {
        if (desc.entries[i].binding == binding)
          return &desc.entries[i];
      }
      return nullptr;
    };
    auto group = std::make_unique<D3D12BindGroup>();
    group->owner = this;
    for (const auto& entry : layout.entries) {
      if (entry.kind != BindingKind::Uniform)
        continue;
      const BindingValue* value = find(entry.binding);
      group->uniforms.push_back(
          {static_cast<D3D12Buffer*>(value->buffer)->address + value->offset, entry.dynamic_offset});
    }
    std::lock_guard lock{mutex};
    if (!layout.srvs.empty()) {
      group->srv_count = static_cast<UINT>(layout.srvs.size());
      group->srv_table = srv_heap.allocate(group->srv_count);
      if (group->srv_table == UINT_MAX)
        return nullptr;
      group->srv_gpu = srv_heap.gpu(group->srv_table);
      for (UINT i = 0; i < group->srv_count; ++i) {
        const BindingValue* value = find(layout.srvs[i]);
        const auto handle = srv_heap.cpu(group->srv_table + i);
        if (value->view) {
          const auto& view = *static_cast<D3D12TextureView*>(value->view);
          device->CreateShaderResourceView(view.resource.Get(), &view.desc, handle);
        } else {
          D3D12_SHADER_RESOURCE_VIEW_DESC view{};
          view.Format = DXGI_FORMAT_R32_TYPELESS;
          view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
          view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
          view.Buffer.FirstElement = value->offset / 4;
          const auto& buffer = *static_cast<D3D12Buffer*>(value->buffer);
          view.Buffer.NumElements = static_cast<UINT>((value->size ? value->size : buffer.size - value->offset) / 4);
          view.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
          device->CreateShaderResourceView(buffer.resource, &view, handle);
        }
      }
    }
    if (!layout.samplers.empty()) {
      std::vector<std::array<uint32_t, 16>> key;
      for (uint32_t binding : layout.samplers) {
        std::array<uint32_t, 16> words{};
        std::memcpy(words.data(), &static_cast<D3D12Sampler*>(find(binding)->sampler)->desc, sizeof(D3D12_SAMPLER_DESC));
        key.push_back(words);
      }
      auto found = sampler_tables.find(key);
      if (found == sampler_tables.end()) {
        const UINT table = sampler_heap.allocate(static_cast<UINT>(key.size()));
        if (table == UINT_MAX)
          return nullptr;
        for (UINT i = 0; i < key.size(); ++i) {
          device->CreateSampler(&static_cast<D3D12Sampler*>(find(layout.samplers[i])->sampler)->desc,
                                sampler_heap.cpu(table + i));
        }
        found = sampler_tables.emplace(std::move(key), table).first;
      }
      group->has_sampler_table = true;
      group->sampler_gpu = sampler_heap.gpu(found->second);
    }
    return group;
  }

  std::unique_ptr<PipelineLayout> create_pipeline_layout(const PipelineLayoutDesc& desc) override {
    auto layout = std::make_unique<D3D12PipelineLayout>();
    std::vector<D3D12_ROOT_PARAMETER> parameters;
    std::deque<std::vector<D3D12_DESCRIPTOR_RANGE>> ranges;
    layout->groups.resize(desc.count);
    for (uint32_t g = 0; g < desc.count; ++g) {
      const auto& group = *static_cast<D3D12BindGroupLayout*>(desc.groups[g]);
      auto& slots = layout->groups[g];
      slots.first_uniform = static_cast<UINT>(parameters.size());
      for (uint32_t binding : group.uniforms) {
        D3D12_ROOT_PARAMETER parameter{};
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameter.Descriptor.ShaderRegister = binding;
        parameter.Descriptor.RegisterSpace = g;
        parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        parameters.push_back(parameter);
      }
      const auto add_table = [&](const std::vector<uint32_t>& bindings, D3D12_DESCRIPTOR_RANGE_TYPE type) {
        auto& table = ranges.emplace_back();
        UINT offset = 0;
        for (uint32_t binding : bindings) {
          D3D12_DESCRIPTOR_RANGE range{};
          range.RangeType = type;
          range.NumDescriptors = 1;
          range.BaseShaderRegister = binding;
          range.RegisterSpace = g;
          range.OffsetInDescriptorsFromTableStart = offset++;
          table.push_back(range);
        }
        D3D12_ROOT_PARAMETER parameter{};
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameter.DescriptorTable.NumDescriptorRanges = static_cast<UINT>(table.size());
        parameter.DescriptorTable.pDescriptorRanges = table.data();
        parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        parameters.push_back(parameter);
        return static_cast<UINT>(parameters.size() - 1);
      };
      if (!group.srvs.empty())
        slots.srv_table = add_table(group.srvs, D3D12_DESCRIPTOR_RANGE_TYPE_SRV);
      if (!group.samplers.empty())
        slots.sampler_table = add_table(group.samplers, D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER);
    }
    if (desc.draw_constant) {
      D3D12_ROOT_PARAMETER parameter{};
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
      parameter.Constants.ShaderRegister = 0;
      parameter.Constants.RegisterSpace = desc.count;
      parameter.Constants.Num32BitValues = 1;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
      layout->draw_constant = static_cast<UINT>(parameters.size());
      parameters.push_back(parameter);
    }
    D3D12_ROOT_SIGNATURE_DESC root{};
    root.NumParameters = static_cast<UINT>(parameters.size());
    root.pParameters = parameters.data();
    ComPtr<ID3DBlob> blob, errors;
    if (FAILED(D3D12SerializeRootSignature(&root, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors))) {
      if (errors)
        std::fprintf(stderr, "root signature: %s\n", static_cast<const char*>(errors->GetBufferPointer()));
      return nullptr;
    }
    if (FAILED(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                           IID_PPV_ARGS(&layout->root))))
      return nullptr;
    return layout;
  }

  std::unique_ptr<RenderPipeline> create_render_pipeline(const RenderPipelineDesc& desc) override {
    auto pipeline = std::make_unique<D3D12RenderPipeline>();
    auto* layout = static_cast<D3D12PipelineLayout*>(desc.layout);
    pipeline->root = layout->root;
    pipeline->layout = layout;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC info{};
    info.pRootSignature = layout->root.Get();
    info.VS = {desc.vertex->data(), desc.vertex->size()};
    if (desc.fragment)
      info.PS = {desc.fragment->data(), desc.fragment->size()};
    auto& blend = info.BlendState.RenderTarget[0];
    blend.RenderTargetWriteMask = desc.color_write;
    blend.BlendEnable = desc.blend;
    blend.SrcBlend = to_blend(desc.blend_color.src, false);
    blend.DestBlend = to_blend(desc.blend_color.dst, false);
    blend.BlendOp = desc.blend_color.op == BlendOp::Add ? D3D12_BLEND_OP_ADD : D3D12_BLEND_OP_REV_SUBTRACT;
    blend.SrcBlendAlpha = to_blend(desc.blend_alpha.src, true);
    blend.DestBlendAlpha = to_blend(desc.blend_alpha.dst, true);
    blend.BlendOpAlpha = desc.blend_alpha.op == BlendOp::Add ? D3D12_BLEND_OP_ADD : D3D12_BLEND_OP_REV_SUBTRACT;
    blend.LogicOp = D3D12_LOGIC_OP_NOOP;
    info.BlendState.AlphaToCoverageEnable = desc.alpha_to_coverage;
    info.SampleMask = UINT_MAX;
    info.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    info.RasterizerState.CullMode = desc.cull == CullMode::None    ? D3D12_CULL_MODE_NONE
                                    : desc.cull == CullMode::Front ? D3D12_CULL_MODE_FRONT
                                                                   : D3D12_CULL_MODE_BACK;
    // Counter-clockwise is front, as Vulkan's flipped viewport sees it.
    info.RasterizerState.FrontCounterClockwise = TRUE;
    info.RasterizerState.DepthClipEnable = TRUE;
    if (desc.depth_format != Format::Undefined) {
      info.DepthStencilState.DepthEnable = TRUE;
      info.DepthStencilState.DepthWriteMask = desc.depth_write ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
      info.DepthStencilState.DepthFunc = to_compare(desc.depth_compare);
      info.DSVFormat = to_dxgi(desc.depth_format);
    }
    info.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    if (desc.color_format != Format::Undefined) {
      info.NumRenderTargets = 1;
      info.RTVFormats[0] = to_dxgi(desc.color_format);
    }
    info.SampleDesc.Count = desc.samples;
    if (FAILED(device->CreateGraphicsPipelineState(&info, IID_PPV_ARGS(&pipeline->state))))
      return nullptr;
    return pipeline;
  }

  std::unique_ptr<Swapchain> create_swapchain(const SwapchainDesc& desc) override;

  std::unique_ptr<CommandList> create_command_list() override {
    auto list = std::make_unique<D3D12CommandList>();
    list->owner = this;
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&list->allocator))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, list->allocator.Get(), nullptr,
                                         IID_PPV_ARGS(&list->list))))
      return nullptr;
    list->list->Close();
    return list;
  }

  uint64_t submit(CommandList& list) override {
    ID3D12CommandList* lists[] = {static_cast<D3D12CommandList&>(list).list.Get()};
    queue->ExecuteCommandLists(1, lists);
    const UINT64 ticket = ++fence_value;
    queue->Signal(fence.Get(), ticket);
    retired.submitted(&list, ticket);
    collect();
    drain_messages();
    return ticket;
  }

  bool is_done(uint64_t ticket) override { return fence->GetCompletedValue() >= ticket; }

  void wait(uint64_t ticket) override {
    if (fence->GetCompletedValue() < ticket) {
      fence->SetEventOnCompletion(ticket, fence_event);
      WaitForSingleObject(fence_event, INFINITE);
    }
  }

  void wait_idle() override {
    if (!queue || !fence)
      return;
    const UINT64 value = ++fence_value;
    queue->Signal(fence.Get(), value);
    if (fence->GetCompletedValue() < value) {
      fence->SetEventOnCompletion(value, fence_event);
      WaitForSingleObject(fence_event, INFINITE);
    }
    collect();
    drain_messages();
  }
};

D3D12Buffer::~D3D12Buffer() {
  owner->retire([allocation = std::move(allocation)] {});
}

D3D12Texture::~D3D12Texture() {
  owner->retire([allocation = std::move(allocation), resource = std::move(resource), device = owner, rtv = rtv,
                 dsv = dsv] {
    if (rtv != UINT_MAX)
      device->rtv_heap.release(rtv, 1);
    if (dsv != UINT_MAX)
      device->dsv_heap.release(dsv, 1);
  });
}

UINT swizzle_component(Swizzle swizzle, UINT identity) {
  switch (swizzle) {
  case Swizzle::Identity: return identity;
  case Swizzle::R: return D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0;
  case Swizzle::G: return D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1;
  case Swizzle::B: return D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2;
  case Swizzle::A: return D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3;
  case Swizzle::Zero: return D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0;
  case Swizzle::One: return D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1;
  }
  return identity;
}

std::unique_ptr<TextureView> D3D12Texture::create_view(const TextureViewDesc& desc) {
  auto view = std::make_unique<D3D12TextureView>();
  view->resource = resource;
  auto& native = view->desc;
  native.Format = copy_format;
  native.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(
      swizzle_component(desc.r, 0), swizzle_component(desc.g, 1), swizzle_component(desc.b, 2),
      swizzle_component(desc.a, 3));
  if (samples > 1) {
    native.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
  } else {
    native.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    native.Texture2D.MostDetailedMip = desc.base_mip;
    native.Texture2D.MipLevels = desc.mips ? desc.mips : mips - desc.base_mip;
  }
  return view;
}

D3D12BindGroup::~D3D12BindGroup() {
  if (srv_table != UINT_MAX) {
    owner->retire([device = owner, table = srv_table, count = srv_count] { device->srv_heap.release(table, count); });
  }
}

D3D12CommandList::~D3D12CommandList() { owner->retired.closed(this); }

void D3D12CommandList::begin() {
  owner->retired.opened(this);
  allocator->Reset();
  list->Reset(allocator.Get(), nullptr);
  ID3D12DescriptorHeap* heaps[] = {owner->srv_heap.heap.Get(), owner->sampler_heap.heap.Get()};
  list->SetDescriptorHeaps(2, heaps);
  list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  bound_root = nullptr;
  layout = nullptr;
}

void D3D12CommandList::begin_render_pass(const RenderPassDesc& desc) {
  pass_color = static_cast<D3D12Texture*>(desc.color.target);
  pass_resolve = static_cast<D3D12Texture*>(desc.color.resolve);
  pass_depth = static_cast<D3D12Texture*>(desc.depth.target);
  D3D12_CPU_DESCRIPTOR_HANDLE rtv{}, dsv{};
  uint32_t width = 0, height = 0;
  if (pass_color) {
    transition(*pass_color, D3D12_RESOURCE_STATE_RENDER_TARGET);
    rtv = owner->rtv_heap.cpu(pass_color->rtv);
    width = pass_color->width;
    height = pass_color->height;
  }
  if (pass_depth) {
    transition(*pass_depth, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    dsv = owner->dsv_heap.cpu(pass_depth->dsv);
    width = pass_depth->width;
    height = pass_depth->height;
  }
  list->OMSetRenderTargets(pass_color ? 1 : 0, pass_color ? &rtv : nullptr, FALSE, pass_depth ? &dsv : nullptr);
  if (pass_color && desc.color.clear) {
    const FLOAT color[4] = {desc.color.clear_color.r, desc.color.clear_color.g, desc.color.clear_color.b,
                            desc.color.clear_color.a};
    list->ClearRenderTargetView(rtv, color, 0, nullptr);
  }
  if (pass_depth && desc.depth.clear)
    list->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, desc.depth.clear_depth, 0, 0, nullptr);
  set_viewport(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f);
  set_scissor(0, 0, width, height);
}

void D3D12CommandList::end_render_pass() {
  if (pass_color && pass_resolve) {
    transition(*pass_color, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
    transition(*pass_resolve, D3D12_RESOURCE_STATE_RESOLVE_DEST);
    list->ResolveSubresource(pass_resolve->resource.Get(), 0, pass_color->resource.Get(), 0, pass_color->copy_format);
    transition(*pass_resolve, pass_resolve->rest);
  }
  if (pass_color)
    transition(*pass_color, pass_color->rest);
  if (pass_depth)
    transition(*pass_depth, pass_depth->rest);
  pass_color = pass_resolve = pass_depth = nullptr;
}

class D3D12Swapchain final : public Swapchain {
public:
  D3D12Device* owner = nullptr;
  ComPtr<IDXGISwapChain3> swapchain;
  HANDLE waitable = nullptr;
  std::vector<std::unique_ptr<D3D12Texture>> images;
  UINT flags = 0;
  UINT present_interval = 1, present_flags = 0;
  uint32_t width = 0, height = 0;
  bool waited = false;

  ~D3D12Swapchain() override {
    release_images();
    if (waitable)
      CloseHandle(waitable);
  }

  void release_images() {
    images.clear();
    // The swapchain's buffers have to be gone, not just retired, to resize.
    owner->wait_idle();
  }

  bool build_images() {
    DXGI_SWAP_CHAIN_DESC1 info{};
    if (FAILED(swapchain->GetDesc1(&info)))
      return false;
    for (UINT i = 0; i < info.BufferCount; ++i) {
      auto image = std::make_unique<D3D12Texture>();
      image->owner = owner;
      if (FAILED(swapchain->GetBuffer(i, IID_PPV_ARGS(&image->resource))))
        return false;
      image->state = image->rest = D3D12_RESOURCE_STATE_PRESENT;
      image->copy_format = info.Format;
      image->width = info.Width;
      image->height = info.Height;
      std::lock_guard lock{owner->mutex};
      image->rtv = owner->rtv_heap.allocate(1);
      if (image->rtv == UINT_MAX)
        return false;
      owner->device->CreateRenderTargetView(image->resource.Get(), nullptr, owner->rtv_heap.cpu(image->rtv));
      images.push_back(std::move(image));
    }
    return true;
  }

  Texture* acquire() override {
    WaitForSingleObjectEx(waitable, 1000, TRUE);
    return images[swapchain->GetCurrentBackBufferIndex()].get();
  }

  void present() override { swapchain->Present(present_interval, present_flags); }

  void resize(uint32_t new_width, uint32_t new_height) override {
    release_images();
    swapchain->ResizeBuffers(0, new_width, new_height, DXGI_FORMAT_UNKNOWN, flags);
    build_images();
  }
};

std::unique_ptr<Swapchain> D3D12Device::create_swapchain(const SwapchainDesc& desc) {
  auto result = std::make_unique<D3D12Swapchain>();
  result->owner = this;
  BOOL tearing = FALSE;
  if (FAILED(factory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing))))
    tearing = FALSE;
  result->flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
  if (tearing)
    result->flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
  if (desc.mode != PresentMode::Fifo)
    result->present_interval = 0;
  if (desc.mode == PresentMode::Immediate && tearing)
    result->present_flags = DXGI_PRESENT_ALLOW_TEARING;
  DXGI_SWAP_CHAIN_DESC1 info{};
  info.Width = desc.width;
  info.Height = desc.height;
  info.Format = to_dxgi(desc.format);
  info.SampleDesc.Count = 1;
  info.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  info.BufferCount = 3;
  info.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
  info.Flags = result->flags;
  const HWND window = static_cast<HWND>(desc.window.window);
  ComPtr<IDXGISwapChain1> created;
  if (FAILED(factory->CreateSwapChainForHwnd(queue.Get(), window, &info, nullptr, nullptr, &created)) ||
      FAILED(created.As(&result->swapchain)))
    return nullptr;
  // The window's owner decides about fullscreen.
  factory->MakeWindowAssociation(window, DXGI_MWA_NO_ALT_ENTER);
  result->swapchain->SetMaximumFrameLatency(2);
  result->waitable = result->swapchain->GetFrameLatencyWaitableObject();
  if (!result->build_images())
    return nullptr;
  return result;
}

} // namespace

std::unique_ptr<Device> create_d3d12_device(const DeviceOptions& options, std::string* error) {
  if (options.validation) {
    ComPtr<ID3D12Debug> debug;
    const HRESULT result = D3D12GetDebugInterface(IID_PPV_ARGS(&debug));
    if (SUCCEEDED(result)) {
      debug->EnableDebugLayer();
      validation_active();
    } else {
      std::fprintf(stderr, "D3D12: no debug layer (0x%08lx)\n", static_cast<unsigned long>(result));
    }
  }
  auto result = std::make_unique<D3D12Device>();
  if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&result->factory)))) {
    *error = "no DXGI 1.6 factory";
    return nullptr;
  }
  auto& factory = result->factory;
  ComPtr<IDXGIAdapter1> adapter;
  for (UINT i = 0;
       factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) !=
       DXGI_ERROR_NOT_FOUND;
       ++i) {
    DXGI_ADAPTER_DESC1 desc{};
    adapter->GetDesc1(&desc);
    if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
        SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&result->device)))) {
      const int size = WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, nullptr, 0, nullptr, nullptr);
      result->name.resize(size > 0 ? size - 1 : 0);
      if (size > 1)
        WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, result->name.data(), size, nullptr, nullptr);
      break;
    }
  }
  if (!result->device) {
    *error = "no adapter with D3D12 feature level 12_0";
    return nullptr;
  }
  if (options.validation && SUCCEEDED(result->device.As(&result->info_queue))) {
    // Clear colors vary with what the game asks for, so no value can be promised at
    // a target's creation; the layer's advice about it is not an error.
    D3D12_MESSAGE_ID ignored[] = {D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,
                                  D3D12_MESSAGE_ID_CLEARDEPTHSTENCILVIEW_MISMATCHINGCLEARVALUE};
    D3D12_INFO_QUEUE_FILTER filter{};
    filter.DenyList.NumIDs = 2;
    filter.DenyList.pIDList = ignored;
    result->info_queue->AddStorageFilterEntries(&filter);
  }
  D3D12MA::ALLOCATOR_DESC allocator{};
  allocator.pDevice = result->device.Get();
  allocator.pAdapter = adapter.Get();
  if (FAILED(D3D12MA::CreateAllocator(&allocator, &result->allocator))) {
    *error = "could not create the memory allocator";
    return nullptr;
  }
  D3D12_COMMAND_QUEUE_DESC queue{};
  queue.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  auto* device = result->device.Get();
  if (FAILED(device->CreateCommandQueue(&queue, IID_PPV_ARGS(&result->queue))) ||
      FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&result->fence))) ||
      !result->rtv_heap.init(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 4096, false) ||
      !result->dsv_heap.init(device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1024, false) ||
      !result->srv_heap.init(device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1000000, true) ||
      !result->sampler_heap.init(device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 2048, true)) {
    *error = "could not create the queue, fence or descriptor heaps";
    return nullptr;
  }
  result->fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);

  // What both a color and a depth target support, among the counts the API has.
  for (UINT samples : {2u, 4u, 8u}) {
    bool supported = true;
    for (DXGI_FORMAT format : {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_D32_FLOAT}) {
      D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS levels{};
      levels.Format = format;
      levels.SampleCount = samples;
      if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &levels, sizeof(levels))) ||
          levels.NumQualityLevels == 0)
        supported = false;
    }
    if (supported)
      result->device_features.max_samples = samples;
  }
  result->device_features.uniform_offset_alignment = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
  return result;
}

} // namespace aurora::rhi
