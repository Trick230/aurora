#include "rhi.hpp"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <climits>
#include <cstdio>
#include <string>

namespace aurora::rhi {
namespace {

using Microsoft::WRL::ComPtr;

DXGI_FORMAT to_dxgi(Format format) {
  switch (format) {
  case Format::RGBA8Unorm:
    return DXGI_FORMAT_R8G8B8A8_UNORM;
  }
  return DXGI_FORMAT_UNKNOWN;
}

std::string narrow(const wchar_t* text) {
  const int size = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
  std::string result(size > 0 ? size - 1 : 0, '\0');
  if (size > 1)
    WideCharToMultiByte(CP_UTF8, 0, text, -1, result.data(), size, nullptr, nullptr);
  return result;
}

class D3D12Buffer final : public Buffer {
public:
  ComPtr<ID3D12Resource> resource;
  void* pointer = nullptr;
  void* mapped() override { return pointer; }
};

class D3D12Texture final : public Texture {
public:
  ComPtr<ID3D12Resource> resource;
  D3D12_CPU_DESCRIPTOR_HANDLE rtv{};
  D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
  DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
  uint32_t width = 0, height = 0;
};

class D3D12RenderPipeline final : public RenderPipeline {
public:
  ComPtr<ID3D12RootSignature> root;
  ComPtr<ID3D12PipelineState> state;
};

class D3D12CommandList final : public CommandList {
public:
  ComPtr<ID3D12CommandAllocator> allocator;
  ComPtr<ID3D12GraphicsCommandList> list;

  void begin() override {
    allocator->Reset();
    list->Reset(allocator.Get(), nullptr);
  }
  void end() override { list->Close(); }

  void transition(D3D12Texture& texture, D3D12_RESOURCE_STATES state) {
    if (texture.state == state)
      return;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = texture.resource.Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = texture.state;
    barrier.Transition.StateAfter = state;
    list->ResourceBarrier(1, &barrier);
    texture.state = state;
  }

  void begin_render_pass(Texture& target, const ClearColor& clear) override {
    auto& texture = static_cast<D3D12Texture&>(target);
    transition(texture, D3D12_RESOURCE_STATE_RENDER_TARGET);
    list->OMSetRenderTargets(1, &texture.rtv, FALSE, nullptr);
    const FLOAT color[4] = {clear.r, clear.g, clear.b, clear.a};
    list->ClearRenderTargetView(texture.rtv, color, 0, nullptr);
    const D3D12_VIEWPORT viewport{0.0f, 0.0f, static_cast<FLOAT>(texture.width), static_cast<FLOAT>(texture.height),
                                  0.0f, 1.0f};
    const D3D12_RECT scissor{0, 0, static_cast<LONG>(texture.width), static_cast<LONG>(texture.height)};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
  }
  void end_render_pass() override {}

  void set_pipeline(RenderPipeline& pipeline) override {
    auto& native = static_cast<D3D12RenderPipeline&>(pipeline);
    list->SetGraphicsRootSignature(native.root.Get());
    list->SetPipelineState(native.state.Get());
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  }
  void draw(uint32_t vertices) override { list->DrawInstanced(vertices, 1, 0, 0); }

  void copy_texture_to_buffer(Texture& source, Buffer& destination, uint32_t bytes_per_row) override {
    auto& texture = static_cast<D3D12Texture&>(source);
    auto& buffer = static_cast<D3D12Buffer&>(destination);
    transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION from{};
    from.pResource = texture.resource.Get();
    from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION to{};
    to.pResource = buffer.resource.Get();
    to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    to.PlacedFootprint.Footprint.Format = texture.format;
    to.PlacedFootprint.Footprint.Width = texture.width;
    to.PlacedFootprint.Footprint.Height = texture.height;
    to.PlacedFootprint.Footprint.Depth = 1;
    to.PlacedFootprint.Footprint.RowPitch = bytes_per_row;
    list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
  }
};

class D3D12Device final : public Device {
public:
  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<ID3D12Fence> fence;
  ComPtr<ID3D12DescriptorHeap> rtv_heap;
  HANDLE fence_event = nullptr;
  UINT64 fence_value = 0;
  UINT rtv_size = 0, rtv_used = 0;
  std::string name;
  static constexpr UINT RtvCount = 256;

  ~D3D12Device() override {
    wait_idle();
    if (fence_event)
      CloseHandle(fence_event);
  }

  Backend backend() const override { return Backend::D3D12; }
  const std::string& adapter_name() const override { return name; }

  std::unique_ptr<Buffer> create_buffer(const BufferDesc& desc) override {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = desc.memory == MemoryKind::Upload     ? D3D12_HEAP_TYPE_UPLOAD
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
    const D3D12_RESOURCE_STATES state = desc.memory == MemoryKind::Upload     ? D3D12_RESOURCE_STATE_GENERIC_READ
                                        : desc.memory == MemoryKind::Readback ? D3D12_RESOURCE_STATE_COPY_DEST
                                                                              : D3D12_RESOURCE_STATE_COMMON;
    auto buffer = std::make_unique<D3D12Buffer>();
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &info, state, nullptr,
                                               IID_PPV_ARGS(&buffer->resource))))
      return nullptr;
    if (desc.memory != MemoryKind::Device && FAILED(buffer->resource->Map(0, nullptr, &buffer->pointer)))
      return nullptr;
    return buffer;
  }

  std::unique_ptr<Texture> create_texture(const TextureDesc& desc) override {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC info{};
    info.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    info.Width = desc.width;
    info.Height = desc.height;
    info.DepthOrArraySize = 1;
    info.MipLevels = 1;
    info.Format = to_dxgi(desc.format);
    info.SampleDesc.Count = 1;
    info.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    auto texture = std::make_unique<D3D12Texture>();
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &info, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&texture->resource))))
      return nullptr;
    if (rtv_used == RtvCount)
      return nullptr;
    texture->rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    texture->rtv.ptr += SIZE_T{rtv_used++} * rtv_size;
    device->CreateRenderTargetView(texture->resource.Get(), nullptr, texture->rtv);
    texture->format = info.Format;
    texture->width = desc.width;
    texture->height = desc.height;
    return texture;
  }

  std::unique_ptr<CommandList> create_command_list() override {
    auto list = std::make_unique<D3D12CommandList>();
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&list->allocator))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, list->allocator.Get(), nullptr,
                                         IID_PPV_ARGS(&list->list))))
      return nullptr;
    list->list->Close();
    return list;
  }

  std::unique_ptr<RenderPipeline> create_render_pipeline(const RenderPipelineDesc& desc) override {
    auto pipeline = std::make_unique<D3D12RenderPipeline>();
    D3D12_ROOT_SIGNATURE_DESC root{};
    ComPtr<ID3DBlob> blob;
    if (FAILED(D3D12SerializeRootSignature(&root, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr)) ||
        FAILED(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                           IID_PPV_ARGS(&pipeline->root))))
      return nullptr;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC info{};
    info.pRootSignature = pipeline->root.Get();
    info.VS = {desc.vertex->data(), desc.vertex->size()};
    info.PS = {desc.fragment->data(), desc.fragment->size()};
    info.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    info.SampleMask = UINT_MAX;
    info.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    info.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    info.RasterizerState.DepthClipEnable = TRUE;
    info.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    info.NumRenderTargets = 1;
    info.RTVFormats[0] = to_dxgi(desc.color_format);
    info.SampleDesc.Count = 1;
    if (FAILED(device->CreateGraphicsPipelineState(&info, IID_PPV_ARGS(&pipeline->state))))
      return nullptr;
    return pipeline;
  }

  void submit(CommandList& list) override {
    ID3D12CommandList* lists[] = {static_cast<D3D12CommandList&>(list).list.Get()};
    queue->ExecuteCommandLists(1, lists);
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
  }
};

} // namespace

std::unique_ptr<Device> create_d3d12_device(std::string* error) {
  ComPtr<IDXGIFactory6> factory;
  if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
    *error = "no DXGI 1.6 factory";
    return nullptr;
  }
  auto result = std::make_unique<D3D12Device>();
  ComPtr<IDXGIAdapter1> adapter;
  for (UINT i = 0;
       factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) !=
       DXGI_ERROR_NOT_FOUND;
       ++i) {
    DXGI_ADAPTER_DESC1 desc{};
    adapter->GetDesc1(&desc);
    if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
        SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&result->device)))) {
      result->name = narrow(desc.Description);
      break;
    }
  }
  if (!result->device) {
    *error = "no adapter with D3D12 feature level 12_0";
    return nullptr;
  }
  D3D12_COMMAND_QUEUE_DESC queue{};
  queue.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  D3D12_DESCRIPTOR_HEAP_DESC heap{};
  heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
  heap.NumDescriptors = D3D12Device::RtvCount;
  if (FAILED(result->device->CreateCommandQueue(&queue, IID_PPV_ARGS(&result->queue))) ||
      FAILED(result->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&result->fence))) ||
      FAILED(result->device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&result->rtv_heap)))) {
    *error = "could not create the queue, fence or descriptor heap";
    return nullptr;
  }
  result->rtv_size = result->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
  result->fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  return result;
}

} // namespace aurora::rhi
