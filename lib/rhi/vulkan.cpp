#include "rhi_internal.hpp"

#include <volk.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace aurora::rhi {
namespace {

VkFormat to_vk(Format format) {
  switch (format) {
  case Format::Undefined: return VK_FORMAT_UNDEFINED;
  case Format::R8Unorm: return VK_FORMAT_R8_UNORM;
  case Format::RG8Unorm: return VK_FORMAT_R8G8_UNORM;
  case Format::RGBA8Unorm: return VK_FORMAT_R8G8B8A8_UNORM;
  case Format::RGBA8Srgb: return VK_FORMAT_R8G8B8A8_SRGB;
  case Format::BGRA8Unorm: return VK_FORMAT_B8G8R8A8_UNORM;
  case Format::BGRA8Srgb: return VK_FORMAT_B8G8R8A8_SRGB;
  case Format::R16Sint: return VK_FORMAT_R16_SINT;
  case Format::R32Float: return VK_FORMAT_R32_SFLOAT;
  case Format::Depth32Float: return VK_FORMAT_D32_SFLOAT;
  case Format::BC1: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
  case Format::BC2: return VK_FORMAT_BC2_UNORM_BLOCK;
  case Format::BC3: return VK_FORMAT_BC3_UNORM_BLOCK;
  case Format::BC4: return VK_FORMAT_BC4_UNORM_BLOCK;
  case Format::BC5: return VK_FORMAT_BC5_UNORM_BLOCK;
  case Format::BC6H: return VK_FORMAT_BC6H_UFLOAT_BLOCK;
  case Format::BC7: return VK_FORMAT_BC7_UNORM_BLOCK;
  }
  return VK_FORMAT_UNDEFINED;
}

// What a row of pixels takes in a copy, in the units VkBufferImageCopy counts.
struct BlockInfo {
  uint32_t bytes, size;
};
BlockInfo block_info(Format format) {
  switch (format) {
  case Format::R8Unorm: return {1, 1};
  case Format::RG8Unorm:
  case Format::R16Sint: return {2, 1};
  case Format::BC1:
  case Format::BC4: return {8, 4};
  case Format::BC2:
  case Format::BC3:
  case Format::BC5:
  case Format::BC6H:
  case Format::BC7: return {16, 4};
  default: return {4, 1};
  }
}

VkCompareOp to_compare(Compare compare) {
  switch (compare) {
  case Compare::Never: return VK_COMPARE_OP_NEVER;
  case Compare::Less: return VK_COMPARE_OP_LESS;
  case Compare::Equal: return VK_COMPARE_OP_EQUAL;
  case Compare::LessEqual: return VK_COMPARE_OP_LESS_OR_EQUAL;
  case Compare::Greater: return VK_COMPARE_OP_GREATER;
  case Compare::NotEqual: return VK_COMPARE_OP_NOT_EQUAL;
  case Compare::GreaterEqual: return VK_COMPARE_OP_GREATER_OR_EQUAL;
  case Compare::Always: return VK_COMPARE_OP_ALWAYS;
  }
  return VK_COMPARE_OP_ALWAYS;
}

VkBlendFactor to_blend(BlendFactor factor) {
  switch (factor) {
  case BlendFactor::Zero: return VK_BLEND_FACTOR_ZERO;
  case BlendFactor::One: return VK_BLEND_FACTOR_ONE;
  case BlendFactor::Src: return VK_BLEND_FACTOR_SRC_COLOR;
  case BlendFactor::OneMinusSrc: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
  case BlendFactor::SrcAlpha: return VK_BLEND_FACTOR_SRC_ALPHA;
  case BlendFactor::OneMinusSrcAlpha: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  case BlendFactor::Dst: return VK_BLEND_FACTOR_DST_COLOR;
  case BlendFactor::OneMinusDst: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
  case BlendFactor::DstAlpha: return VK_BLEND_FACTOR_DST_ALPHA;
  case BlendFactor::OneMinusDstAlpha: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
  case BlendFactor::Constant: return VK_BLEND_FACTOR_CONSTANT_COLOR;
  case BlendFactor::Src1Alpha: return VK_BLEND_FACTOR_SRC1_ALPHA;
  case BlendFactor::OneMinusSrc1Alpha: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
  }
  return VK_BLEND_FACTOR_ONE;
}

VkSamplerAddressMode to_address(AddressMode mode) {
  switch (mode) {
  case AddressMode::Repeat: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  case AddressMode::MirrorRepeat: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  case AddressMode::ClampToEdge: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  }
  return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}

VkComponentSwizzle to_swizzle(Swizzle swizzle) {
  switch (swizzle) {
  case Swizzle::Identity: return VK_COMPONENT_SWIZZLE_IDENTITY;
  case Swizzle::R: return VK_COMPONENT_SWIZZLE_R;
  case Swizzle::G: return VK_COMPONENT_SWIZZLE_G;
  case Swizzle::B: return VK_COMPONENT_SWIZZLE_B;
  case Swizzle::A: return VK_COMPONENT_SWIZZLE_A;
  case Swizzle::Zero: return VK_COMPONENT_SWIZZLE_ZERO;
  case Swizzle::One: return VK_COMPONENT_SWIZZLE_ONE;
  }
  return VK_COMPONENT_SWIZZLE_IDENTITY;
}

// Registers to bindings, as compile_shader tells DXC to number them.
uint32_t binding_number(BindingKind kind, uint32_t binding) {
  switch (kind) {
  case BindingKind::Uniform: return binding;
  case BindingKind::StorageRead:
  case BindingKind::Texture: return binding + 1000;
  case BindingKind::Sampler: return binding + 2000;
  }
  return binding;
}

class VulkanDevice;

class VulkanBuffer final : public Buffer {
public:
  VulkanDevice* owner = nullptr;
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  uint64_t size = 0;
  void* pointer = nullptr;
  ~VulkanBuffer() override;
  void* mapped() override { return pointer; }
};

class VulkanTexture final : public Texture {
public:
  VulkanDevice* owner = nullptr;
  VkImage image = VK_NULL_HANDLE;
  VkImageView target_view = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageLayout rest = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
  VkFormat vk_format = VK_FORMAT_UNDEFINED;
  Format format = Format::Undefined;
  uint32_t width = 0, height = 0, mips = 1, samples = 1;
  ~VulkanTexture() override;
  std::unique_ptr<TextureView> create_view(const TextureViewDesc& desc) override;
};

class VulkanTextureView final : public TextureView {
public:
  VulkanDevice* owner = nullptr;
  VkImageView view = VK_NULL_HANDLE;
  ~VulkanTextureView() override;
};

class VulkanSampler final : public Sampler {
public:
  VulkanDevice* owner = nullptr;
  VkSampler sampler = VK_NULL_HANDLE;
  ~VulkanSampler() override;
};

class VulkanBindGroupLayout final : public BindGroupLayout {
public:
  VulkanDevice* owner = nullptr;
  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  std::vector<BindingLayout> entries;
  ~VulkanBindGroupLayout() override;
};

class VulkanBindGroup final : public BindGroup {
public:
  VulkanDevice* owner = nullptr;
  VkDescriptorPool pool = VK_NULL_HANDLE;
  VkDescriptorSet set = VK_NULL_HANDLE;
  ~VulkanBindGroup() override;
};

class VulkanPipelineLayout final : public PipelineLayout {
public:
  VulkanDevice* owner = nullptr;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  ~VulkanPipelineLayout() override;
};

class VulkanRenderPipeline final : public RenderPipeline {
public:
  VulkanDevice* owner = nullptr;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  ~VulkanRenderPipeline() override;
};

VkImageMemoryBarrier2 image_barrier(const VulkanTexture& texture, VkImageLayout from, VkImageLayout to) {
  VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
  barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
  barrier.oldLayout = from;
  barrier.newLayout = to;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = texture.image;
  barrier.subresourceRange = {texture.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, 1};
  return barrier;
}

class VulkanCommandList final : public CommandList {
public:
  VulkanDevice* owner = nullptr;
  VkDevice device = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandBuffer list = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VulkanTexture* pass_color = nullptr;
  VulkanTexture* pass_resolve = nullptr;
  VulkanTexture* pass_depth = nullptr;

  ~VulkanCommandList() override { vkDestroyCommandPool(device, pool, nullptr); }

  void begin() override {
    vkResetCommandPool(device, pool, 0);
    VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(list, &info);
    layout = VK_NULL_HANDLE;
  }
  void end() override { vkEndCommandBuffer(list); }

  void transition(VulkanTexture& texture, VkImageLayout to) {
    if (texture.layout == to)
      return;
    const VkImageMemoryBarrier2 barrier = image_barrier(texture, texture.layout, to);
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(list, &dependency);
    texture.layout = to;
  }

  void memory_barrier(VkPipelineStageFlags2 stage, VkAccessFlags2 access, VkPipelineStageFlags2 next,
                      VkAccessFlags2 next_access) {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = stage;
    barrier.srcAccessMask = access;
    barrier.dstStageMask = next;
    barrier.dstAccessMask = next_access;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(list, &dependency);
  }

  void begin_render_pass(const RenderPassDesc& desc) override {
    pass_color = static_cast<VulkanTexture*>(desc.color.target);
    pass_resolve = static_cast<VulkanTexture*>(desc.color.resolve);
    pass_depth = static_cast<VulkanTexture*>(desc.depth.target);
    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    VkRenderingInfo info{VK_STRUCTURE_TYPE_RENDERING_INFO};
    info.layerCount = 1;
    uint32_t width = 0, height = 0;
    if (pass_color) {
      transition(*pass_color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
      color.imageView = pass_color->target_view;
      color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      color.loadOp = desc.color.clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
      color.storeOp = desc.color.store ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
      color.clearValue.color = {
          {desc.color.clear_color.r, desc.color.clear_color.g, desc.color.clear_color.b, desc.color.clear_color.a}};
      if (pass_resolve) {
        transition(*pass_resolve, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        color.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
        color.resolveImageView = pass_resolve->target_view;
        color.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      }
      info.colorAttachmentCount = 1;
      info.pColorAttachments = &color;
      width = pass_color->width;
      height = pass_color->height;
    }
    if (pass_depth) {
      transition(*pass_depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
      depth.imageView = pass_depth->target_view;
      depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
      depth.loadOp = desc.depth.clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
      depth.storeOp = desc.depth.store ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
      depth.clearValue.depthStencil = {desc.depth.clear_depth, 0};
      info.pDepthAttachment = &depth;
      width = pass_depth->width;
      height = pass_depth->height;
    }
    info.renderArea.extent = {width, height};
    vkCmdBeginRendering(list, &info);
    set_viewport(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f);
    set_scissor(0, 0, width, height);
  }

  void end_render_pass() override {
    vkCmdEndRendering(list);
    if (pass_resolve)
      transition(*pass_resolve, pass_resolve->rest);
    if (pass_color)
      transition(*pass_color, pass_color->rest);
    if (pass_depth)
      transition(*pass_depth, pass_depth->rest);
    pass_color = pass_resolve = pass_depth = nullptr;
  }

  void set_pipeline(RenderPipeline& pipeline) override {
    auto& native = static_cast<VulkanRenderPipeline&>(pipeline);
    vkCmdBindPipeline(list, VK_PIPELINE_BIND_POINT_GRAPHICS, native.pipeline);
    layout = native.layout;
  }
  void set_bind_group(uint32_t index, BindGroup& group, const uint32_t* dynamic_offsets,
                      uint32_t dynamic_count) override {
    vkCmdBindDescriptorSets(list, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, index, 1,
                            &static_cast<VulkanBindGroup&>(group).set, dynamic_count, dynamic_offsets);
  }
  void set_index_buffer_u16(Buffer& buffer, uint64_t offset) override {
    vkCmdBindIndexBuffer(list, static_cast<VulkanBuffer&>(buffer).buffer, offset, VK_INDEX_TYPE_UINT16);
  }
  void set_viewport(float x, float y, float width, float height, float min_depth, float max_depth) override {
    // Flipped, for the orientation of clip space D3D12 has.
    const VkViewport viewport{x, y + height, width, -height, min_depth, max_depth};
    vkCmdSetViewport(list, 0, 1, &viewport);
  }
  void set_scissor(uint32_t x, uint32_t y, uint32_t width, uint32_t height) override {
    const VkRect2D rect{{static_cast<int32_t>(x), static_cast<int32_t>(y)}, {width, height}};
    vkCmdSetScissor(list, 0, 1, &rect);
  }
  void set_blend_constant(float r, float g, float b, float a) override {
    const float constants[4] = {r, g, b, a};
    vkCmdSetBlendConstants(list, constants);
  }
  void set_draw_constant(uint32_t value) override {
    vkCmdPushConstants(list, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(value), &value);
  }
  void draw(uint32_t vertices, uint32_t first_vertex, uint32_t instances) override {
    vkCmdDraw(list, vertices, instances, first_vertex, 0);
  }
  void draw_indexed(uint32_t indices, uint32_t first_index, int32_t base_vertex, uint32_t instances) override {
    vkCmdDrawIndexed(list, indices, instances, first_index, base_vertex, 0);
  }

  void copy_buffer(Buffer& source, uint64_t source_offset, Buffer& destination, uint64_t destination_offset,
                   uint64_t size) override {
    // The destination may still be read by what an earlier submit draws.
    memory_barrier(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    const VkBufferCopy region{source_offset, destination_offset, size};
    vkCmdCopyBuffer(list, static_cast<VulkanBuffer&>(source).buffer, static_cast<VulkanBuffer&>(destination).buffer, 1,
                    &region);
    memory_barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                   VK_ACCESS_2_MEMORY_READ_BIT);
  }

  static VkBufferImageCopy image_copy(const VulkanTexture& texture, uint64_t offset, uint32_t bytes_per_row,
                                      const TextureRegion& region) {
    const BlockInfo block = block_info(texture.format);
    VkBufferImageCopy copy{};
    copy.bufferOffset = offset;
    copy.bufferRowLength = bytes_per_row / block.bytes * block.size;
    copy.imageSubresource = {texture.aspect, region.mip, 0, 1};
    copy.imageOffset = {static_cast<int32_t>(region.x), static_cast<int32_t>(region.y), 0};
    copy.imageExtent = {region.width ? region.width : MipExtent(texture.width, region.mip),
                        region.height ? region.height : MipExtent(texture.height, region.mip), 1};
    return copy;
  }

  void copy_buffer_to_texture(Buffer& source, uint64_t offset, uint32_t bytes_per_row, Texture& destination,
                              const TextureRegion& region) override {
    auto& texture = static_cast<VulkanTexture&>(destination);
    transition(texture, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    const VkBufferImageCopy copy = image_copy(texture, offset, bytes_per_row, region);
    vkCmdCopyBufferToImage(list, static_cast<VulkanBuffer&>(source).buffer, texture.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    transition(texture, texture.rest);
  }

  void copy_texture_to_buffer(Texture& source, const TextureRegion& region, Buffer& destination, uint64_t offset,
                              uint32_t bytes_per_row) override {
    auto& texture = static_cast<VulkanTexture&>(source);
    transition(texture, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    const VkBufferImageCopy copy = image_copy(texture, offset, bytes_per_row, region);
    vkCmdCopyImageToBuffer(list, texture.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           static_cast<VulkanBuffer&>(destination).buffer, 1, &copy);
    // The host reads it once the queue has gone idle.
    memory_barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                   VK_ACCESS_2_HOST_READ_BIT);
    transition(texture, texture.rest);
  }

  void copy_texture(Texture& source, const TextureRegion& region, Texture& destination, uint32_t x,
                    uint32_t y) override {
    auto& from = static_cast<VulkanTexture&>(source);
    auto& to = static_cast<VulkanTexture&>(destination);
    transition(from, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    transition(to, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageCopy copy{};
    copy.srcSubresource = {from.aspect, region.mip, 0, 1};
    copy.srcOffset = {static_cast<int32_t>(region.x), static_cast<int32_t>(region.y), 0};
    copy.dstSubresource = {to.aspect, 0, 0, 1};
    copy.dstOffset = {static_cast<int32_t>(x), static_cast<int32_t>(y), 0};
    copy.extent = {region.width ? region.width : MipExtent(from.width, region.mip),
                   region.height ? region.height : MipExtent(from.height, region.mip), 1};
    vkCmdCopyImage(list, from.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, to.image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    transition(to, to.rest);
    transition(from, from.rest);
  }
};

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
                                              const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
  std::fprintf(stderr, "Vulkan: %s\n", data->pMessage);
  g_validation_errors.fetch_add(1);
  return VK_FALSE;
}

class VulkanDevice final : public Device {
public:
  VkInstance instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  VmaAllocator allocator = VK_NULL_HANDLE;
  VkSemaphore timeline = VK_NULL_HANDLE;
  // Takes new textures to the layout they rest in, ahead of the next submit.
  VkCommandPool init_pool = VK_NULL_HANDLE;
  VkCommandBuffer init_list = VK_NULL_HANDLE;
  // The image a swapchain handed out, until a submit waits for it.
  VkSemaphore pending_acquire = VK_NULL_HANDLE;
  uint64_t* pending_acquire_done = nullptr;
  bool can_present = false;
  uint64_t timeline_value = 0;
  uint32_t queue_family = 0;
  std::string name;
  DeviceFeatures device_features;
  std::deque<std::pair<uint64_t, std::function<void()>>> retired;
  std::vector<VkDescriptorPool> pools;
  std::mutex mutex;
  float max_anisotropy = 16.0f;

  ~VulkanDevice() override {
    if (device) {
      vkDeviceWaitIdle(device);
      collect(true);
      if (init_pool)
        vkDestroyCommandPool(device, init_pool, nullptr);
      for (VkDescriptorPool pool : pools)
        vkDestroyDescriptorPool(device, pool, nullptr);
      vkDestroySemaphore(device, timeline, nullptr);
      vmaDestroyAllocator(allocator);
      vkDestroyDevice(device, nullptr);
    }
    if (messenger)
      vkDestroyDebugUtilsMessengerEXT(instance, messenger, nullptr);
    if (instance)
      vkDestroyInstance(instance, nullptr);
  }

  Backend backend() const override { return Backend::Vulkan; }
  const std::string& adapter_name() const override { return name; }
  const DeviceFeatures& features() const override { return device_features; }

  void retire(std::function<void()> release) {
    std::lock_guard lock{mutex};
    retired.emplace_back(timeline_value + 1, std::move(release));
  }
  void collect(bool everything = false) {
    uint64_t done = 0;
    vkGetSemaphoreCounterValue(device, timeline, &done);
    std::lock_guard lock{mutex};
    while (!retired.empty() && (everything || retired.front().first <= done)) {
      retired.front().second();
      retired.pop_front();
    }
  }

  std::unique_ptr<Buffer> create_buffer(const BufferDesc& desc) override {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = desc.size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    VmaAllocationCreateInfo memory{};
    memory.usage = VMA_MEMORY_USAGE_AUTO;
    if (desc.memory == MemoryKind::Upload)
      memory.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    else if (desc.memory == MemoryKind::Readback)
      memory.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    auto buffer = std::make_unique<VulkanBuffer>();
    VmaAllocationInfo allocated{};
    if (vmaCreateBuffer(allocator, &info, &memory, &buffer->buffer, &buffer->allocation, &allocated) != VK_SUCCESS)
      return nullptr;
    buffer->owner = this;
    buffer->size = desc.size;
    buffer->pointer = allocated.pMappedData;
    return buffer;
  }

  std::unique_ptr<Texture> create_texture(const TextureDesc& desc) override {
    const bool depth = desc.format == Format::Depth32Float;
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = to_vk(desc.format);
    info.extent = {desc.width, desc.height, 1};
    info.mipLevels = desc.mips;
    info.arrayLayers = 1;
    info.samples = static_cast<VkSampleCountFlagBits>(desc.samples);
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (desc.usage & TextureSampled)
      info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (desc.usage & TextureColorTarget)
      info.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (desc.usage & TextureDepthTarget)
      info.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    VmaAllocationCreateInfo memory{};
    memory.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (desc.usage & (TextureColorTarget | TextureDepthTarget))
      memory.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    auto texture = std::make_unique<VulkanTexture>();
    texture->owner = this;
    if (vmaCreateImage(allocator, &info, &memory, &texture->image, &texture->allocation, nullptr) != VK_SUCCESS)
      return nullptr;
    texture->aspect = depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    texture->vk_format = info.format;
    texture->format = desc.format;
    texture->width = desc.width;
    texture->height = desc.height;
    texture->mips = desc.mips;
    texture->samples = desc.samples;
    texture->rest = (desc.usage & TextureSampled) ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    : depth                       ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
                                                  : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    if (desc.usage & (TextureColorTarget | TextureDepthTarget)) {
      VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      view.image = texture->image;
      view.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view.format = info.format;
      view.subresourceRange = {texture->aspect, 0, 1, 0, 1};
      if (vkCreateImageView(device, &view, nullptr, &texture->target_view) != VK_SUCCESS)
        return nullptr;
    }
    // Resting where everything else expects it, as D3D12's creation state does.
    {
      std::lock_guard lock{mutex};
      if (!init_list && !begin_init_list())
        return nullptr;
      const VkImageMemoryBarrier2 barrier = image_barrier(*texture, VK_IMAGE_LAYOUT_UNDEFINED, texture->rest);
      VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      dependency.imageMemoryBarrierCount = 1;
      dependency.pImageMemoryBarriers = &barrier;
      vkCmdPipelineBarrier2(init_list, &dependency);
    }
    texture->layout = texture->rest;
    return texture;
  }

  std::unique_ptr<Sampler> create_sampler(const SamplerDesc& desc) override {
    VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    info.magFilter = desc.mag == Filter::Linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    info.minFilter = desc.min == Filter::Linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    info.mipmapMode = desc.mip == Filter::Linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info.addressModeU = to_address(desc.u);
    info.addressModeV = to_address(desc.v);
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.mipLodBias = desc.lod_bias;
    info.anisotropyEnable = desc.anisotropy > 1;
    info.maxAnisotropy = std::min(static_cast<float>(desc.anisotropy), max_anisotropy);
    info.minLod = desc.min_lod;
    info.maxLod = desc.max_lod;
    auto sampler = std::make_unique<VulkanSampler>();
    sampler->owner = this;
    if (vkCreateSampler(device, &info, nullptr, &sampler->sampler) != VK_SUCCESS)
      return nullptr;
    return sampler;
  }

  std::unique_ptr<BindGroupLayout> create_bind_group_layout(const BindGroupLayoutDesc& desc) override {
    auto layout = std::make_unique<VulkanBindGroupLayout>();
    layout->owner = this;
    layout->entries.assign(desc.entries, desc.entries + desc.count);
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (const auto& entry : layout->entries) {
      VkDescriptorSetLayoutBinding binding{};
      binding.binding = binding_number(entry.kind, entry.binding);
      binding.descriptorCount = 1;
      binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
      switch (entry.kind) {
      case BindingKind::Uniform:
        binding.descriptorType =
            entry.dynamic_offset ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        break;
      case BindingKind::StorageRead: binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; break;
      case BindingKind::Texture: binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; break;
      case BindingKind::Sampler: binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; break;
      }
      bindings.push_back(binding);
    }
    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.bindingCount = static_cast<uint32_t>(bindings.size());
    info.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &info, nullptr, &layout->layout) != VK_SUCCESS)
      return nullptr;
    return layout;
  }

  VkDescriptorPool new_pool() {
    constexpr uint32_t Sets = 16384;
    const VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, Sets},  {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, Sets},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, Sets},  {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, Sets * 8},
        {VK_DESCRIPTOR_TYPE_SAMPLER, Sets * 8},
    };
    VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    info.maxSets = Sets;
    info.poolSizeCount = 5;
    info.pPoolSizes = sizes;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    vkCreateDescriptorPool(device, &info, nullptr, &pool);
    return pool;
  }

  std::unique_ptr<BindGroup> create_bind_group(const BindGroupDesc& desc) override {
    const auto& layout = *static_cast<VulkanBindGroupLayout*>(desc.layout);
    auto group = std::make_unique<VulkanBindGroup>();
    group->owner = this;
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &layout.layout;
    {
      std::lock_guard lock{mutex};
      for (auto it = pools.rbegin(); it != pools.rend() && !group->set; ++it) {
        allocate.descriptorPool = *it;
        if (vkAllocateDescriptorSets(device, &allocate, &group->set) != VK_SUCCESS)
          group->set = VK_NULL_HANDLE;
        else
          group->pool = *it;
      }
      if (!group->set) {
        pools.push_back(new_pool());
        allocate.descriptorPool = pools.back();
        if (vkAllocateDescriptorSets(device, &allocate, &group->set) != VK_SUCCESS)
          return nullptr;
        group->pool = pools.back();
      }
    }
    std::vector<VkWriteDescriptorSet> writes;
    std::deque<VkDescriptorBufferInfo> buffers;
    std::deque<VkDescriptorImageInfo> images;
    for (const auto& entry : layout.entries) {
      const BindingValue* value = nullptr;
      for (uint32_t i = 0; i < desc.count && !value; ++i) {
        if (desc.entries[i].binding == entry.binding)
          value = &desc.entries[i];
      }
      VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      write.dstSet = group->set;
      write.dstBinding = binding_number(entry.kind, entry.binding);
      write.descriptorCount = 1;
      switch (entry.kind) {
      case BindingKind::Uniform:
      case BindingKind::StorageRead: {
        auto& info = buffers.emplace_back();
        info.buffer = static_cast<VulkanBuffer*>(value->buffer)->buffer;
        info.offset = value->offset;
        info.range = value->size ? value->size : VK_WHOLE_SIZE;
        write.pBufferInfo = &info;
        write.descriptorType = entry.kind == BindingKind::StorageRead ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                               : entry.dynamic_offset                 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                                                                      : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        break;
      }
      case BindingKind::Texture: {
        auto& info = images.emplace_back();
        info.imageView = static_cast<VulkanTextureView*>(value->view)->view;
        info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        write.pImageInfo = &info;
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        break;
      }
      case BindingKind::Sampler: {
        auto& info = images.emplace_back();
        info.sampler = static_cast<VulkanSampler*>(value->sampler)->sampler;
        write.pImageInfo = &info;
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        break;
      }
      }
      writes.push_back(write);
    }
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    return group;
  }

  std::unique_ptr<PipelineLayout> create_pipeline_layout(const PipelineLayoutDesc& desc) override {
    std::vector<VkDescriptorSetLayout> sets;
    for (uint32_t i = 0; i < desc.count; ++i)
      sets.push_back(static_cast<VulkanBindGroupLayout*>(desc.groups[i])->layout);
    VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    info.setLayoutCount = desc.count;
    info.pSetLayouts = sets.data();
    const VkPushConstantRange draw_constant{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(uint32_t)};
    if (desc.draw_constant) {
      info.pushConstantRangeCount = 1;
      info.pPushConstantRanges = &draw_constant;
    }
    auto layout = std::make_unique<VulkanPipelineLayout>();
    layout->owner = this;
    if (vkCreatePipelineLayout(device, &info, nullptr, &layout->layout) != VK_SUCCESS)
      return nullptr;
    return layout;
  }

  VkShaderModule create_module(const ShaderCode& code) {
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = code.size();
    info.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule module = VK_NULL_HANDLE;
    vkCreateShaderModule(device, &info, nullptr, &module);
    return module;
  }

  std::unique_ptr<RenderPipeline> create_render_pipeline(const RenderPipelineDesc& desc) override {
    auto pipeline = std::make_unique<VulkanRenderPipeline>();
    pipeline->owner = this;
    pipeline->layout = static_cast<VulkanPipelineLayout*>(desc.layout)->layout;
    VkShaderModule vertex = create_module(*desc.vertex);
    VkShaderModule fragment = desc.fragment ? create_module(*desc.fragment) : VK_NULL_HANDLE;
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertex;
    stages[0].pName = desc.vertex_entry;
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragment;
    stages[1].pName = desc.fragment_entry;
    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = desc.cull == CullMode::None    ? VK_CULL_MODE_NONE
                      : desc.cull == CullMode::Front ? VK_CULL_MODE_FRONT_BIT
                                                     : VK_CULL_MODE_BACK_BIT;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = static_cast<VkSampleCountFlagBits>(desc.samples);
    multisample.alphaToCoverageEnable = desc.alpha_to_coverage;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = desc.depth_format != Format::Undefined;
    depth.depthWriteEnable = desc.depth_write;
    depth.depthCompareOp = to_compare(desc.depth_compare);
    VkPipelineColorBlendAttachmentState attachment{};
    attachment.colorWriteMask = desc.color_write;
    attachment.blendEnable = desc.blend;
    attachment.srcColorBlendFactor = to_blend(desc.blend_color.src);
    attachment.dstColorBlendFactor = to_blend(desc.blend_color.dst);
    attachment.colorBlendOp = desc.blend_color.op == BlendOp::Add ? VK_BLEND_OP_ADD : VK_BLEND_OP_REVERSE_SUBTRACT;
    attachment.srcAlphaBlendFactor = to_blend(desc.blend_alpha.src);
    attachment.dstAlphaBlendFactor = to_blend(desc.blend_alpha.dst);
    attachment.alphaBlendOp = desc.blend_alpha.op == BlendOp::Add ? VK_BLEND_OP_ADD : VK_BLEND_OP_REVERSE_SUBTRACT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = desc.color_format != Format::Undefined ? 1 : 0;
    blend.pAttachments = &attachment;
    const VkDynamicState dynamics[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                       VK_DYNAMIC_STATE_BLEND_CONSTANTS};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 3;
    dynamic.pDynamicStates = dynamics;
    const VkFormat color_format = to_vk(desc.color_format);
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = desc.color_format != Format::Undefined ? 1 : 0;
    rendering.pColorAttachmentFormats = &color_format;
    rendering.depthAttachmentFormat = to_vk(desc.depth_format);
    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.pNext = &rendering;
    info.stageCount = desc.fragment ? 2 : 1;
    info.pStages = stages;
    info.pVertexInputState = &vertex_input;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &depth;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = pipeline->layout;
    const VkResult result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline->pipeline);
    vkDestroyShaderModule(device, vertex, nullptr);
    if (fragment)
      vkDestroyShaderModule(device, fragment, nullptr);
    if (result != VK_SUCCESS)
      return nullptr;
    return pipeline;
  }

  std::unique_ptr<CommandList> create_command_list() override {
    auto list = std::make_unique<VulkanCommandList>();
    list->owner = this;
    list->device = device;
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.queueFamilyIndex = queue_family;
    if (vkCreateCommandPool(device, &pool, nullptr, &list->pool) != VK_SUCCESS)
      return nullptr;
    VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    info.commandPool = list->pool;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &info, &list->list) != VK_SUCCESS)
      return nullptr;
    return list;
  }

  bool begin_init_list() {
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool.queueFamilyIndex = queue_family;
    if (vkCreateCommandPool(device, &pool, nullptr, &init_pool) != VK_SUCCESS)
      return false;
    VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    info.commandPool = init_pool;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &info, &init_list) != VK_SUCCESS)
      return false;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(init_list, &begin);
    return true;
  }

  std::unique_ptr<Swapchain> create_swapchain(const SwapchainDesc& desc) override;

  uint64_t submit(CommandList& list) override {
    queue_submit(static_cast<VulkanCommandList&>(list).list, VK_NULL_HANDLE);
    return timeline_value;
  }

  bool is_done(uint64_t ticket) override {
    uint64_t done = 0;
    vkGetSemaphoreCounterValue(device, timeline, &done);
    return done >= ticket;
  }

  void wait(uint64_t ticket) override {
    VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    info.semaphoreCount = 1;
    info.pSemaphores = &timeline;
    info.pValues = &ticket;
    vkWaitSemaphores(device, &info, UINT64_MAX);
  }

  // Everything goes through here: new textures' layouts first, and a wait for the
  // image of a swapchain if one was just acquired. A signal for a present comes
  // after all that was submitted before, in the same batch or not.
  void queue_submit(VkCommandBuffer list, VkSemaphore present_signal) {
    VkCommandBuffer buffers[2];
    uint32_t buffer_count = 0;
    {
      std::lock_guard lock{mutex};
      if (init_list) {
        vkEndCommandBuffer(init_list);
        buffers[buffer_count++] = init_list;
        retired.emplace_back(timeline_value + 1, [device = device, pool = init_pool] {
          vkDestroyCommandPool(device, pool, nullptr);
        });
        init_list = VK_NULL_HANDLE;
        init_pool = VK_NULL_HANDLE;
      }
    }
    if (list)
      buffers[buffer_count++] = list;
    const uint64_t value = ++timeline_value;
    const uint64_t signal_values[2] = {value, 0};
    const VkSemaphore signals[2] = {timeline, present_signal};
    const uint64_t wait_value = 0;
    const VkPipelineStageFlags wait_stage =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkTimelineSemaphoreSubmitInfo timeline_info{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    timeline_info.signalSemaphoreValueCount = present_signal ? 2 : 1;
    timeline_info.pSignalSemaphoreValues = signal_values;
    VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    info.pNext = &timeline_info;
    info.commandBufferCount = buffer_count;
    info.pCommandBuffers = buffers;
    info.signalSemaphoreCount = present_signal ? 2 : 1;
    info.pSignalSemaphores = signals;
    if (pending_acquire) {
      info.waitSemaphoreCount = 1;
      info.pWaitSemaphores = &pending_acquire;
      info.pWaitDstStageMask = &wait_stage;
      timeline_info.waitSemaphoreValueCount = 1;
      timeline_info.pWaitSemaphoreValues = &wait_value;
      *pending_acquire_done = value;
    }
    vkQueueSubmit(queue, 1, &info, VK_NULL_HANDLE);
    pending_acquire = VK_NULL_HANDLE;
    collect();
  }

  void wait_idle() override {
    vkQueueWaitIdle(queue);
    collect();
  }
};

VulkanBuffer::~VulkanBuffer() {
  owner->retire([allocator = owner->allocator, buffer = buffer, allocation = allocation] {
    vmaDestroyBuffer(allocator, buffer, allocation);
  });
}

VulkanTexture::~VulkanTexture() {
  owner->retire([device = owner->device, allocator = owner->allocator, image = image, view = target_view,
                 allocation = allocation] {
    if (view)
      vkDestroyImageView(device, view, nullptr);
    if (allocation)
      vmaDestroyImage(allocator, image, allocation);
  });
}

std::unique_ptr<TextureView> VulkanTexture::create_view(const TextureViewDesc& desc) {
  VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  info.image = image;
  info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  info.format = vk_format;
  info.components = {to_swizzle(desc.r), to_swizzle(desc.g), to_swizzle(desc.b), to_swizzle(desc.a)};
  info.subresourceRange = {aspect, desc.base_mip, desc.mips ? desc.mips : mips - desc.base_mip, 0, 1};
  auto view = std::make_unique<VulkanTextureView>();
  view->owner = owner;
  if (vkCreateImageView(owner->device, &info, nullptr, &view->view) != VK_SUCCESS)
    return nullptr;
  return view;
}

VulkanTextureView::~VulkanTextureView() {
  owner->retire([device = owner->device, view = view] { vkDestroyImageView(device, view, nullptr); });
}
VulkanSampler::~VulkanSampler() {
  owner->retire([device = owner->device, sampler = sampler] { vkDestroySampler(device, sampler, nullptr); });
}
VulkanBindGroupLayout::~VulkanBindGroupLayout() {
  owner->retire([device = owner->device, layout = layout] { vkDestroyDescriptorSetLayout(device, layout, nullptr); });
}
VulkanBindGroup::~VulkanBindGroup() {
  owner->retire([device = owner->device, pool = pool, set = set] { vkFreeDescriptorSets(device, pool, 1, &set); });
}
VulkanPipelineLayout::~VulkanPipelineLayout() {
  owner->retire([device = owner->device, layout = layout] { vkDestroyPipelineLayout(device, layout, nullptr); });
}
VulkanRenderPipeline::~VulkanRenderPipeline() {
  owner->retire([device = owner->device, pipeline = pipeline] { vkDestroyPipeline(device, pipeline, nullptr); });
}

class VulkanSwapchain final : public Swapchain {
public:
  VulkanDevice* owner = nullptr;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
  Format format = Format::BGRA8Unorm;
  std::vector<std::unique_ptr<VulkanTexture>> images;
  // One semaphore per image each way: an acquire's can be used again once the
  // submit that waited for it is done, and a present's once its image comes back.
  std::vector<VkSemaphore> acquire_semaphores, present_semaphores;
  std::vector<uint64_t> acquire_done;
  uint32_t next_slot = 0, current = 0;

  ~VulkanSwapchain() override {
    owner->wait_idle();
    release_images();
    owner->collect();
    vkDestroySwapchainKHR(owner->device, swapchain, nullptr);
    vkDestroySurfaceKHR(owner->instance, surface, nullptr);
  }

  void release_images() {
    images.clear();
    for (VkSemaphore semaphore : acquire_semaphores)
      vkDestroySemaphore(owner->device, semaphore, nullptr);
    for (VkSemaphore semaphore : present_semaphores)
      vkDestroySemaphore(owner->device, semaphore, nullptr);
    acquire_semaphores.clear();
    present_semaphores.clear();
    acquire_done.clear();
    next_slot = 0;
  }

  bool build(uint32_t width, uint32_t height) {
    VkSurfaceCapabilitiesKHR capabilities{};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(owner->physical, surface, &capabilities) != VK_SUCCESS)
      return false;
    VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    info.surface = surface;
    info.minImageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount)
      info.minImageCount = std::min(info.minImageCount, capabilities.maxImageCount);
    info.imageFormat = to_vk(format);
    info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    info.imageExtent = {std::clamp(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                        std::clamp(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)};
    info.imageArrayLayers = 1;
    info.imageUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = capabilities.currentTransform;
    info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    info.presentMode = mode;
    info.clipped = VK_TRUE;
    info.oldSwapchain = swapchain;
    VkSwapchainKHR created = VK_NULL_HANDLE;
    if (vkCreateSwapchainKHR(owner->device, &info, nullptr, &created) != VK_SUCCESS)
      return false;
    if (swapchain)
      vkDestroySwapchainKHR(owner->device, swapchain, nullptr);
    swapchain = created;

    uint32_t count = 0;
    vkGetSwapchainImagesKHR(owner->device, swapchain, &count, nullptr);
    std::vector<VkImage> handles(count);
    vkGetSwapchainImagesKHR(owner->device, swapchain, &count, handles.data());
    for (VkImage handle : handles) {
      auto image = std::make_unique<VulkanTexture>();
      image->owner = owner;
      image->image = handle;
      image->rest = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
      image->vk_format = info.imageFormat;
      image->format = format;
      image->width = info.imageExtent.width;
      image->height = info.imageExtent.height;
      VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      view.image = handle;
      view.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view.format = info.imageFormat;
      view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      if (vkCreateImageView(owner->device, &view, nullptr, &image->target_view) != VK_SUCCESS)
        return false;
      images.push_back(std::move(image));
      const VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      VkSemaphore acquire_semaphore = VK_NULL_HANDLE, present_semaphore = VK_NULL_HANDLE;
      vkCreateSemaphore(owner->device, &semaphore, nullptr, &acquire_semaphore);
      vkCreateSemaphore(owner->device, &semaphore, nullptr, &present_semaphore);
      acquire_semaphores.push_back(acquire_semaphore);
      present_semaphores.push_back(present_semaphore);
      acquire_done.push_back(0);
    }
    return true;
  }

  Texture* acquire() override {
    const uint32_t slot = next_slot;
    VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wait.semaphoreCount = 1;
    wait.pSemaphores = &owner->timeline;
    wait.pValues = &acquire_done[slot];
    vkWaitSemaphores(owner->device, &wait, UINT64_MAX);
    const VkResult result = vkAcquireNextImageKHR(owner->device, swapchain, UINT64_MAX, acquire_semaphores[slot],
                                                  VK_NULL_HANDLE, &current);
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
      return nullptr;
    owner->pending_acquire = acquire_semaphores[slot];
    owner->pending_acquire_done = &acquire_done[slot];
    next_slot = (slot + 1) % static_cast<uint32_t>(acquire_semaphores.size());
    return images[current].get();
  }

  void present() override {
    const VkSemaphore semaphore = present_semaphores[current];
    owner->queue_submit(VK_NULL_HANDLE, semaphore);
    VkPresentInfoKHR info{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &semaphore;
    info.swapchainCount = 1;
    info.pSwapchains = &swapchain;
    info.pImageIndices = &current;
    // Out of date shows at the next acquire.
    vkQueuePresentKHR(owner->queue, &info);
  }

  void resize(uint32_t width, uint32_t height) override {
    owner->wait_idle();
    vkDeviceWaitIdle(owner->device);
    release_images();
    owner->collect(true);
    build(width, height);
  }
};

std::unique_ptr<Swapchain> VulkanDevice::create_swapchain(const SwapchainDesc& desc) {
  if (!can_present)
    return nullptr;
  auto result = std::make_unique<VulkanSwapchain>();
  result->owner = this;
  result->format = desc.format;
#ifdef _WIN32
  VkWin32SurfaceCreateInfoKHR surface{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
  surface.hinstance = static_cast<HINSTANCE>(desc.window.instance);
  surface.hwnd = static_cast<HWND>(desc.window.window);
  if (vkCreateWin32SurfaceKHR(instance, &surface, nullptr, &result->surface) != VK_SUCCESS)
    return nullptr;
#endif
  VkBool32 supported = VK_FALSE;
  vkGetPhysicalDeviceSurfaceSupportKHR(physical, queue_family, result->surface, &supported);
  if (!supported) {
    vkDestroySurfaceKHR(instance, result->surface, nullptr);
    return nullptr;
  }
  uint32_t count = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(physical, result->surface, &count, nullptr);
  std::vector<VkPresentModeKHR> modes(count);
  vkGetPhysicalDeviceSurfacePresentModesKHR(physical, result->surface, &count, modes.data());
  const auto has_mode = [&](VkPresentModeKHR mode) { return std::find(modes.begin(), modes.end(), mode) != modes.end(); };
  if (desc.mode == PresentMode::Immediate && has_mode(VK_PRESENT_MODE_IMMEDIATE_KHR))
    result->mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
  else if (desc.mode != PresentMode::Fifo && has_mode(VK_PRESENT_MODE_MAILBOX_KHR))
    result->mode = VK_PRESENT_MODE_MAILBOX_KHR;
  if (!result->build(desc.width, desc.height))
    return nullptr;
  return result;
}

} // namespace

std::unique_ptr<Device> create_vulkan_device(const DeviceOptions& options, std::string* error) {
  if (volkInitialize() != VK_SUCCESS) {
    *error = "no Vulkan loader";
    return nullptr;
  }
  if (volkGetInstanceVersion() < VK_API_VERSION_1_3) {
    *error = "the Vulkan loader is older than 1.3";
    return nullptr;
  }
  auto result = std::make_unique<VulkanDevice>();
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo instance{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instance.pApplicationInfo = &app;
  const char* layer = "VK_LAYER_KHRONOS_validation";
  bool validating = false;
  // Without a display's extensions the device still works, headless.
  std::vector<const char*> instance_extensions;
  {
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, available.data());
    const auto has = [&](const char* name) {
      for (const auto& properties : available) {
        if (std::strcmp(properties.extensionName, name) == 0)
          return true;
      }
      return false;
    };
#ifdef _WIN32
    if (has(VK_KHR_SURFACE_EXTENSION_NAME) && has(VK_KHR_WIN32_SURFACE_EXTENSION_NAME)) {
      instance_extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
      instance_extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
    }
#endif
  }
  const bool has_surface = !instance_extensions.empty();
  if (options.validation) {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const auto& properties : layers)
      validating |= std::strcmp(properties.layerName, layer) == 0;
    if (validating) {
      instance.enabledLayerCount = 1;
      instance.ppEnabledLayerNames = &layer;
      instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
  }
  instance.enabledExtensionCount = static_cast<uint32_t>(instance_extensions.size());
  instance.ppEnabledExtensionNames = instance_extensions.data();
  if (vkCreateInstance(&instance, nullptr, &result->instance) != VK_SUCCESS) {
    *error = "could not create a Vulkan instance";
    return nullptr;
  }
  volkLoadInstanceOnly(result->instance);
  if (validating) {
    validation_active();
    VkDebugUtilsMessengerCreateInfoEXT messenger{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    messenger.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    messenger.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    messenger.pfnUserCallback = debug_callback;
    vkCreateDebugUtilsMessengerEXT(result->instance, &messenger, nullptr, &result->messenger);
  }

  uint32_t count = 0;
  vkEnumeratePhysicalDevices(result->instance, &count, nullptr);
  std::vector<VkPhysicalDevice> physicals(count);
  vkEnumeratePhysicalDevices(result->instance, &count, physicals.data());
  // A discrete GPU first, as the D3D12 backend's high-performance preference.
  int best = 0;
  VkPhysicalDeviceProperties chosen{};
  for (VkPhysicalDevice physical : physicals) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical, &properties);
    if (properties.apiVersion < VK_API_VERSION_1_3 || properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
      continue;
    const int score = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 : 1;
    if (score > best) {
      best = score;
      result->physical = physical;
      result->name = properties.deviceName;
      chosen = properties;
    }
  }
  if (!result->physical) {
    *error = "no GPU with Vulkan 1.3";
    return nullptr;
  }
  result->max_anisotropy = chosen.limits.maxSamplerAnisotropy;
  const VkSampleCountFlags samples =
      chosen.limits.framebufferColorSampleCounts & chosen.limits.framebufferDepthSampleCounts;
  for (uint32_t count_try : {2u, 4u, 8u}) {
    if (samples & count_try)
      result->device_features.max_samples = count_try;
  }
  // The same on both backends, to lay uniforms out once: D3D12's is the larger.
  result->device_features.uniform_offset_alignment =
      std::max<uint32_t>(256, static_cast<uint32_t>(chosen.limits.minUniformBufferOffsetAlignment));

  vkGetPhysicalDeviceQueueFamilyProperties(result->physical, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(result->physical, &count, families.data());
  bool found = false;
  for (uint32_t i = 0; i < count && !found; ++i) {
    if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      result->queue_family = i;
      found = true;
    }
  }
  if (!found) {
    *error = "no graphics queue";
    return nullptr;
  }

  VkPhysicalDeviceFeatures supported{};
  vkGetPhysicalDeviceFeatures(result->physical, &supported);
  if (!supported.textureCompressionBC || !supported.dualSrcBlend || !supported.samplerAnisotropy) {
    *error = "the GPU lacks BC textures, dual source blending or anisotropic filtering";
    return nullptr;
  }

  const float priority = 1.0f;
  VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue.queueFamilyIndex = result->queue_family;
  queue.queueCount = 1;
  queue.pQueuePriorities = &priority;
  VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  features12.timelineSemaphore = VK_TRUE;
  VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  features13.pNext = &features12;
  features13.dynamicRendering = VK_TRUE;
  features13.synchronization2 = VK_TRUE;
  features13.shaderDemoteToHelperInvocation = VK_TRUE; // what DXC makes of discard
  VkPhysicalDeviceFeatures features{};
  features.textureCompressionBC = VK_TRUE;
  features.dualSrcBlend = VK_TRUE;
  features.samplerAnisotropy = VK_TRUE;
  VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  device.pNext = &features13;
  device.queueCreateInfoCount = 1;
  device.pQueueCreateInfos = &queue;
  device.pEnabledFeatures = &features;
  const char* swapchain_extension = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
  {
    uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(result->physical, nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> available(extension_count);
    vkEnumerateDeviceExtensionProperties(result->physical, nullptr, &extension_count, available.data());
    for (const auto& properties : available) {
      if (has_surface && std::strcmp(properties.extensionName, swapchain_extension) == 0)
        result->can_present = true;
    }
  }
  if (result->can_present) {
    device.enabledExtensionCount = 1;
    device.ppEnabledExtensionNames = &swapchain_extension;
  }
  if (vkCreateDevice(result->physical, &device, nullptr, &result->device) != VK_SUCCESS) {
    *error = "could not create a Vulkan device";
    return nullptr;
  }
  volkLoadDevice(result->device);
  vkGetDeviceQueue(result->device, result->queue_family, 0, &result->queue);

  VkSemaphoreTypeCreateInfo timeline_type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  timeline_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo timeline{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  timeline.pNext = &timeline_type;
  if (vkCreateSemaphore(result->device, &timeline, nullptr, &result->timeline) != VK_SUCCESS) {
    *error = "could not create the timeline semaphore";
    return nullptr;
  }
  result->pools.push_back(result->new_pool());

  VmaVulkanFunctions functions{};
  functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
  VmaAllocatorCreateInfo allocator{};
  allocator.physicalDevice = result->physical;
  allocator.device = result->device;
  allocator.instance = result->instance;
  allocator.pVulkanFunctions = &functions;
  allocator.vulkanApiVersion = VK_API_VERSION_1_3;
  if (vmaCreateAllocator(&allocator, &result->allocator) != VK_SUCCESS) {
    *error = "could not create the memory allocator";
    return nullptr;
  }
  return result;
}

} // namespace aurora::rhi
