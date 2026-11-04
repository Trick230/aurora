#include "rhi.hpp"

#include <volk.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <string>
#include <vector>

namespace aurora::rhi {
namespace {

VkFormat to_vk(Format format) {
  switch (format) {
  case Format::RGBA8Unorm:
    return VK_FORMAT_R8G8B8A8_UNORM;
  }
  return VK_FORMAT_UNDEFINED;
}

class VulkanBuffer final : public Buffer {
public:
  VmaAllocator allocator = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  void* pointer = nullptr;
  ~VulkanBuffer() override { vmaDestroyBuffer(allocator, buffer, allocation); }
  void* mapped() override { return pointer; }
};

class VulkanTexture final : public Texture {
public:
  VkDevice device = VK_NULL_HANDLE;
  VmaAllocator allocator = VK_NULL_HANDLE;
  VkImage image = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  uint32_t width = 0, height = 0;
  ~VulkanTexture() override {
    vkDestroyImageView(device, view, nullptr);
    vmaDestroyImage(allocator, image, allocation);
  }
};

class VulkanRenderPipeline final : public RenderPipeline {
public:
  VkDevice device = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  ~VulkanRenderPipeline() override {
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, layout, nullptr);
  }
};

class VulkanCommandList final : public CommandList {
public:
  VkDevice device = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandBuffer list = VK_NULL_HANDLE;

  ~VulkanCommandList() override { vkDestroyCommandPool(device, pool, nullptr); }

  void begin() override {
    vkResetCommandPool(device, pool, 0);
    VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(list, &info);
  }
  void end() override { vkEndCommandBuffer(list); }

  void transition(VulkanTexture& texture, VkImageLayout layout) {
    if (texture.layout == layout)
      return;
    VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    barrier.oldLayout = texture.layout;
    barrier.newLayout = layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = texture.image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(list, &dependency);
    texture.layout = layout;
  }

  void begin_render_pass(Texture& target, const ClearColor& clear) override {
    auto& texture = static_cast<VulkanTexture&>(target);
    transition(texture, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageView = texture.view;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = {{clear.r, clear.g, clear.b, clear.a}};
    VkRenderingInfo info{VK_STRUCTURE_TYPE_RENDERING_INFO};
    info.renderArea.extent = {texture.width, texture.height};
    info.layerCount = 1;
    info.colorAttachmentCount = 1;
    info.pColorAttachments = &color;
    vkCmdBeginRendering(list, &info);
    // Flipped, for the orientation of clip space D3D12 has.
    const VkViewport viewport{0.0f,
                              static_cast<float>(texture.height),
                              static_cast<float>(texture.width),
                              -static_cast<float>(texture.height),
                              0.0f,
                              1.0f};
    const VkRect2D scissor{{0, 0}, {texture.width, texture.height}};
    vkCmdSetViewport(list, 0, 1, &viewport);
    vkCmdSetScissor(list, 0, 1, &scissor);
  }
  void end_render_pass() override { vkCmdEndRendering(list); }

  void set_pipeline(RenderPipeline& pipeline) override {
    vkCmdBindPipeline(list, VK_PIPELINE_BIND_POINT_GRAPHICS, static_cast<VulkanRenderPipeline&>(pipeline).pipeline);
  }
  void draw(uint32_t vertices) override { vkCmdDraw(list, vertices, 1, 0, 0); }

  void copy_texture_to_buffer(Texture& source, Buffer& destination, uint32_t bytes_per_row) override {
    auto& texture = static_cast<VulkanTexture&>(source);
    transition(texture, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy region{};
    region.bufferRowLength = bytes_per_row / 4;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {texture.width, texture.height, 1};
    vkCmdCopyImageToBuffer(list, texture.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           static_cast<VulkanBuffer&>(destination).buffer, 1, &region);
    // The host reads the buffer once the queue has gone idle.
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(list, &dependency);
  }
};

class VulkanDevice final : public Device {
public:
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  VmaAllocator allocator = VK_NULL_HANDLE;
  uint32_t queue_family = 0;
  std::string name;

  ~VulkanDevice() override {
    if (device) {
      vkDeviceWaitIdle(device);
      vmaDestroyAllocator(allocator);
      vkDestroyDevice(device, nullptr);
    }
    if (instance)
      vkDestroyInstance(instance, nullptr);
  }

  Backend backend() const override { return Backend::Vulkan; }
  const std::string& adapter_name() const override { return name; }

  std::unique_ptr<Buffer> create_buffer(const BufferDesc& desc) override {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = desc.size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
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
    buffer->allocator = allocator;
    buffer->pointer = allocated.pMappedData;
    return buffer;
  }

  std::unique_ptr<Texture> create_texture(const TextureDesc& desc) override {
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = to_vk(desc.format);
    info.extent = {desc.width, desc.height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo memory{};
    memory.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    auto texture = std::make_unique<VulkanTexture>();
    texture->device = device;
    texture->allocator = allocator;
    if (vmaCreateImage(allocator, &info, &memory, &texture->image, &texture->allocation, nullptr) != VK_SUCCESS)
      return nullptr;
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = texture->image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = info.format;
    view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(device, &view, nullptr, &texture->view) != VK_SUCCESS)
      return nullptr;
    texture->width = desc.width;
    texture->height = desc.height;
    return texture;
  }

  std::unique_ptr<CommandList> create_command_list() override {
    auto list = std::make_unique<VulkanCommandList>();
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
    pipeline->device = device;
    VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    if (vkCreatePipelineLayout(device, &layout, nullptr, &pipeline->layout) != VK_SUCCESS)
      return nullptr;
    VkShaderModule vertex = create_module(*desc.vertex);
    VkShaderModule fragment = create_module(*desc.fragment);
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
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState attachment{};
    attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                                VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &attachment;
    const VkDynamicState dynamics[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamics;
    const VkFormat format = to_vk(desc.color_format);
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &format;
    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.pNext = &rendering;
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &vertex_input;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = pipeline->layout;
    const VkResult result =
        vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline->pipeline);
    vkDestroyShaderModule(device, vertex, nullptr);
    vkDestroyShaderModule(device, fragment, nullptr);
    if (result != VK_SUCCESS)
      return nullptr;
    return pipeline;
  }

  void submit(CommandList& list) override {
    VkCommandBuffer buffer = static_cast<VulkanCommandList&>(list).list;
    VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    info.commandBufferCount = 1;
    info.pCommandBuffers = &buffer;
    vkQueueSubmit(queue, 1, &info, VK_NULL_HANDLE);
  }

  void wait_idle() override { vkQueueWaitIdle(queue); }
};

} // namespace

std::unique_ptr<Device> create_vulkan_device(std::string* error) {
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
  if (vkCreateInstance(&instance, nullptr, &result->instance) != VK_SUCCESS) {
    *error = "could not create a Vulkan instance";
    return nullptr;
  }
  volkLoadInstanceOnly(result->instance);

  uint32_t count = 0;
  vkEnumeratePhysicalDevices(result->instance, &count, nullptr);
  std::vector<VkPhysicalDevice> physicals(count);
  vkEnumeratePhysicalDevices(result->instance, &count, physicals.data());
  // A discrete GPU first, as the D3D12 backend's high-performance preference.
  int best = 0;
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
    }
  }
  if (!result->physical) {
    *error = "no GPU with Vulkan 1.3";
    return nullptr;
  }

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

  const float priority = 1.0f;
  VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue.queueFamilyIndex = result->queue_family;
  queue.queueCount = 1;
  queue.pQueuePriorities = &priority;
  VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  features13.dynamicRendering = VK_TRUE;
  features13.synchronization2 = VK_TRUE;
  VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  device.pNext = &features13;
  device.queueCreateInfoCount = 1;
  device.pQueueCreateInfos = &queue;
  if (vkCreateDevice(result->physical, &device, nullptr, &result->device) != VK_SUCCESS) {
    *error = "could not create a Vulkan device";
    return nullptr;
  }
  volkLoadDevice(result->device);
  vkGetDeviceQueue(result->device, result->queue_family, 0, &result->queue);

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
