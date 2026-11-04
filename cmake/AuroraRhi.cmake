include_guard(GLOBAL)
include(FetchContent)

# The native renderer's layer over D3D12 and Vulkan. Only headers and single-file
# libraries come from these projects, so none of their own targets are configured.
set(_aurora_rhi_vulkan_tag "vulkan-sdk-1.4.363.0")
FetchContent_Declare(aurora_vulkan_headers
        GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
        GIT_TAG ${_aurora_rhi_vulkan_tag} GIT_SHALLOW TRUE SOURCE_SUBDIR _none)
FetchContent_Declare(aurora_volk
        GIT_REPOSITORY https://github.com/zeux/volk.git
        GIT_TAG ${_aurora_rhi_vulkan_tag} GIT_SHALLOW TRUE SOURCE_SUBDIR _none)
FetchContent_Declare(aurora_vma
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
        GIT_TAG v3.5.0 GIT_SHALLOW TRUE SOURCE_SUBDIR _none)
FetchContent_MakeAvailable(aurora_vulkan_headers aurora_volk aurora_vma)

set(_aurora_rhi_dir "${CMAKE_CURRENT_LIST_DIR}/../lib/rhi")
add_library(aurora_rhi STATIC
        ${_aurora_rhi_dir}/rhi.cpp
        ${_aurora_rhi_dir}/vulkan.cpp
        ${aurora_volk_SOURCE_DIR}/volk.c)
set_target_properties(aurora_rhi PROPERTIES FOLDER "aurora")
target_include_directories(aurora_rhi PUBLIC ${_aurora_rhi_dir})
target_include_directories(aurora_rhi SYSTEM PRIVATE
        ${aurora_vulkan_headers_SOURCE_DIR}/include ${aurora_volk_SOURCE_DIR} ${aurora_vma_SOURCE_DIR}/include)
if (WIN32)
    target_sources(aurora_rhi PRIVATE ${_aurora_rhi_dir}/d3d12.cpp)
    target_compile_definitions(aurora_rhi PRIVATE VK_USE_PLATFORM_WIN32_KHR NOMINMAX WIN32_LEAN_AND_MEAN)
    target_link_libraries(aurora_rhi PRIVATE d3d12 dxgi dxguid)
else ()
    target_link_libraries(aurora_rhi PRIVATE ${CMAKE_DL_LIBS})
endif ()
