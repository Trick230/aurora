cmake_minimum_required(VERSION 3.25)
project(aurora_rhi_tests C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/AuroraRhi.cmake")

add_executable(test_rhi test_rhi.cpp)
target_link_libraries(test_rhi PRIVATE aurora_rhi)
target_include_directories(test_rhi PRIVATE ../../include)
if (MINGW)
    target_link_options(test_rhi PRIVATE -static)
endif ()
add_custom_command(TARGET test_rhi POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${AURORA_RHI_RUNTIME_FILES} $<TARGET_FILE_DIR:test_rhi>)
# Compiles a folder of generated HLSL on every backend: AURORA_HLSL_DUMP's output.
add_executable(check_hlsl check_hlsl.cpp)
target_link_libraries(check_hlsl PRIVATE aurora_rhi)
if (MINGW)
    target_link_options(check_hlsl PRIVATE -static)
endif ()
enable_testing()
add_test(NAME rhi COMMAND test_rhi)
