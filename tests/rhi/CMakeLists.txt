cmake_minimum_required(VERSION 3.25)
project(aurora_rhi_tests C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/AuroraRhi.cmake")

add_executable(test_rhi test_rhi.cpp)
target_link_libraries(test_rhi PRIVATE aurora_rhi)
if (MINGW)
    target_link_options(test_rhi PRIVATE -static)
endif ()
enable_testing()
add_test(NAME rhi COMMAND test_rhi)
