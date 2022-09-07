cmake_minimum_required(VERSION 3.8)
project(cuda_tutorial)

## Compile as C++11
add_compile_options(-std=c++11)

## Find CUDA
find_package(CUDA 11 REQUIRED)

cuda_add_executable(
  test1 src/test1.cu
)

target_link_libraries(test1
    ${CUDA_LIBRARIES}
    )