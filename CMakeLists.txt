cmake_minimum_required(VERSION 3.21)
project(cuda_learning LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(add_thread add_thread.cu)
add_executable(add add.cu)
add_executable(test test.cu)
add_executable(dot dot.cu)
add_executable(heat heat.cu)
add_executable(matrix matrix.cu)
add_executable(mattest MatTester.cu MatTester.cuh
        matTest.cu)
target_link_libraries(mattest PRIVATE
CUDA::cudart
CUDA::cublas
)
