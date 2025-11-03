cmake_minimum_required(VERSION 3.21)
project(cuda_learning LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(add_thread add_thread.cu)
add_executable(add add.cu)
add_executable(test test.cu)
add_executable(dot dot.cu)
