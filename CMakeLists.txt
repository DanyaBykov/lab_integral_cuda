CMAKE_MINIMUM_REQUIRED(VERSION 3.15)
PROJECT(integrate_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 52)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")


set(SOURCES
    integral/main.cu
    integral/functions.cu
    integral/cfg_reader.cpp
)

add_executable(integrate_cuda ${SOURCES})

target_link_libraries(integrate_cuda cuda)
