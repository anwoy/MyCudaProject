cmake_minimum_required(VERSION 3.18)
project(MyCudaProject LANGUAGES CXX CUDA)

# 1. Standard and Optimization Settings
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Force "Release" mode (applies -O3) if the user didn't specify one
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 2. Find Multithreading support (pthreads/std::thread)
find_package(Threads REQUIRED)

# 3. CUDA-specific Architecture
set(CMAKE_CUDA_ARCHITECTURES 50)

# 4. Find all sources (.cu and .cpp)
file(GLOB SOURCE_FILES src/*.cu src/*.cpp)

add_executable(a.out ${SOURCE_FILES})

# 5. Compiler Flags (Optimization & CUDA)
target_compile_options(a.out PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true> 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall>    # Optimization and Warnings for C++
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -O3> # Pass -O3 through nvcc to the host
)

# 6. Linking
# Link the threading library and set CUDA properties
target_link_libraries(a.out PRIVATE Threads::Threads curand)

set_target_properties(a.out PROPERTIES 
    CUDA_SEPARATE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
)

target_include_directories(a.out PUBLIC src)
