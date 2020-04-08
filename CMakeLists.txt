cmake_minimum_required(VERSION 3.5)
project(PWCNet)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# compile a CUDA file into an object file
IF(NOT CMAKE_CUDA_COMPILE_OBJECT)
    SET(CMAKE_CUDA_COMPILE_OBJECT
            "${CMAKE_CUDA_COMPILER} -o <OBJECT> -c <SOURCE>")
ENDIF(NOT CMAKE_CUDA_COMPILE_OBJECT)

add_executable(
        PWCNet
        main.cpp
        PWCNet.cpp
        PWCNet.h
        Utils/common.h
        Utils/buffer.h
        Utils/logging.h
        Utils/logger.h
        Utils/logger.cpp
        Utils/half.h
        Utils/cudaUtils.cpp
        Utils/cudaUtils.h
        Layers/LeakyReluLayer.cpp
        Layers/LeakyReluLayer.h
        Layers/LeakyReluLayerGpu.cu
        Layers/LeakyReluLayerGpu.h
        Layers/PluginFactory.cpp
        Layers/PluginFactory.h
        Layers/LeakyReluLayerGpu.h)

target_compile_options(PWCNet PUBLIC "-fno-rtti")

set_source_files_properties( Layers/LeakyReluLayerGpu.cu PROPERTIES LANGUAGE CUDA )

set(CUDA_RUNTIME "/usr/local/cuda/lib64/libcudart.so" )

set(CUDA_DEP_LIBS
        ${CUDART_LIB}
        ${CUBLAS_LIB}
        ${CUDNN_LIB}
        ${CUDA_RUNTIME}
        nvinfer
        ${RT_LIB}
        ${CMAKE_DL_LIBS}
        ${CMAKE_THREAD_LIBS_INIT}
)

# Necessary to link nvinfer_plugin library.
target_link_libraries(PWCNet
        PUBLIC ${CUDA_DEP_LIBS}
        -Wl,--unresolved-symbols=ignore-in-shared-libs
)

target_include_directories(PWCNet PRIVATE "/usr/local/cuda/include")

set_target_properties(PWCNet PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")

add_custom_command(
        TARGET PWCNet POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/pwcnet_trt
        ${CMAKE_CURRENT_BINARY_DIR}/pwcnet_trt
)