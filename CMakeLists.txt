cmake_minimum_required(VERSION 3.5)
project(PWCNet)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_STANDARD 14)

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
        Layers/LeakyReluLayer.cpp
        Layers/LeakyReluLayer.h
        Layers/LeakyReluLayerGpu.cu
        Layers/LeakyReluLayerGpu.h
        Layers/PluginFactory.cpp
        Layers/PluginFactory.h
        Layers/LeakyReluLayerGpu.h
        Layers/CostVolumeLayer.cpp
        Layers/CostVolumeLayer.h
        Layers/ImageWarpLayer.cpp
        Layers/ImageWarpLayer.h
        Utils/common.h
        Utils/buffer.h
        Utils/logging.h
        Utils/logger.h
        Utils/logger.cpp
        Utils/half.h
        Utils/cudaUtils.cpp
        Utils/cudaUtils.h
        Utils/cudaUtilsGpu.cu
        Utils/cudaUtilsGpu.h
        Utils/TensorUtils.cpp
        Utils/TensorUtils.h
        Layers/ReshapeLayer.cpp
        Layers/ReshapeLayer.h
)

target_compile_options(PWCNet PUBLIC "-fno-rtti")

set_source_files_properties( Layers/LeakyReluLayerGpu.cu Utils/cudaUtilsGpu.cu PROPERTIES LANGUAGE CUDA )

set(CUDART_LIB "/usr/local/cuda/lib64/libcudart.so" )
set(CUDNN_LIB "/usr/lib/aarch64-linux-gnu/libcudnn.so" )


set(CUDA_DEP_LIBS
        ${CUDART_LIB}
        ${CUBLAS_LIB}
        ${CUDNN_LIB}
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