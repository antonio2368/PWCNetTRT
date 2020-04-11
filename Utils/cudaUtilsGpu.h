//
// Created by Antonio on 11/04/2020.
//

#ifndef PWCNET_CUDAUTILSGPU_H
#define PWCNET_CUDAUTILSGPU_H

#include <driver_types.h>

#define CUDA_THREADS_NUM 512

inline int cudaBlockNum( const int N )
{
    return ( N + CUDA_THREADS_NUM - 1 ) / CUDA_THREADS_NUM;
}

#define CUDA_KERNEL_LOOP( i, n ) \
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; \
          i < ( n ); \
          i += blockDim.x * gridDim.x )

cudaError_t subtractTensors( const int count, const float* firstInput, const float* secondInput, float* output );

cudaError_t clipTensor( const int count, const float* input, float* output, float minValue, float maxValue );

cudaError_t roundToIntTensor( const int count, const float* input, float* output );

cudaError_t addValueToTensor( const int count, const float* input, float* output, float value );

#endif //PWCNET_CUDAUTILSGPU_H
