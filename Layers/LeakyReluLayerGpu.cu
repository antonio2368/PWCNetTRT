#include "LeakyReluLayerGpu.h"

__global__ void LeakyRelu( const int n, const float* in, float* out, float alpha )
{
    CUDA_KERNEL_LOOP(index, n)
    {
        out[ index ] = in[ index ] < 0.0f ? in[ index ] * alpha : in[ index ];
    }
}

cudaError_t LeakyReluForward( const int count, const float* input, float* output, float alpha )
{
    LeakyRelu<<< cudaBlockNum( count ), CUDA_THREADS_NUM >>>( count, input, output, alpha );
}
