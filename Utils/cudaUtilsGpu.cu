#include "cudaUtilsGpu.h"

#include <cuda_device_runtime_api.h>

__global__ void subtract( const int n, const float* firstInput, const float* secondInput, float* output )
{
    CUDA_KERNEL_LOOP(index, n)
    {
        output[ index ] = firstInput[ index ] - secondInput[ index ];
    }
}

__global__ void clip( const int n, const float* input, float* output, float minValue, float maxValue )
{
    CUDA_KERNEL_LOOP(index, n)
    {
        float value = input[ index ];
        value = value < minValue ? minValue : value;
        value = value > maxValue ? maxValue : value;
        output[ index ] = value;
    }
}

__global__ void roundToInt( const int n, const float* input, float* output )
{
    CUDA_KERNEL_LOOP(index, n)
    {
        int value = __float2int_rd( input[ index ] );
        output[ index ] = __int2float_rd( value );
    }
}

__global__ void addValue( const int n, const float* input, float* output, float value )
{
    CUDA_KERNEL_LOOP(index, n)
    {
        output[ index ] = input[ index ] + value;
    }
}

cudaError_t subtractTensors( const int count, const float* firstInput, const float* secondInput, float* output )
{
    subtract<<< cudaBlockNum( count ), CUDA_THREADS_NUM >>>( count, firstInput, secondInput, output );
}

cudaError_t clipTensor( const int count, const float* input, float* output, float minValue, float maxValue )
{
    clip<<< cudaBlockNum( count ), CUDA_THREADS_NUM >>>( count, input, output, minValue, maxValue );
}

cudaError_t roundToIntTensor( const int count, const float* input, float* output )
{
    roundToInt<<< cudaBlockNum( count ), CUDA_THREADS_NUM >>>( count, input, output );
}

cudaError_t addValueToTensor( const int count, const float* input, float* output, float value )
{
    addValue<<< cudaBlockNum( count ), CUDA_THREADS_NUM >>>( count, input, output, value );
}

