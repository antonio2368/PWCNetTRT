//
// Created by Antonio on 06/04/2020.
//

#include "LeakyReluLayer.h"
#include "LeakyReluLayerGpu.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdio>

LeakyReluLayer::LeakyReluLayer(const void* buffer, std::size_t size )
{
    assert( size == sizeof( float ) + sizeof( std::size_t ) );
    const float* alpha = reinterpret_cast< const float* >( buffer );
    mAlpha = *alpha;
    ++alpha;
    mInputSize = *reinterpret_cast< const std::size_t* >( alpha );
}

void LeakyReluLayer::configure(const nvinfer1::Dims* inputs, int nbInputs, const nvinfer1::Dims* outputs, int nbOutputs, int maxBatchSize )
{
    assert( nbInputs == 1 );
    assert( inputs->nbDims == 3 );

    mInputSize = inputs->d[ 0 ] * inputs->d[ 1 ] * inputs->d[ 2 ];
}

void LeakyReluLayer::serialize(void* buffer )
{
    // memcpy
    float* f = reinterpret_cast< float* >( buffer );
    *f = mAlpha;
    ++f;
    std::size_t* sizePtr = reinterpret_cast< std::size_t* >( f );
    *sizePtr = mInputSize;
}

int LeakyReluLayer::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream )
{
    float* outputsFloat = reinterpret_cast< float* >( outputs[ 0 ] );
    const float* inputsFloat = reinterpret_cast< const float* >( inputs[ 0 ] );
    std::size_t const totalDataSize = batchSize * mInputSize;
    LeakyReluForward( totalDataSize, inputsFloat, outputsFloat, mAlpha );

    return 0;
}