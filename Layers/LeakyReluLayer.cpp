//
// Created by Antonio on 06/04/2020.
//

#include "LeakyReluLayer.h"
#include "LeakyReluLayerGpu.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdio>
#include <cstring>

LeakyReluLayer::LeakyReluLayer(const void* buffer, std::size_t size )
{
    assert( size == sizeof( float ) + sizeof( std::size_t ) );
    memcpy( &mAlpha, buffer, sizeof( float ) );
    buffer += sizeof( float );
    memcpy( &mInputSize, buffer, sizeof( std::size_t ) );
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
    memcpy( buffer, &mAlpha, sizeof( float ) );
    buffer += sizeof( float );
    memcpy( buffer, &mInputSize, sizeof( std::size_t ) );
}

int LeakyReluLayer::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream )
{
    float* outputsFloat = reinterpret_cast< float* >( outputs[ 0 ] );
    const float* inputsFloat = reinterpret_cast< const float* >( inputs[ 0 ] );
    std::size_t const totalDataSize = batchSize * mInputSize;
    LeakyReluForward( totalDataSize, inputsFloat, outputsFloat, mAlpha );

    return 0;
}