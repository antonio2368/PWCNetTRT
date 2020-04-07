//
// Created by Antonio on 05/04/2020.
//

#include "SplitLayer.h"

#include <cassert>
#include <cuda_runtime_api.h>


SplitLayer::SplitLayer( const void* buffer, std::size_t size )
{
    assert( size == 3 * sizeof( int ) );
    const int* d = reinterpret_cast< const int * >( buffer );
    mDimOutputs = nvinfer1::DimsCHW( d[ 0 ], d[ 1 ], d[ 2 ] );
    mSize = d[ 0 ] * d[ 1 ] * d[ 2 ];
}

nvinfer1::Dims SplitLayer::getOutputDimensions( int index, const nvinfer1::Dims* inputs, int nbInputDims )
{
    assert( nbInputDims == 1 );
    assert( inputs[ 0 ].nbDims == 4 );
    assert( inputs[ 0 ].d[ 0 ] == 2 );
    return nvinfer1::DimsCHW( inputs[ 0 ].d[ 1 ], inputs[ 0 ].d[ 2 ], inputs[ 0 ].d[ 3 ] );
}

void SplitLayer::configure( const nvinfer1::Dims* inputs, int nbInputs, const nvinfer1::Dims* outputs, int nbOutputs, int maxBatchSize )
{
    assert( nbInputs == 1 );
    assert( inputs[ 0 ].nbDims == 4 );
    assert( inputs[ 0 ].d[ 0 ] == 2 );
    mDimOutputs = nvinfer1::DimsCHW( inputs[ 0 ].d[ 1 ], inputs[ 0 ].d[ 2 ], inputs[ 0 ].d[ 3 ] );
    mSize = inputs[ 0 ].d[ 1 ] * inputs[ 0 ].d[ 2 ] * inputs[ 0 ].d[ 3 ];
}

void SplitLayer::serialize( void* buffer )
{
    int* d = reinterpret_cast< int* >( buffer );
    d[ 0 ] = mDimOutputs.c();
    d[ 1 ] = mDimOutputs.h();
    d[ 2 ] = mDimOutputs.w();
    d[ 3 ] = mSize;
}

int SplitLayer::enqueue( int batchSize, const void* const *inputs, void** outputs, void* workspace, cudaStream_t stream )
{
    int batchOffset = 2 * mSize;
    for ( int batch = 0; batch < batchSize; ++batch )
    {
        for ( int i = 0; i < 2; ++i )
        {
            cudaMemcpyAsync
            (
                outputs[ i ],
                inputs[ 0 ] + ( batchOffset * batch + i * mSize ) * sizeof( float ),
                mSize * sizeof( float ),
                cudaMemcpyDeviceToDevice,
                stream
            );
        }
    }

    return 0;
}
