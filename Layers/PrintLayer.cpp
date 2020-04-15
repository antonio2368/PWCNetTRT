//
// Created by Antonio on 05/04/2020.
//

#include "PrintLayer.h"
#include "../Utils/TensorUtils.h"
#include "../Utils/cudaUtils.h"

#include <cassert>
#include <cstdio>
#include <cuda_runtime_api.h>

nvinfer1::Dims PrintLayer::getOutputDimensions( int index, const nvinfer1::Dims* inputs, int nbInputDims )
{
    assert( nbInputDims == 1 );
    return inputs[ 0 ];
}

void PrintLayer::configure( const nvinfer1::Dims* inputs, int nbInputs, const nvinfer1::Dims* outputs, int nbOutputs, int maxBatchSize )
{
    mDimInputs = nvinfer1::DimsCHW( inputs[ 0 ].d[ 0 ], inputs[ 0 ].d[ 1 ], inputs[ 0 ].d[ 2 ] );
}

void PrintLayer::serialize( void* buffer )
{
}

int PrintLayer::enqueue( int batchSize, const void* const *inputs, void** outputs, void* workspace, cudaStream_t stream )
{
    std::size_t size = Utils::tensorSize( mDimInputs ) * batchSize;

    CHECK( cudaMemcpyAsync
        (
                outputs[ 0 ],
                inputs[ 0 ],
                size * sizeof( float ),
                cudaMemcpyDeviceToDevice,
                stream
        ) )

    float* printArray = new float[ size ];
    CHECK( cudaMemcpyAsync
    (
        printArray,
        inputs[ 0 ],
        size * sizeof( float ),
        cudaMemcpyDeviceToHost,
        stream
    ) )

    printf( "\n%s\n", mName.c_str() );
    for ( int i = 0; i < size; ++i )
    {
        printf( "%.10f ", printArray[ i ] );
    }

    delete [] printArray;
    return 0;
}
