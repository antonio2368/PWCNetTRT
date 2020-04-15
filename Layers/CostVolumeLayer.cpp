//
// Created by Antonio on 08/04/2020.
//

#include "CostVolumeLayer.h"
#include "../Utils/TensorUtils.h"
#include "../Utils/cudaUtils.h"

#include <cuda_runtime_api.h>

#include <cstring>
#include <cassert>

CostVolumeLayer::CostVolumeLayer( int searchRange )
{
    mMaxOffset = 2 * searchRange + 1;
    createDescriptors();
}

void CostVolumeLayer::createDescriptors()
{
    if ( mCudnn == nullptr )
    {
        CHECK( cudnnCreate( &mCudnn ) )
    }

    if ( mMultiplyDesc == nullptr )
    {
        CHECK( cudnnCreateOpTensorDescriptor( &mMultiplyDesc ) )
    }

    if ( mMultiplyOperandDesc == nullptr )
    {
        CHECK( cudnnCreateTensorDescriptor( &mMultiplyOperandDesc ) )
    }

    if ( mMeanDesc == nullptr )
    {
        CHECK( cudnnCreateReduceTensorDescriptor( &mMeanDesc ) )
    }

    if ( mOutputDesc == nullptr )
    {
        CHECK( cudnnCreateTensorDescriptor( &mOutputDesc ) )
    }
}

CostVolumeLayer::CostVolumeLayer( const void* buffer, std::size_t size )
{
    memcpy( &mMaxOffset, buffer, sizeof( int ) );
}

nvinfer1::Dims CostVolumeLayer::getOutputDimensions( int index, const nvinfer1::Dims* inputs, int nbInputsDims )
{
    assert( nbInputsDims == 2 );
    assert( inputs[ 0 ].nbDims == 3 );
    return nvinfer1::DimsCHW{ 1, inputs[ 0 ].d[ 1 ], inputs[ 0 ].d[ 2 ] };
}

void CostVolumeLayer::configure( const nvinfer1::Dims* inputs, int nbInputs, const nvinfer1::Dims* outputs, int nbOutputs, int maxBatchSize )
{
    assert( nbInputs == 2 );
    mFirstInputDims = nvinfer1::DimsCHW{ inputs[ 0 ].d[ 0 ], inputs[ 0 ].d[ 1 ], inputs[ 0 ].d[ 2 ] };
    mSecondInputDims = nvinfer1::DimsCHW{ inputs[ 1 ].d[ 0 ], inputs[ 1 ].d[ 1 ], inputs[ 1 ].d[ 2 ] };
}

void CostVolumeLayer::serialize( void* buffer )
{
    memcpy( buffer, &mMaxOffset, sizeof( int ) );
}

void CostVolumeLayer::terminate()
{
    if ( mMultiplyOperandDesc != nullptr )
    {
        CHECK( cudnnDestroyTensorDescriptor( mMultiplyOperandDesc ) )
    }

    if ( mMultiplyDesc != nullptr )
    {
        CHECK( cudnnDestroyOpTensorDescriptor( mMultiplyDesc ) )
    }

    if ( mMeanDesc != nullptr )
    {
        CHECK( cudnnDestroyReduceTensorDescriptor( mMeanDesc ) )
    }

    if ( mOutputDesc != nullptr )
    {
        CHECK( cudnnDestroyTensorDescriptor( mOutputDesc ) )
    }

    if ( mCudnn != nullptr )
    {
        CHECK( cudnnDestroy( mCudnn ) )
    }

    mCudnn = nullptr;
    mMultiplyDesc = nullptr;
    mMultiplyOperandDesc = nullptr;
}

int CostVolumeLayer::enqueue( int batchSize, const void* const * inputs, void** outputs, void* workspace, cudaStream_t stream )
{
    CHECK( cudnnSetStream( mCudnn, stream ) )

    setTensorDescriptors( batchSize );

    for ( int i = 0; i < mMaxOffset; ++i )
    {
        for ( int j = 0; j < mMaxOffset; ++j )
        {
            float* pdst = static_cast< float* >( workspace );
            Utils::getSlice
            (
                static_cast< const float* >(inputs[ 1 ]),
                pdst,
                mSecondInputDims,
                nvinfer1::DimsCHW{ 0, i, j },
                mFirstInputDims,
                batchSize,
                stream
            );

            std::size_t inputSize{ batchSize * Utils::tensorSize( mFirstInputDims ) };
            CHECK( cudnnOpTensor( mCudnn, mMultiplyDesc, &Consts::kOne, mMultiplyOperandDesc, pdst, &Consts::kOne, mMultiplyOperandDesc, inputs[ 0 ], &Consts::kZero, mMultiplyOperandDesc, pdst ) )
            CHECK( cudnnReduceTensor( mCudnn, mMeanDesc, nullptr, 0, pdst + inputSize, inputSize * sizeof( float ),
                                      &Consts::kOne, mMultiplyOperandDesc, pdst,
                                      &Consts::kZero, mOutputDesc, *outputs ) )
            ++outputs;
        }
    }
    return 0;
}

void CostVolumeLayer::setTensorDescriptors( int const batchSize )
{
    auto const cuDataType = CUDNN_DATA_FLOAT;
    auto tensorDims = nvinfer1::Dims{ 4, { batchSize, mFirstInputDims.c(), mFirstInputDims.h(), mFirstInputDims.w() } };
    auto tensorStrides = Utils::getStrides( tensorDims );
    CHECK( cudnnSetTensorNdDescriptor ( mMultiplyOperandDesc, cuDataType, tensorDims.nbDims, tensorDims.d, tensorStrides.d ) )

    CHECK( cudnnSetOpTensorDescriptor( mMultiplyDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN ) )

    tensorDims.d[ 1 ] = 1;
    tensorStrides = Utils::getStrides( tensorDims );
    CHECK( cudnnSetTensorNdDescriptor( mOutputDesc, cuDataType, tensorDims.nbDims, tensorDims.d, tensorStrides.d ) )

    CHECK( cudnnSetReduceTensorDescriptor( mMeanDesc, CUDNN_REDUCE_TENSOR_AVG, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES ) )
}