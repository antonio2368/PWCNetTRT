//
// Created by Antonio on 11/04/2020.
//

#include "ImageWarpLayer.h"
#include "../Utils/cudaUtils.h"
#include "../Utils/cudaUtilsGpu.h"
#include "../Utils//TensorUtils.h"

#include <cassert>

ImageWarpLayer::ImageWarpLayer()
{
    createDescriptors();
}

void ImageWarpLayer::createDescriptors()
{
    if ( mCudnn == nullptr )
    {
        CHECK( cudnnCreate( &mCudnn ) )
    }

    if ( mFlowChannelFlatDesc == nullptr )
    {
        CHECK( cudnnCreateTensorDescriptor( &mFlowChannelFlatDesc ) )
    }

    if ( mFlatAddOp == nullptr )
    {
        CHECK( cudnnCreateOpTensorDescriptor( &mFlatAddOp ) )
    }

    if ( mFlatMultiplyOp == nullptr )
    {
        CHECK( cudnnCreateOpTensorDescriptor( &mFlatMultiplyOp ) )
    }
}

nvinfer1::Dims ImageWarpLayer::getOutputDimensions( int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
    assert( nbInputDims == 2 );

    return inputs[ 0 ];
}

void ImageWarpLayer::terminate()
{

    if ( mFlowChannelFlatDesc != nullptr )
    {
        CHECK( cudnnDestroyTensorDescriptor( mFlowChannelFlatDesc ) )
    }

    if ( mFlatAddOp != nullptr )
    {
        CHECK( cudnnDestroyOpTensorDescriptor( mFlatAddOp ) )
    }

    if ( mFlatMultiplyOp != nullptr )
    {
        CHECK( cudnnDestroyOpTensorDescriptor( mFlatMultiplyOp ) )
    }

    if ( mCudnn != nullptr )
    {
        CHECK( cudnnDestroy( mCudnn ) )
    }

    mCudnn = nullptr;
    mFlowChannelFlatDesc = nullptr;
    mFlatAddOp = nullptr;
    mFlatMultiplyOp = nullptr;
}

std::size_t ImageWarpLayer::getWorkspaceSize( int maxBatchSize ) const
{
    std::size_t size = maxBatchSize * 2 * mImageDimensions.h() * mImageDimensions.w() * sizeof( float );
    return maxBatchSize * 2 * mImageDimensions.h() * mImageDimensions.w() * sizeof( float ) * 7;
}

void ImageWarpLayer::configure(const nvinfer1::Dims *inputs, int nbInputs, const nvinfer1::Dims *outputs, int nbOutputs,
                               int maxBatchSize)
{
    assert( nbInputs == 2 );
    assert( nbOutputs == 1);

    mImageDimensions = nvinfer1::DimsCHW{ inputs[ 0 ].d[ 0 ], inputs[ 0 ].d[ 1 ], inputs[ 0 ].d[ 2 ] };
    mFlowDimensions = nvinfer1::DimsCHW{ inputs[ 1 ].d[ 0 ], inputs[ 1 ].d[ 1 ], inputs[ 1 ].d[ 2 ] };
}

std::size_t ImageWarpLayer::getSerializationSize()
{
    return 0;
}

void ImageWarpLayer::serialize(void *buffer)
{
}

void ImageWarpLayer::createBatchedGrid( float* grid, int const batchSize, cudaStream_t stream )
{
    int rowOffset{ mImageDimensions.w() };
    int channelOffset{ rowOffset * mImageDimensions.h() };
    int batchOffset{ channelOffset * mImageDimensions.c() };

     float* output{ grid };
     for ( int i = 0; i < mImageDimensions.h(); ++i )
     {
        float* batchStart{ output };

        for ( int batch = 0; batch < batchSize; ++batch )
        {
            float* columnStart{ batchStart };
            for ( int column = 0; column < mImageDimensions.w(); ++column )
            {
                auto const value = static_cast< float >( i );

                CHECK( cudaMemcpyAsync
                (
                    columnStart,
                    &value,
                    sizeof( float ),
                    cudaMemcpyHostToDevice,
                    stream
                ) )
                ++columnStart;
            }

            batchStart += batchOffset;
        }

        output += rowOffset;
     }

     output = grid + channelOffset;
     for ( int i = 0; i < mImageDimensions.w(); ++i )
     {
         float* batchStart{ output };

         for ( int batch = 0; batch < batchSize; ++batch )
         {
             float* rowStart{ batchStart };
             for ( int row = 0; row < mImageDimensions.h(); ++row )
             {
                 auto value = static_cast< float >( i );

                 CHECK( cudaMemcpyAsync
                 (
                    rowStart,
                    &value,
                    sizeof( float ),
                    cudaMemcpyHostToDevice,
                    stream
                 ) )
                 rowStart += rowOffset;
             }
             batchStart += batchOffset;
         }

         ++output;
     }
}

void ImageWarpLayer::setTensorDescriptors( const int batchSize )
{
    auto const cuDataType{ CUDNN_DATA_FLOAT };
    auto tensorDims = nvinfer1::Dims{ 4, { 1, 1, 1, mFlowDimensions.h() * mFlowDimensions.w() } };
    auto tensorStrides = Utils::getStrides( tensorDims );
    CHECK( cudnnSetTensorNdDescriptor( mFlowChannelFlatDesc, cuDataType, tensorDims.nbDims, tensorDims.d, tensorStrides.d ) )

    CHECK( cudnnSetOpTensorDescriptor( mFlatAddOp, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN ) )
    CHECK( cudnnSetOpTensorDescriptor( mFlatMultiplyOp, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN ) )
}

int ImageWarpLayer::enqueue(int batchSize, const void *const * inputs, void ** outputs, void * workspace, cudaStream_t stream)
{
    CHECK( cudnnSetStream( mCudnn, stream ) )

    setTensorDescriptors( batchSize );

    std::size_t flowSize = Utils::tensorSize( mFlowDimensions ) * batchSize;
    std::size_t flowChannelSize = mFlowDimensions.w() * mFlowDimensions.h();

    const float* image = static_cast< const float* >( inputs[ 0 ] );
    const float* flow = static_cast< const float* >( inputs[ 1 ] );

    float* batchedGrid = static_cast< float* >( workspace );

    float* output = static_cast< float * >( outputs[ 0 ] );
    createBatchedGrid( batchedGrid, batchSize, stream );
    subtractTensors( flowSize, batchedGrid, flow, batchedGrid );

    float* alpha0{ batchedGrid + flowSize };
    float* alpha1{ alpha0 + flowChannelSize };

    float* floor0{ alpha1 + flowChannelSize };
    float* floor1{ floor0 + flowChannelSize };

    float* ceil0{ floor1 + flowChannelSize };
    float* ceil1{ ceil0 + flowChannelSize };

    auto const getAlpha = [ & ]( int const dimensionSize, float* input, float* alpha, float* floor, float* ceil )
    {
        float max = dimensionSize - 2.f;
        float min = 0.f;

        clipTensor( flowChannelSize, input, floor, min, max );
        roundToIntTensor( flowChannelSize, floor, floor );
        addValueToTensor( flowChannelSize, floor, ceil, 1.f );

        subtractTensors( flowChannelSize, input, floor, alpha );
        clipTensor( flowChannelSize, alpha, alpha, 0.f, 1.f );
    };

    // ignoring batches for now
    getAlpha( mFlowDimensions.h(), batchedGrid, alpha0, floor0, ceil0 );

    getAlpha( mFlowDimensions.w(), batchedGrid + flowChannelSize, alpha1, floor1, ceil1 );

    float* topLeftCoordinates = static_cast< float* >(malloc( flowChannelSize * sizeof( float ) ));
    float* topRightCoordinates = static_cast< float* >(malloc( flowChannelSize * sizeof( float ) ));
    float* bottomLeftCoordinates = static_cast< float* >(malloc( flowChannelSize * sizeof( float ) ));
    float* bottomRightCoordinates = static_cast< float* >(malloc( flowChannelSize * sizeof( float ) ));

    float* topLeft{ ceil1 + flowChannelSize };
    float* topRight{ topLeft + flowChannelSize };
    float* bottomLeft{ topRight + flowChannelSize };
    float* bottomRight{ bottomLeft + flowChannelSize };

    float* linearCoordinates{ bottomRight + flowChannelSize };
    auto const gather = [ & ]( const float* rowIndex, const float* columnIndex, float* outputCoordinates )
    {
        float imageWidth = mImageDimensions.w();
        CHECK( cudnnOpTensor
        (
            mCudnn,
            mFlatAddOp,
            &imageWidth,
            mFlowChannelFlatDesc,
            rowIndex,
            &Consts::kOne,
            mFlowChannelFlatDesc,
            columnIndex,
            &Consts::kZero,
            mFlowChannelFlatDesc,
            linearCoordinates
        ))

        cudaMemcpyAsync
        (
            outputCoordinates,
            linearCoordinates,
            flowChannelSize * sizeof( float ),
            cudaMemcpyDeviceToHost,
            stream
        );
    };

    gather( floor0, floor1, topLeftCoordinates );
    gather( floor0, ceil1, topRightCoordinates );
    gather( ceil0 , floor1, bottomLeftCoordinates );
    gather( ceil0, ceil1, bottomRightCoordinates );

    float* interpTop{ batchedGrid };
    float* interpBottom{ batchedGrid + flowChannelSize };
    auto interpolateChannel = [ & ]( const float* channel, float* output )
    {
        gatherFromChannel( channel, topLeftCoordinates, flowChannelSize, topLeft, stream );
        gatherFromChannel( channel, topRightCoordinates, flowChannelSize, topRight, stream );
        gatherFromChannel( channel, bottomLeftCoordinates, flowChannelSize, bottomLeft, stream );
        gatherFromChannel( channel, bottomRightCoordinates, flowChannelSize, bottomRight, stream );

        //             interp_top = alphas[1] * (top_right - top_left) + top_left
        subtractTensors( flowChannelSize, topRight, topLeft, interpTop );
        CHECK( cudnnOpTensor
        (
            mCudnn,
            mFlatMultiplyOp,
            &Consts::kOne,
            mFlowChannelFlatDesc,
            alpha1,
            &Consts::kOne,
            mFlowChannelFlatDesc,
            interpTop,
            &Consts::kZero,
            mFlowChannelFlatDesc,
            interpTop
        ))
        CHECK( cudnnOpTensor
        (
            mCudnn,
            mFlatAddOp,
            &Consts::kOne,
            mFlowChannelFlatDesc,
            interpTop,
            &Consts::kOne,
            mFlowChannelFlatDesc,
            topLeft,
            &Consts::kZero,
            mFlowChannelFlatDesc,
            interpTop
        ))

        //             interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
        subtractTensors( flowChannelSize, bottomRight, bottomLeft, interpBottom );
        CHECK( cudnnOpTensor
       (
           mCudnn,
           mFlatMultiplyOp,
           &Consts::kOne,
           mFlowChannelFlatDesc,
           alpha1,
           &Consts::kOne,
           mFlowChannelFlatDesc,
           interpBottom,
           &Consts::kZero,
           mFlowChannelFlatDesc,
           interpBottom
       ))
        CHECK( cudnnOpTensor
       (
           mCudnn,
           mFlatAddOp,
           &Consts::kOne,
           mFlowChannelFlatDesc,
           interpBottom,
           &Consts::kOne,
           mFlowChannelFlatDesc,
           bottomLeft,
           &Consts::kZero,
           mFlowChannelFlatDesc,
           interpBottom
       ))

        //             interp_top = alphas[1] * (top_right - top_left) + top_left
        subtractTensors( flowChannelSize, interpBottom, interpTop, output );
        CHECK( cudnnOpTensor
       (
           mCudnn,
           mFlatMultiplyOp,
           &Consts::kOne,
           mFlowChannelFlatDesc,
           alpha0,
           &Consts::kOne,
           mFlowChannelFlatDesc,
           output,
           &Consts::kZero,
           mFlowChannelFlatDesc,
           output
       ))
        CHECK( cudnnOpTensor
       (
           mCudnn,
           mFlatAddOp,
           &Consts::kOne,
           mFlowChannelFlatDesc,
           output,
           &Consts::kOne,
           mFlowChannelFlatDesc,
           interpTop,
           &Consts::kZero,
           mFlowChannelFlatDesc,
           output
       ))
    };

    const float* channel{ image };
    for ( int i = 0; i < mImageDimensions.c(); ++i )
    {
        interpolateChannel( channel, output );
        channel += flowChannelSize;
        output += flowChannelSize;
    }

    free( topLeftCoordinates );
    free( topRightCoordinates );
    free( bottomLeftCoordinates );
    free ( bottomRightCoordinates );
}

void ImageWarpLayer::gatherFromChannel( const float* input, float* coordinates, std::size_t coordinateSize, float* output, cudaStream_t stream )
{
    for ( int i = 0; i < coordinateSize; ++i )
    {
        CHECK( cudaMemcpyAsync
        (
            output + i,
            input + static_cast< int >( coordinates[ i ] ),
            sizeof( float ),
            cudaMemcpyDeviceToDevice,
            stream
        ) )
    }

}


