//
// Created by Antonio on 09/04/2020.
//

#include "TensorUtils.h"
#include "cudaUtils.h"

#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>

namespace Utils
{

std::size_t tensorSize( nvinfer1::Dims const & dims ) noexcept
{
    std::size_t sum{ 1 };

    for ( int i = 0; i < dims.nbDims; ++i )
    {
        sum *= dims.d[ i ];
    }

    return sum;
}

void getSlice
(
    const float* input,
    float* output,
    nvinfer1::DimsCHW const & inputDims,
    nvinfer1::DimsCHW const & sliceStart,
    nvinfer1::DimsCHW const & sliceSize,
    int batchSize,
    cudaStream_t stream
)
{
    nvinfer1::Dims strides = getStrides( { 4, { batchSize, inputDims.c(), inputDims.h(), inputDims.w() } } );
    int const batchOffset = strides.d[ 0 ];
    int const channelOffset = strides.d[ 1 ];
    int const rowOffset = strides.d[ 2 ];

    const float* batchStart{ input };
    float* outputStart{ output };
    for ( int batch = 0; batch < batchSize; ++batch )
    {
        const float* channelStart{ batchStart };
        for ( int channel = 0; channel < sliceSize.c(); ++channel )
        {
            const float* rowStart{ channelStart + sliceStart.h() * rowOffset };
            for ( int row = 0; row < sliceSize.h(); ++row )
            {
                CHECK( cudaMemcpyAsync
                (
                    outputStart,
                    rowStart + sliceStart.w(),
                    sliceSize.w() * sizeof(float),
                   cudaMemcpyDeviceToDevice,
                    stream
                ))
                rowStart += rowOffset;
                outputStart += sliceSize.w();
            }
            channelStart += channelOffset;
        }
        batchStart += batchOffset;
    }
}

nvinfer1::Dims getStrides(nvinfer1::Dims const & dims)
{
    nvinfer1::Dims strides;
    strides.nbDims = dims.nbDims;
    strides.d[strides.nbDims - 1] = 1;
    for (int i = strides.nbDims - 2; i >= 0; i--)
    {
        strides.d[i]    = strides.d[i + 1] * dims.d[i + 1];
        strides.type[i] = nvinfer1::DimensionType::kSPATIAL;
    }
    return strides;
}

std::pair< nvinfer1::DimsHW, nvinfer1::DimsHW > getPadding
(
    nvinfer1::Dims const& inputDimension,
    nvinfer1::DimsHW const& strides,
    nvinfer1::DimsHW const& kernelDimension,
    nvinfer1::DimsHW const& dilation
)
{
    assert( inputDimension.nbDims == 3 );
    auto const calculatePadding = [ & ]( int size, int stride, int kernelSize, int dilation  ) noexcept -> int
    {
        int effectiveFilterSize = ( kernelSize - 1 ) * dilation + 1;

        int outSize = ( size + stride - 1 ) / stride;

        int totalPadding = ( outSize - 1 ) * stride + effectiveFilterSize - size;

        totalPadding = totalPadding > 0 ? totalPadding : 0;

        return totalPadding;
    };

    int totalHeightPadding = calculatePadding( inputDimension.d[ 1 ], strides.h(), kernelDimension.h(), dilation.h() );
    int totalWidthPadding = calculatePadding( inputDimension.d[ 2 ], strides.w(), kernelDimension.w(), dilation.w() );

    int preHeightPadding = totalHeightPadding / 2;
    int preWidthPadding = totalWidthPadding / 2;

    return
    {
        nvinfer1::DimsHW{ preHeightPadding, preWidthPadding },
        nvinfer1::DimsHW{ totalHeightPadding - preHeightPadding, totalWidthPadding - preWidthPadding }
    };
}

}

