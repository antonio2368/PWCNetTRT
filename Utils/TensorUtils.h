//
// Created by Antonio on 09/04/2020.
//

#ifndef PWCNET_TENSORUTILS_H
#define PWCNET_TENSORUTILS_H

#include "NvInfer.h"

namespace Utils
{

std::size_t tensorSize( nvinfer1::Dims const & dims ) noexcept;

void getSlice
(
    const float* input,
    float* output,
    nvinfer1::DimsCHW const & inputDims,
    nvinfer1::DimsCHW const & sliceStart,
    nvinfer1::DimsCHW const & sliceSize,
    int batchSize,
    cudaStream_t stream
);

nvinfer1::Dims getStrides(nvinfer1::Dims const & dims);

nvinfer1::DimsHW getPadding
(
    nvinfer1::Dims const& inputDimension,
    nvinfer1::DimsHW const& strides,
    nvinfer1::DimsHW const& kernelDimension,
    nvinfer1::DimsHW const& dilation
);

}


#endif // PWCNET_TENSORUTILS_H
