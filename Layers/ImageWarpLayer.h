//
// Created by Antonio on 11/04/2020.
//

#ifndef PWCNET_IMAGEWARPLAYER_H
#define PWCNET_IMAGEWARPLAYER_H

#include "NvInfer.h"
#include <cudnn_v7.h>

class ImageWarpLayer : public nvinfer1::IPlugin
{
public:
    ImageWarpLayer();

    inline int getNbOutputs() const override
    {
        return 1;
    }

    nvinfer1::Dims getOutputDimensions( int index, const nvinfer1::Dims* inputs, int nbInputDims ) override;

    int initialize() override
    {
        return 0;
    }

    void terminate() override;

    std::size_t getWorkspaceSize( int maxBatchSize ) const override;

    void configure( const nvinfer1::Dims* inputs, int nbInputs, const nvinfer1::Dims* outputs, int nbOutputs, int maxBatchSize ) override;

    std::size_t getSerializationSize() override;

    void serialize( void* buffer ) override;

    int enqueue( int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream ) override;

    ~ImageWarpLayer() override
    {

    }

protected:
    nvinfer1::DimsCHW mImageDimensions;
    nvinfer1::DimsCHW mFlowDimensions;

    cudnnHandle_t mCudnn{ nullptr };

    cudnnTensorDescriptor_t mFlowChannelFlatDesc{ nullptr };
    cudnnOpTensorDescriptor_t mFlatAddOp{ nullptr };
    cudnnOpTensorDescriptor_t mFlatMultiplyOp{ nullptr };

private:
    void createDescriptors();
    void setTensorDescriptors( int batchSize );

    void createBatchedGrid( float* grid, int batchSize, cudaStream_t stream );
    void gatherFromChannel( const float* input, float* coordinates, std::size_t coordinateSize, float* output, cudaStream_t stream );
};


#endif //PWCNET_IMAGEWARPLAYER_H
