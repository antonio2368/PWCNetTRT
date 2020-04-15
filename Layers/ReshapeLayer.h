//
// Created by Antonio on 13/04/2020.
//

#ifndef PWCNET_RESHAPELAYER_H
#define PWCNET_RESHAPELAYER_H
#include "NvInfer.h"

#include "../Utils/TensorUtils.h"

class ReshapeLayer : public nvinfer1::IPlugin
{
public:
    ReshapeLayer( nvinfer1::DimsCHW outputDims ) : mDimOutputs{ outputDims }
    {
        mSize = Utils::tensorSize( outputDims );
    }

    inline int getNbOutputs() const override
    {
        return 1;
    }

    nvinfer1::Dims getOutputDimensions( int index, const nvinfer1::Dims* inputs, int nbInputDims ) override
    {
        return mDimOutputs;
    }

    int initialize() override
    {
        return 0;
    }

    inline void terminate() override {}

    inline std::size_t getWorkspaceSize( int ) const override
    {
        return 0;
    }

    void configure( const nvinfer1::Dims* inputs, int nbInputs, const nvinfer1::Dims* outputs, int nbOutputs, int maxBatchSize ) override
    {}

    int enqueue( int batchSize, const void* const *inputs, void** outputs, void* workspace, cudaStream_t stream ) override;

    std::size_t getSerializationSize() override
    {
        return 0;
    }

    void serialize( void* buffer ) override
    {

    }

protected:
    nvinfer1::DimsCHW mDimOutputs;
    int mSize;
};


#endif //PWCNET_RESHAPELAYER_H
