//
// Created by Antonio on 05/04/2020.
//

#ifndef PWCNET_SPLITLAYER_H
#define PWCNET_SPLITLAYER_H

#include "NvInfer.h"
#include <string>

class PrintLayer : public nvinfer1::IPlugin
{
public:
    PrintLayer( const std::string & name ) : mName{ name }
    {

    }

    inline int getNbOutputs() const override
    {
        return 1;
    }

    nvinfer1::Dims getOutputDimensions( int index, const nvinfer1::Dims* inputs, int nbInputDims ) override;

    int initialize() override
    {
        return 0;
    }

    inline void terminate() override {}

    inline std::size_t getWorkspaceSize( int ) const override
    {
        return 0;
    }

    void configure( const nvinfer1::Dims* inputs, int nbInputs, const nvinfer1::Dims* outputs, int nbOutputs, int maxBatchSize ) override;

    std::size_t getSerializationSize() override
    {
        return 0;
    }

    void serialize( void* buffer ) override;

    int enqueue( int batchSize, const void* const *inputs, void** outputs, void* workspace, cudaStream_t stream ) override;

protected:
    nvinfer1::DimsCHW mDimInputs;
    std::string mName;
};


#endif //PWCNET_SPLITLAYER_H
