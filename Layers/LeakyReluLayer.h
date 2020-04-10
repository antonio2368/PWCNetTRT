//
// Created by Antonio on 06/04/2020.
//

#ifndef PWCNET_LEAKYRELULAYER_H
#define PWCNET_LEAKYRELULAYER_H

#include "NvInfer.h"

#include <cassert>

class LeakyReluLayer : public nvinfer1::IPlugin
{
public:
    LeakyReluLayer( float alpha ) : mAlpha{alpha }
    {}

    LeakyReluLayer(const void* buffer, std::size_t size );

    inline int getNbOutputs() const override
    {
        return 1;
    }

    nvinfer1::Dims getOutputDimensions( int index, const nvinfer1::Dims* inputs, int nbInputDims ) override
    {
        assert( nbInputDims == 1 );
        return inputs[ 0 ];
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

    void configure( const nvinfer1::Dims* inputs, int nbInputs, const nvinfer1::Dims* outputs, int nbOutputs, int maxBatchSize ) override;

    std::size_t getSerializationSize() override
    {
        return sizeof( float ) + sizeof( std::size_t );
    }

    void serialize( void* buffer ) override;
    int enqueue( int batchSize, const void* const * inputs, void** outputs, void* workspace, cudaStream_t stream ) override;

    ~LeakyReluLayer() override
    {

    }
protected:
    float mAlpha;
    std::size_t mInputSize;
};

#endif //PWCNET_LEAKYRELULAYER_H
