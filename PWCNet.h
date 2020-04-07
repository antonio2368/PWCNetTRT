//
// Created by Antonio on 29/03/2020.
//

#ifndef PWCNET_PWCNET_H
#define PWCNET_PWCNET_H

#include "Utils/buffer.h"
#include "Layers/PluginFactory.h"

#include <NvInfer.h>

#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

struct PWCNetParams
{
    int batchSize{ 1 };

    std::string weightsFile;

    std::vector< std::string > inputTensorNames;
    std::vector< std::string > outputTensorNames;

    int inputH{ 0 };
    int inputW{ 0 };

    int pyramidLevels{ 0 };
    int flowPredLevels{ 0 };
};

struct InferDeleter
{
    template< typename T >
    void operator()( T* obj ) const
    {
        if ( obj )
        {
            obj->destroy();
        }
    }
};

class PWCNet {
    template< typename T >
    using UniquePtr = std::unique_ptr< T, InferDeleter >;

public:
    PWCNet( PWCNetParams const & params )
        : mEngine{ nullptr }
        , mParams{ params  }
    {}

    bool build();
    bool infer();
    bool teardown();

private:
    PWCNetParams mParams;
    std::unordered_map< std::string, nvinfer1::Weights > mWeightMap;

    // mEngine should always be after the factories because it's destructor should be called first
    PluginFactory< LeakyReluLayer > leakyReluFactory;
    std::shared_ptr< nvinfer1::ICudaEngine > mEngine;

    bool constructNetwork( UniquePtr< nvinfer1::IBuilder >& builder, UniquePtr< nvinfer1::INetworkDefinition >& network );

    std::unordered_map< std::string, nvinfer1::Weights > loadWeights( std::string const & file );

    bool processInput( common::BufferManager& buffers );
    void printOutput( common::BufferManager& buffers );
};


#endif //PWCNET_PWCNET_H
