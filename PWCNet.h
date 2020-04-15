//
// Created by Antonio on 29/03/2020.
//

#ifndef PWCNET_PWCNET_H
#define PWCNET_PWCNET_H

#include "Utils/buffer.h"
#include "Utils/Profiler.h"
#include "Layers/PluginFactory.h"

#include <NvInfer.h>

#include <memory>
#include <unordered_map>
#include <string>
#include <utility>
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
    int searchRange{ 0 };
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
    {
        mProfiler = std::unique_ptr< Profiler >{ new Profiler{ false } };
    }

    bool build();
    bool infer();
    bool teardown();

    float getTotalInferenceTime() const noexcept
    {
        return mProfiler->getTotalTime();
    }

private:
    std::unique_ptr< Profiler > mProfiler;
    PWCNetParams mParams;
    std::unordered_map< std::string, nvinfer1::Weights > mWeightMap;

    // mEngine should always be after the factories because it's destructor should be called first
    PluginFactory mPluginFactory;
    std::shared_ptr< nvinfer1::ICudaEngine > mEngine;

    std::unordered_map< std::string, nvinfer1::Weights > loadWeights( std::string const & file );

    bool processInput( common::BufferManager& buffers );
    void printOutput( common::BufferManager& buffers );

    // Network definition
    bool constructNetwork( UniquePtr< nvinfer1::IBuilder >& builder, UniquePtr< nvinfer1::INetworkDefinition >& network );
    nvinfer1::ILayer* addConvolutionLayer
    (
        const std::string & layerName,
        nvinfer1::ITensor* input,
        int filterNum,
        nvinfer1::DimsHW && kernelSize,
        nvinfer1::DimsHW && strides,
        PWCNet::UniquePtr< INetworkDefinition >& network,
        bool activation = true,
        std::string const& layerNameExtension = "",
        nvinfer1::DimsHW && dilation = nvinfer1::DimsHW{ 1, 1 }
    );
    nvinfer1::ILayer* addDeconvolutionLayer
    (
            const std::string & layerName,
            nvinfer1::ITensor* input,
            int filterNum,
            nvinfer1::DimsHW && kernelSize,
            nvinfer1::DimsHW && strides,
            PWCNet::UniquePtr< INetworkDefinition >& network
    );
    nvinfer1::ITensor* calculateCostVolume
    (
        nvinfer1::ITensor* firstInput,
        nvinfer1::ITensor* secondInput,
        PWCNet::UniquePtr< INetworkDefinition >& network
    );
    std::pair< nvinfer1::ITensor*, nvinfer1::ITensor* > predictFlow
    (
        nvinfer1::ITensor* corr,
        nvinfer1::ITensor* c1,
        nvinfer1::ITensor* upFlow,
        nvinfer1::ITensor* upFeat,
        int level,
        PWCNet::UniquePtr< INetworkDefinition >& network
    );

    nvinfer1::ITensor* refineFlow
    (
        nvinfer1::ITensor* upFeat,
        nvinfer1::ITensor* upFlow,
        int level,
        PWCNet::UniquePtr< INetworkDefinition >& network
    );
    void extractFeatures
    (
        nvinfer1::ITensor* input,
        PWCNet::UniquePtr< INetworkDefinition >& network,
        std::vector< nvinfer1::ITensor* >& extractedFeatures,
        std::string const & inputName
    );
};


#endif //PWCNET_PWCNET_H
