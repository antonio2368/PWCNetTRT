//
// Created by Antonio on 29/03/2020.
//

#include "PWCNet.h"
#include "Layers/LeakyReluLayer.h"
#include "Layers/CostVolumeLayer.h"
#include "Layers/ImageWarpLayer.h"

#include <array>
#include <vector>

namespace
{

    template< typename T >
    typename std::enable_if_t< std::is_integral< T >::value, void >
    readInput( std::ifstream& is, T& num )
    {
        is.read( reinterpret_cast< char* >( &num ), sizeof( num ) );
    }

    void readInput( std::ifstream& is, std::string& string )
    {
        std::getline( is, string, ' ' );
        is >> std::ws;
    }

    void readInput( std::ifstream& is, float& num )
    {
        std::string stringNum;
        std::getline( is, stringNum , ' ');
        num = std::strtod( stringNum.c_str(), NULL );

        is >> std::ws;
    }

    void printTensorDimensions( nvinfer1::ITensor const& tensor )
    {
        std::cout << "Printing dimensions for: " << tensor.getName() << '\n';
        auto const dimensions = tensor.getDimensions().d;
        for ( int i = 0; i < tensor.getDimensions().nbDims; ++i )
        {
            std::cout << dimensions[ i ] << ' ';
        }

        std::cout << std::endl;
    }

} // namespace

bool PWCNet::build()
{
    mWeightMap = loadWeights( mParams.weightsFile );

    auto builder = UniquePtr< nvinfer1::IBuilder >( nvinfer1::createInferBuilder( gLogger ) );
    if ( !builder )
    {
        return false;
    }
    builder->setMaxWorkspaceSize( 1 << 20 );

    auto network = UniquePtr< nvinfer1::INetworkDefinition >( builder->createNetwork() );
    if ( !network )
    {
        return false;
    }

    auto constructed = constructNetwork( builder, network );
    if ( !constructed )
    {
        return false;
    }

    assert( network->getNbInputs() == 2 );
    auto inputDims = network->getInput( 0 )->getDimensions();
    assert( inputDims.nbDims == 3 );

    assert( network->getNbOutputs() == 1 );
    auto outputDims = network->getOutput( 0 )->getDimensions();
    assert( outputDims.nbDims == 3 );

    return true;
}

bool PWCNet::constructNetwork( PWCNet::UniquePtr< IBuilder >& builder, PWCNet::UniquePtr< INetworkDefinition >& network)
{
    nvinfer1::ITensor* firstImage = network->addInput
    (
            mParams.inputTensorNames[ 0 ].c_str(),
            nvinfer1::DataType::kFLOAT,
            nvinfer1::DimsCHW( 3, mParams.inputH, mParams.inputW )
    );
    assert( firstImage );

    nvinfer1::ITensor* secondImage = network->addInput
    (
        mParams.inputTensorNames[ 1 ].c_str(),
        nvinfer1::DataType::kFLOAT,
        nvinfer1::DimsCHW( 3, mParams.inputH, mParams.inputW )
    );
    assert( secondImage );

    // extract features
    std::vector< ITensor* > c1;
    std::vector< ITensor* > c2;

    extractFeatures( firstImage, network, c1, "firstImage" );
    extractFeatures( secondImage, network, c2, "secondImage" );

    nvinfer1::ITensor* feat{ nullptr };
    nvinfer1::ITensor* flow{ nullptr };
    nvinfer1::ITensor* upFlow{ nullptr };
    nvinfer1::ITensor* upFeat{ nullptr };
    for ( int level{ mParams.pyramidLevels }; level >= mParams.flowPredLevels; --level )
    {
        if ( level == mParams.pyramidLevels )
        {
            auto corr = calculateCostVolume
            (
                c1[ level - 1 ],
                c2[ level - 1 ],
                network
            );
            assert( corr );

            auto const resultPair = predictFlow
            (
                corr,
                nullptr,
                nullptr,
                nullptr,
                level,
                network
            );

            feat = resultPair.first;
            flow = resultPair.second;
        }
        else
        {

            float scaler = 20.f / ( 1 << level );
            nvinfer1::Weights scaleWeights{ nvinfer1::DataType::kFLOAT, &scaler, 1 };
            nvinfer1::Weights powWeights{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
            nvinfer1::Weights shiftWeights{ nvinfer1::DataType::kFLOAT, nullptr, 0 };

            IScaleLayer* scalerLayer = network->addScale
            (
                *upFlow,
                nvinfer1::ScaleMode::kUNIFORM,
                shiftWeights,
                scaleWeights,
                powWeights
            );
            assert( scalerLayer );

            std::vector< ITensor* > warpInputs{ c2[ level - 1], scalerLayer->getOutput( 0 ) };
//            std::vector< ITensor* > warpInputs{ c2[ level - 1], upFlow };

            nvinfer1::IPluginLayer* warpLayer = network->addPlugin
            (
                warpInputs.data(),
                2,
                *mPluginFactory.createPlugin< ImageWarpLayer >( "warp" )
            );
            assert( warpLayer );

            auto corr = calculateCostVolume
            (
                c1[ level - 1 ],
                warpLayer->getOutput( 0 ),
                network
            );

            auto const resultPair = predictFlow
            (
                 corr,
                 c1[ level - 1],
                 upFlow,
                 upFeat,
                 level,
                 network
            );

            feat = resultPair.first;
            flow = resultPair.second;
        }

        if ( level != mParams.flowPredLevels )
        {
            auto const upFlowLayer = addDeconvolutionLayer
            (
                std::string{ "deconv" } + std::to_string( level ),
                flow,
                2,
                nvinfer1::DimsHW{ 4, 4 },
                nvinfer1::DimsHW{ 2, 2 },
                network
            );

            upFlow = upFlowLayer->getOutput( 0 );

            auto const upFeatLayer = addDeconvolutionLayer
            (
                std::string{ "upfeat" } + std::to_string( level ),
                feat,
                2,
                nvinfer1::DimsHW{ 4, 4 },
                nvinfer1::DimsHW{ 2, 2 },
                network
            );

            upFeat = upFeatLayer->getOutput( 0 );
        }
        else
        {
            flow = refineFlow
            (
                feat,
                flow,
                level,
                network
            );
        }
    }

    flow->setName( mParams.outputTensorNames[ 0 ].c_str() );
    network->markOutput( *flow );

    builder->setMaxBatchSize( mParams.batchSize );

    mEngine = std::shared_ptr< nvinfer1::ICudaEngine >
    (
        builder->buildCudaEngine( *network ), InferDeleter()
    );

    if ( !mEngine )
    {
        return false;
    }

    return true;
}

bool PWCNet::infer()
{
    common::BufferManager buffers( mEngine, mParams.batchSize );

    auto context = UniquePtr< nvinfer1::IExecutionContext >( mEngine->createExecutionContext() );
    if ( !context )
    {
        return false;
    }

    IProfiler* profiler{ mProfiler.get() };
    context->setProfiler( profiler );

    assert( mParams.inputTensorNames.size() == 2 );
    if ( !processInput( buffers ) )
    {
        return false;
    }

    buffers.copyInputToDevice();

    bool const status = context->execute( mParams.batchSize, buffers.getDeviceBindings().data() );
    if ( !status )
    {
        return status;
    }

    buffers.copyOutputToHost();
    printOutput( buffers );

    return true;
}

bool PWCNet::processInput( common::BufferManager& buffers)
{
    float* firstHostDataBuffer = static_cast< float* >( buffers.getHostBuffer( ( mParams.inputTensorNames[0] ) ) );
    float* secondHostDataBuffer = static_cast< float* >( buffers.getHostBuffer( ( mParams.inputTensorNames[1] ) ) );

    std::size_t elemNumber = 3 * mParams.inputH * mParams.inputW;

    for ( int i = 0; i < elemNumber; ++i )
    {
        firstHostDataBuffer[ i ] = 0.5;
    }

    elemNumber = 3 * mParams.inputH * mParams.inputW;
    for ( int i = 0; i < elemNumber; ++i )
    {
        secondHostDataBuffer[ i ] = 0.5;
    }

    return true;
}

void PWCNet::printOutput( common::BufferManager& buffers)
{
    float* prob = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    std::cout << "Output:\n";
    for (int i = 0; i < 2 * 16 * 16; ++i)
    {
        std::cout << i << ": " << prob[i] << '\n';
    }

    std::cout << std::endl;
}

std::unordered_map< std::string, nvinfer1::Weights > PWCNet::loadWeights( std::string const& file )
{
    gLogInfo << "Loading weights from: " << file << std::endl;
    std::ifstream input( file, std::ios::binary );
    assert( input.is_open() && "Unable to load weight file." );

    std::int32_t count;
    readInput( input, count );
    assert( count > 0 && "Invalid weight map file." );
    gLogInfo << "Weight count: " << count << std::endl;

    std::unordered_map< std::string, nvinfer1::Weights > weightMap;
    while( count -- )
    {
        nvinfer1::Weights wt{ nvinfer1::DataType::kFLOAT, nullptr, 0 };

        std::string name;
        readInput( input, name );
        gLogInfo << "Weight name: " << name << std::endl;

        int16_t type{ 0 };
        readInput( input, type );
        gLogInfo << "Wegiht type: " << type << std::endl;

        std::uint32_t size{ 0 };
        readInput( input, size );
        gLogInfo << "Weight size: " << size << std::endl;

        wt.type = static_cast< nvinfer1::DataType >( type );

        gLogInfo << "Reading weights..." << std::endl;
        if ( wt.type == nvinfer1::DataType::kFLOAT )
        {
            float* val = new float[ size ];
            for ( std::uint32_t x{ 0 }; x < size; ++ x )
            {
                readInput( input, val[x] );
            }
            wt.values = val;
        }

        wt.count = size;
        weightMap[ name ] = wt;
    }

    return weightMap;
}

void PWCNet::extractFeatures
(
    nvinfer1::ITensor* input,
    PWCNet::UniquePtr< INetworkDefinition >& network,
    std::vector< nvinfer1::ITensor* >& extractedFeatures,
    std::string const & inputName
)
{
    std::array< int, 6 > channels{ 16, 32, 64, 96, 128, 196 };

    nvinfer1::ITensor* currentTensor{ input };
    for ( int i{ 0 }; i < mParams.pyramidLevels; ++i )
    {
        std::string layerLevelName = std::string{ "conv" } + std::to_string( i + 1 );

        std::string layerName = layerLevelName + "a";
        auto const conva = addConvolutionLayer
        (
            layerName,
            currentTensor,
            channels[ i ],
            nvinfer1::DimsHW{ 3, 3 },
            nvinfer1::DimsHW{ 2, 2 },
            network,
            true,
            inputName
        );

        currentTensor = conva->getOutput( 0 );

        layerName = layerLevelName + "aa";
        auto const convaa = addConvolutionLayer
        (
            layerName,
            currentTensor,
            channels[ i ],
            nvinfer1::DimsHW{ 3, 3 },
            nvinfer1::DimsHW{ 1, 1 },
            network,
            true,
            inputName
        );

        currentTensor = convaa->getOutput( 0 );

        layerName = layerLevelName + "b";
        auto const convb = addConvolutionLayer
        (
            layerName,
            currentTensor,
            channels[ i ],
            nvinfer1::DimsHW{ 3, 3 },
            nvinfer1::DimsHW{ 1, 1 },
            network,
            true,
            inputName
        );

        currentTensor = convb->getOutput( 0 );

        extractedFeatures.push_back( currentTensor );
    }
}

nvinfer1::ILayer* PWCNet::addConvolutionLayer
(
    const std::string & layerName,
    nvinfer1::ITensor *input,
    int filterNum,
    nvinfer1::DimsHW && kernelSize,
    nvinfer1::DimsHW && strides,
    PWCNet::UniquePtr<INetworkDefinition> &network,
    bool activation,
    std::string const& layerNameExtension,
    nvinfer1::DimsHW && dilation
)
{
    // Pad input
    auto const paddings = Utils::getPadding( input->getDimensions(), strides, kernelSize, dilation );

    IPaddingLayer* paddingLayer = network->addPadding
    (
        *input,
        paddings.first,
        paddings.second
    );

    assert( paddingLayer );

    std::string const kernelWeights = layerName + "kernel";
    std::string const biasWeights = layerName + "bias";
    IConvolutionLayer* conv = network->addConvolution
    (
        *paddingLayer->getOutput( 0 ),
        filterNum,
        kernelSize,
        mWeightMap[ kernelWeights.c_str() ],
        mWeightMap[ biasWeights.c_str() ]
    );
    assert( conv );
    conv->setStride( strides );
    conv->setDilation( dilation );
    conv->setName( ( layerName + layerNameExtension ).c_str() );

    if ( !activation )
    {
        return conv;
    }

    input = conv->getOutput( 0 );

    nvinfer1::IPluginLayer* leakyRelu = network->addPlugin
    (
        &input,
        1,
        *mPluginFactory.createPlugin< LeakyReluLayer >( "leakyrelu", 0.1f )
    );
    assert( leakyRelu );

    return leakyRelu;
}

nvinfer1::ILayer* PWCNet::addDeconvolutionLayer
(
    const std::string & layerName,
    nvinfer1::ITensor* input,
    int filterNum,
    nvinfer1::DimsHW && kernelSize,
    nvinfer1::DimsHW && strides,
    PWCNet::UniquePtr< INetworkDefinition >& network
)
{
    std::string const kernelWeights = layerName + "kernel";
    std::string const biasWeights = layerName + "bias";
    IDeconvolutionLayer* deconv = network->addDeconvolution
    (
        *input,
        filterNum,
        kernelSize,
        mWeightMap[ kernelWeights.c_str() ],
        mWeightMap[ biasWeights.c_str() ]
    );
    assert( deconv );
    deconv->setStride( strides );
    deconv->setPadding( nvinfer1::DimsHW{ 1, 1 } );
    deconv->setName( layerName.c_str() );

    return deconv;
}

nvinfer1::ITensor* PWCNet::calculateCostVolume
(
    nvinfer1::ITensor* firstInput,
    nvinfer1::ITensor* secondInput,
    PWCNet::UniquePtr< INetworkDefinition >& network
)
{
    IPaddingLayer* paddingLayer = network->addPadding
    (
        *secondInput,
        nvinfer1::DimsHW{ mParams.searchRange, mParams.searchRange },
        nvinfer1::DimsHW{ mParams.searchRange, mParams.searchRange }
    );
    assert( paddingLayer );

    std::vector< ITensor* > inputs{ firstInput, paddingLayer->getOutput( 0 ) };
    IPluginLayer* costVolumeLayer = network->addPlugin
    (
        inputs.data(),
        2,
        *mPluginFactory.createPlugin< CostVolumeLayer >( "costvolume", mParams.searchRange )
    );
    assert( costVolumeLayer );

    std::vector< nvinfer1::ITensor* > costVolumeOutputs;

    for( int i = 0; i < costVolumeLayer->getNbOutputs(); ++i )
    {
        costVolumeOutputs.push_back( costVolumeLayer->getOutput( i ) );
    }

    IConcatenationLayer* concatLayer = network->addConcatenation
    (
        costVolumeOutputs.data(),
        costVolumeOutputs.size()
    );
    assert( concatLayer );

    nvinfer1::ITensor* concatOutput = concatLayer->getOutput( 0 );
    IPluginLayer* leakyRelu = network->addPlugin
    (
        &concatOutput,
        1,
        *mPluginFactory.createPlugin< LeakyReluLayer >( "leakyrelu", 0.1f )
    );
    assert( leakyRelu );

    return leakyRelu->getOutput( 0 );
}

std::pair< nvinfer1::ITensor*, nvinfer1::ITensor* > PWCNet::predictFlow
(
    nvinfer1::ITensor* corr,
    nvinfer1::ITensor* c1,
    nvinfer1::ITensor* upFlow,
    nvinfer1::ITensor* upFeat,
    int const level,
    PWCNet::UniquePtr< INetworkDefinition >& network
)
{
    nvinfer1::ITensor* currentTensor = corr;
    if ( c1 != nullptr && upFlow != nullptr && upFeat != nullptr )
    {
        std::vector< nvinfer1::ITensor* > inputs{ corr, c1, upFlow, upFeat };
        IConcatenationLayer* concatLayer = network->addConcatenation
        (
            inputs.data(),
            inputs.size()
        );
        assert( concatLayer );
        currentTensor = concatLayer->getOutput( 0 );

    }

    std::array< int, 5 > channels{ 128, 128, 96, 64, 32 };

    std::string layerName = std::string{ "conv" } + std::to_string( level ) + std::string{ "_" };
    for ( int i = 0; i < channels.size(); ++i )
    {
        auto const convolution = addConvolutionLayer
        (
            layerName + std::to_string( i ),
            currentTensor,
            channels[ i ],
            nvinfer1::DimsHW{ 3, 3 },
            nvinfer1::DimsHW{ 1, 1 },
            network
        );
        currentTensor = convolution->getOutput( 0 );
    }

    layerName = std::string{ "flow" } + std::to_string( level );
    auto const flowPredictor = addConvolutionLayer
    (
        layerName,
        currentTensor,
        2,
        nvinfer1::DimsHW{ 3, 3 },
        nvinfer1::DimsHW{ 1, 1 },
        network,
        false
    );

    return { currentTensor, flowPredictor->getOutput( 0 ) };
}

nvinfer1::ITensor* PWCNet::refineFlow
(
        nvinfer1::ITensor* feat,
        nvinfer1::ITensor* flow,
        int level,
        PWCNet::UniquePtr< INetworkDefinition >& network
)
{
    ITensor* currentTensor{ feat };

    std::array< int, 7 > channels{ 128, 128, 128, 96, 64, 32, 2 };
    std::array< int, 7 > dilation{ 1, 2, 4, 8, 16, 1, 1 };

    for ( int i = 0; i < channels.size(); ++i )
    {
        auto const convolutionLayer = addConvolutionLayer
        (
            std::string{ "dc_conv" } + std::to_string( i + 1 ),
            currentTensor,
            channels[ i ],
            nvinfer1::DimsHW{ 3, 3 },
            nvinfer1::DimsHW{ 1, 1 },
            network,
            true,
            "",
            nvinfer1::DimsHW{ dilation[ i ], dilation [ i ] }
        );

        currentTensor = convolutionLayer->getOutput( 0 );
    }


    IElementWiseLayer* adder = network->addElementWise
    (
        *currentTensor,
        *flow,
        nvinfer1::ElementWiseOperation::kSUM
    );
    assert( adder );

    return adder->getOutput( 0 );

}

bool PWCNet::teardown()
{
    // Release weights host memory
    for (auto& mem : mWeightMap)
    {
        auto weight = mem.second;
        if (weight.type == DataType::kFLOAT)
        {
            delete[] static_cast<const uint32_t*>(weight.values);
        }
        else
        {
            delete[] static_cast<const uint16_t*>(weight.values);
        }
    }

    return true;
}
