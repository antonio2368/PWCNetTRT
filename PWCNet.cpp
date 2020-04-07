//
// Created by Antonio on 29/03/2020.
//

#include "PWCNet.h"
#include "Layers/LeakyReluLayer.h"

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
//    assert( outputDims.nbDims == 3 );

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

    IPluginLayer* leakyReluLayer = network->addPlugin
    (
        &firstImage,
        1,
        leakyReluFactory.createPlugin( "leakyRelu", 0.01f )
    );
    assert( leakyReluLayer );

    std::vector< ITensor* > inputs = { leakyReluLayer->getOutput( 0 ), secondImage };
    IConcatenationLayer* concat = network->addConcatenation
    (
        inputs.data(),
        2
    );
    assert( concat );

    auto const concatOutput{ concat->getOutput( 0 ) };
    IPluginLayer* leakyReluLayer2 = network->addPlugin
    (
        &concatOutput,
        1,
        leakyReluFactory.createPlugin( "leakyRelu2", 0.02f )
    );
    assert(leakyReluLayer2);

    leakyReluLayer2->getOutput( 0 )->setName( mParams.outputTensorNames[ 0 ].c_str() );
    network->markOutput( *leakyReluLayer2->getOutput( 0 ) );

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
        firstHostDataBuffer[ i ] = -1;
        secondHostDataBuffer[ i ] = -2;
    }

    return true;
}

void PWCNet::printOutput( common::BufferManager& buffers)
{
    float* prob = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    std::cout << "Output:\n";
    for (int i = 0; i < 2 * 3 * mParams.inputH * mParams.inputW; ++i)
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

//bool SampleMNISTAPI::teardown()
//{
//    // Release weights host memory
//    for (auto& mem : mWeightMap)
//    {
//        auto weight = mem.second;
//        if (weight.type == DataType::kFLOAT)
//        {
//            delete[] static_cast<const uint32_t*>(weight.values);
//        }
//        else
//        {
//            delete[] static_cast<const uint16_t*>(weight.values);
//        }
//    }
//
//    return true;
//}
