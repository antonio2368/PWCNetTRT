#include "PWCNet.h"
#include "Utils/logger.h"

#include <iostream>

int main() {
    PWCNetParams params;
    params.weightsFile = "pwcnet_trt";
    params.inputTensorNames.push_back( "firstImage" );
    params.inputTensorNames.push_back( "secondImage" );
    params.outputTensorNames.push_back( "output" );
    params.pyramidLevels = 6;
    params.flowPredLevels = 2;
    params.inputH = 64;
    params.inputW = 64;
    params.searchRange = 4;

    setReportableSeverity( Logger::Severity::kINFO );

    PWCNet model{ params };

    gLogInfo << "Building model..." << std::endl;
    if ( !model.build() )
    {
        gLogError << "Failed building model!" << std::endl;
    }

    gLogInfo << "Model built!" << std::endl;
    gLogInfo << "Starting inference..." << std::endl;

    if ( !model.infer() )
    {
        gLogError << "Failed inference!" << std::endl;
    }

    gLogInfo << "Inference done!" << std::endl;
    gLogInfo << "Total execution time: " << model.getTotalInferenceTime() << "ms" << std::endl;

    if ( !model.teardown() )
    {
        gLogError << "Failed teardown!" << std::endl;
    }

    gLogInfo << "Teardown done!" << std::endl;
}
