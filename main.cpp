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
    params.inputH = 3;
    params.inputW = 3;

    setReportableSeverity( Logger::Severity::kWARNING );

    PWCNet model{ params };
    assert( model.build() );
    model.infer();
}
