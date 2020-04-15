//
// Created by Antonio on 15/04/2020.
//

#ifndef PWCNET_PROFILER_H
#define PWCNET_PROFILER_H

#include "NvInfer.h"
#include "logger.h"

class Profiler : public nvinfer1::IProfiler
{
private:
    float mTotalTime{ 0.0f };
    bool mLayerTime;
public:
    explicit Profiler( bool layerTime = false ) : mLayerTime{ layerTime }
    {

    }

    virtual void reportLayerTime( const char* layerName, float layerMs ) override
    {
        mTotalTime += layerMs;
        if ( mLayerTime )
        {
            gLogInfo << "[TENSORT PROFILER] ";
            gLogInfo << layerName << ": " << layerMs << "ms" << std::endl;
        }
    }

    float getTotalTime() const noexcept
    {
        return mTotalTime;
    }
};

#endif //PWCNET_PROFILER_H
