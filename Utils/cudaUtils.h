//
// Created by Antonio on 07/04/2020.
//

#ifndef PWCNET_CUDAUTILS_H
#define PWCNET_CUDAUTILS_H

#include "logger.h"

#include <cuda_runtime_api.h>

#define CUDA_THREADS_NUM 512

inline int cudaBlockNum( const int N )
{
    return ( N + CUDA_THREADS_NUM - 1 ) / CUDA_THREADS_NUM;
}

struct Consts
{
    static const float kZero;
    static const float kOne;
};

#define CHECK(status) {\
                auto res = (status); \
                if ( res ) \
                { \
                    gLogError << "Cuda failure: " << __FILE__ << " at line " << __LINE__ << " in " << __FUNCTION__ << std::endl; \
                } \
            }

#endif //PWCNET_CUDAUTILS_H
