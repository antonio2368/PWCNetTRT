//
// Created by Antonio on 07/04/2020.
//

#ifndef PWCNET_CUDAUTILS_H
#define PWCNET_CUDAUTILS_H

#define CUDA_THREADS_NUM 512

inline int cudaBlockNum( const int N )
{
    return ( N + CUDA_THREADS_NUM - 1 ) / CUDA_THREADS_NUM;
}


#endif //PWCNET_CUDAUTILS_H
