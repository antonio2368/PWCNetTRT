//
// Created by Antonio on 05/04/2020.
//

#include "ReshapeLayer.h"

#include <cassert>
#include <cuda_runtime_api.h>


int ReshapeLayer::enqueue( int batchSize, const void* const *inputs, void** outputs, void* workspace, cudaStream_t stream )
{
    int size = batchSize * mSize;
    cudaMemcpyAsync
    (
        outputs[ 0 ],
        inputs[ 0 ],
        size * sizeof( float ),
        cudaMemcpyDeviceToDevice,
        stream
    );

    return 0;
}
