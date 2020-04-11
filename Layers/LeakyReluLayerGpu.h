//
// Created by Antonio on 07/04/2020.
//

#ifndef PWCNET_LEAKYRELULAYERGPU_H
#define PWCNET_LEAKYRELULAYERGPU_H

#include "../Utils/cudaUtilsGpu.h"

#include <driver_types.h>

cudaError_t LeakyReluForward( const int count, const float* input, float* output, float alpha );

#endif //PWCNET_LEAKYRELULAYERGPU_H
