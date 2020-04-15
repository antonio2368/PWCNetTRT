//
// Created by Antonio on 08/04/2020.
//

#ifndef PWCNET_COSTVOLUMELAYER_H
#define PWCNET_COSTVOLUMELAYER_H

#include "NvInfer.h"
#include "../Utils/TensorUtils.h"

#include <cudnn_v7.h>

class CostVolumeLayer : public nvinfer1::IPlugin {
public:
    CostVolumeLayer(int searchRange);

    CostVolumeLayer(const void *buffer, std::size_t size);

    inline int getNbOutputs() const override {
        return mMaxOffset * mMaxOffset;
    }

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims *inputs, int nbInputsDims) override;

    int initialize() override {
        return 0;
    }

    void terminate() override;

    inline std::size_t getWorkspaceSize(int maxBatchSize) const override {
        return 2 * maxBatchSize * Utils::tensorSize(mFirstInputDims) * sizeof(float);
    }

    void configure(const nvinfer1::Dims *inputs, int nbInputs, const nvinfer1::Dims *outputs, int nbOutputs,
                   int maxBatchSize) override;

    inline std::size_t getSerializationSize() override {
        return sizeof(int);
    }

    void serialize(void *buffer) override;

    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override;

    ~CostVolumeLayer()
    {}

protected:
    int mMaxOffset;
    nvinfer1::DimsCHW mFirstInputDims;
    nvinfer1::DimsCHW mSecondInputDims;

private:
    cudnnHandle_t mCudnn{ nullptr };
    cudnnOpTensorDescriptor_t mMultiplyDesc{ nullptr };
    cudnnTensorDescriptor_t mMultiplyOperandDesc{ nullptr };
    cudnnReduceTensorDescriptor_t mMeanDesc{ nullptr };
    cudnnTensorDescriptor_t mOutputDesc{ nullptr };

    void createDescriptors();
    void setTensorDescriptors( int batchSize );


};

#endif //PWCNET_COSTVOLUMELAYER_H
