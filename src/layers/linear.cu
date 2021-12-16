#include "../utils.h"

__global__ void linear(float *output, float *input, float *W, float *b, int inSize, int outSize, bool isFinal) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if ((j >= outSize)) {
        return;
    }

    float sum = 0.0;
    for (int i = 0; i < inSize; i++) {
        int offset = i * outSize + j;
        sum += input[i] * W[offset];
    }

    // final linear layer don't use relu
    if (isFinal) {
        output[j] = sum + b[j];
    }
    else {
        output[j] = relu(sum + b[j]);
    }
}
