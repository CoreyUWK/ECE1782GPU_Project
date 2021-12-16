#ifndef LINEAR_H
#define LINEAR_H

__global__ void linear(float *output, float *input, float *W, float *b,
                       int inSize, int outSize, bool isFinal);

#endif
