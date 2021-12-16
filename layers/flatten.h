#ifndef FLATTEN_H
#define FLATTEN_H

__global__ void flatten(float *d_conv_out, float *d_flattened, int channel,
                        int size);

#endif
