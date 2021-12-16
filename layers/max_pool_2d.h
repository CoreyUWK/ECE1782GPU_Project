#ifndef MAX_POOL_2D_H
#define MAX_POOL_2D_H

__global__ void max_pool_2d(float *in, int in_rows, int in_cols, float *out,
                            int out_rows, int out_cols);

#endif
