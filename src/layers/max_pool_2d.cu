#include "../config.h"

__global__ void max_pool_2d(float *in, int in_rows, int in_cols, float *out, int out_rows, int out_cols) {
    unsigned int o_col = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int o_row = blockDim.y * blockIdx.y + threadIdx.y;

    float out_element = PAD_VALUE;
    float current_element;
    unsigned int addr;

    if (o_col >= out_cols || o_row >= out_rows) {
        return;
    }

    // Implement padding=same from tensorflow
    int px_pre = (in_cols % STRIDE == 0) ? max(POOL_SIZE - STRIDE, 0) : max(POOL_SIZE - in_cols % STRIDE, 0);
    int py_pre = (in_rows % STRIDE == 0) ? max(POOL_SIZE - STRIDE, 0) : max(POOL_SIZE - in_rows % STRIDE, 0);
    px_pre /= 2;
    py_pre /= 2;

    int i_y_min = o_row * STRIDE - py_pre;
    int i_x_min = o_col * STRIDE - px_pre;

    for (int i_col = i_x_min; i_col < i_x_min + POOL_SIZE; i_col++) {
        for (int i_row = i_y_min; i_row < i_y_min + POOL_SIZE; i_row++) {
            addr = i_row * in_cols + i_col;

            current_element = (
                i_col >= 0 && i_col < in_cols && i_row >= 0 && i_row < in_rows
            ) ? in[addr] : PAD_VALUE;

            if (current_element > out_element)
                out_element = current_element;
        }
    }

    addr = o_row * out_cols + o_col;
    out[addr] = out_element;
}
