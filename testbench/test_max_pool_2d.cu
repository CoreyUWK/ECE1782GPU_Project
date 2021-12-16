#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <sys/time.h>
#include "../src/utils.cu"
#include "../src/layers/max_pool_2d.cu"

#define PAD_VALUE -INFINITY
#define MAX_TOL 1e-3

// #define STRIDE 3

// #define DEBUG


void initData(float *M, int n_rows, int n_cols) {
    unsigned int offset;

    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            offset = (i * n_cols) + j;
            // M[offset] = (i + j) % 2 == 0 ? -(i+j) : (i + j);
            M[offset] = i * n_cols + j;
        }
    }
}

void printMatrix(float *M, int n_rows, int n_cols) {
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            int offset = i * n_cols + j;
            // printf("%d,%d:%f\t", i, j, M[offset]);
            printf("%f\t", M[offset]);
        }
        printf("\n");
    }
}

void host_max_pool_2d(
        float *X,
        int in_rows,
        int in_cols,
        float *Y,
        int out_rows,
        int out_cols
    ) {
    // padding=same from tensorflow

    int px_pre = (in_cols % STRIDE == 0) ? max(POOL_SIZE - STRIDE, 0) : max(POOL_SIZE - in_cols % STRIDE, 0);
    int py_pre = (in_rows % STRIDE == 0) ? max(POOL_SIZE - STRIDE, 0) : max(POOL_SIZE - in_rows % STRIDE, 0);

    px_pre /= 2;
    py_pre /= 2;

    for (int o_y = 0; o_y < out_rows; o_y++) {
        for (int o_x = 0; o_x < out_cols; o_x++) {

          float output_element = PAD_VALUE;
          float current_element;
          int addr;

          int i_y_min = o_y * STRIDE - py_pre;
          int i_x_min = o_x * STRIDE - px_pre;

          // input dimensions
          for (int i_y = i_y_min; i_y < i_y_min + POOL_SIZE; i_y++) {
              for (int i_x = i_x_min; i_x < i_x_min + POOL_SIZE; i_x++) {

                  addr = i_y * in_cols + i_x;

                  current_element = (
                      i_x >= 0 && i_x < in_cols && i_y >= 0 && i_y < in_rows
                  ) ? X[addr] : PAD_VALUE;

                  output_element = max(output_element, current_element);
              }
          }

          addr = o_y * out_cols + o_x;
          Y[addr] = output_element;
        }
    }
}


int main(int argc, char *argv[]) {
    if (argc != 3) {
       printf(
           "Please pass 2 arguments to the program: the first being the " \
           " number of rows and the second being the number of columns.\n"
       ); 
       return -1;
    }

    // set matrix size
    int in_rows = atoi(argv[1]);
    int in_cols = atoi(argv[2]);

    int in_elements = in_rows * in_cols;
    int in_bytes = in_elements * sizeof(float);

    unsigned int out_cols = (in_cols - 1) / STRIDE + 1;
    unsigned int out_rows = (in_rows - 1) / STRIDE + 1;

    int out_elements = out_rows * out_cols;
    int out_bytes = out_elements * sizeof(float);


    // alloc memory host-side
    float *h_X = (float *) malloc(in_bytes);
    float *h_hY = (float *) malloc(out_bytes);
    float *h_dY = (float *) malloc(out_bytes);
    float *d_X;
    float *d_Y;

    gpuErrchk(cudaDeviceReset());

    double start_time = getTimeStamp();

    cudaHostRegister(h_X, in_bytes, 0);
    cudaHostRegister(h_dY, out_bytes, 0);
    gpuErrchk(cudaMalloc((void **) &d_X, in_bytes));
    gpuErrchk(cudaMalloc((void **) &d_Y, out_bytes));

    initData(h_X, in_rows, in_cols);

#ifdef DEBUG
    printf("X:\n");
    printMatrix(h_X, in_rows, in_cols);
#endif

    gpuErrchk(cudaMemcpy(d_X, h_X, in_bytes, cudaMemcpyHostToDevice));

    dim3 block(32, 32); // configure
    dim3 grid(
            (out_cols + block.x - 1) / block.x,
            (out_rows + block.y - 1) / block.y);
    max_pool_2d<<<grid, block>>>(d_X, in_rows, in_cols, d_Y, out_rows, out_cols);

    gpuErrchk(cudaMemcpy(h_dY, d_Y, out_bytes, cudaMemcpyDeviceToHost));

    double end_time = getTimeStamp();
    int total_time_ms = (int) ceil((end_time - start_time) * 1000);

    host_max_pool_2d(h_X, in_rows, in_cols, h_hY, out_rows, out_cols);

#ifdef DEBUG
    printf("Y:\n");
    printMatrix(h_hY, out_rows, out_cols);
#endif

    for (int i = 0; i < out_elements; i++) {
        if (fabs(h_hY[i] - h_dY[i]) >= MAX_TOL) {
            printf(
                "Error: Result mismatch at offset: %d. " \
                "Expected: %f, Got: %f\n",
                i,
                h_hY[i],
                h_dY[i]
            );
            // return -1;
        }
    }

    printf("Elapsed: %d ms\n", total_time_ms);

    cudaHostUnregister(h_X);
    cudaHostUnregister(h_dY);
    free(h_X);
    free(h_hY);
    free(h_dY);
    gpuErrchk(cudaFree(d_X));
    gpuErrchk(cudaFree(d_Y));

    gpuErrchk(cudaDeviceReset());

    return 0;
}
