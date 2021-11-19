#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <sys/time.h>

#define PAD_VALUE -INFINITY
#define MAX_TOL 1e-3

#define KX 2
#define KY 2
#define STRIDE 2

#define DEBUG

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code == cudaSuccess) return;

    fprintf(stderr,"Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
}


/*Use the following to get a timestamp*/
double getTimeStamp() {
        struct timeval tv;
        gettimeofday( &tv, NULL );
        return (double) tv.tv_usec/1000000 + tv.tv_sec;
}


void initData(float *M, int n_rows, int n_cols) {
    unsigned int offset;

    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            offset = (i * n_cols) + j;
            M[offset] = (i + j) % 2 == 0 ? -(i+j) : (i + j);
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
        float *Y,
        unsigned int in_rows,
        unsigned int in_cols,
        unsigned int out_rows,
        unsigned int out_cols,
        unsigned int kx,
        unsigned int ky,
        unsigned int s  // stride
    ) {
    // padding=same

    for (int o_y = 0; o_y < out_rows; o_y++) {
        for (int o_x = 0; o_x < out_cols; o_x++) {

          float output_element = PAD_VALUE;
          float current_element;
          int addr;

          // input dimensions
          for (int i_y = (int)(o_y * s - ky / 2); i_y < (int)(o_y * s + ky / 2 + 1); i_y++) {
              for (int i_x = (int)(o_x * s - kx / 2); i_x < (int)(o_x * s + kx / 2 + 1); i_x++) {

                  addr = i_y * in_cols + i_x;

                  current_element =
                      // padding
                      (i_x >= 0 && i_x < in_cols && i_y >= 0 && i_y < in_rows) ?
                      X[addr]
                      : PAD_VALUE;

                  output_element = max(output_element, current_element);
              }
          }

          addr = o_y * out_cols + o_x;
          Y[addr] = output_element;
        }
    }
}


__global__ void max_pool_2d(
        float *X,
        float *Y,
        unsigned int in_rows,
        unsigned int in_cols,
        unsigned int out_rows,
        unsigned int out_cols,
        unsigned int kx,
        unsigned int ky,
        unsigned int s
    ) {
    // padding=same
    unsigned int o_col = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int o_row = blockDim.y * blockIdx.y + threadIdx.y;

    float out_element = PAD_VALUE;
    float current_element;

    unsigned int addr;

    if ((o_col >= out_cols) || (o_row >= out_rows)) {
        return;
    }

    for (int i_col = (int)(o_col * s - kx / 2); i_col < (int)(o_col * s + kx / 2 + 1); i_col++) {
        for (int i_row = (int)(o_row * s - ky / 2); i_row < (int)(o_row * s + ky / 2 + 1); i_row++) {

            addr = i_row * in_cols + i_col;

            current_element = (
                i_col >= 0 && i_col < in_cols && i_row >= 0 && i_row < in_rows
            ) ? X[addr] : PAD_VALUE;

            if (current_element > out_element)
                out_element = current_element;
        }
    }

    addr = o_row * out_cols + o_col;
    Y[addr] = out_element;
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

    unsigned int out_cols = ceil((in_cols) / STRIDE);
    unsigned int out_rows = ceil((in_rows) / STRIDE);

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
    max_pool_2d<<<grid, block>>>(
            d_X, d_Y, in_rows, in_cols, out_rows, out_cols, KX, KY, STRIDE);

    gpuErrchk(cudaMemcpy(h_dY, d_Y, out_bytes, cudaMemcpyDeviceToHost));

    double end_time = getTimeStamp();
    int total_time_ms = (int) ceil((end_time - start_time) * 1000);

    host_max_pool_2d(
            h_X, h_hY, in_rows, in_cols, out_rows, out_cols, KX, KY, STRIDE);

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
