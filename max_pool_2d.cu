#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <sys/time.h>

#define PAD_VALUE -INFINITY
#define MAX_TOL 1e-3

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

void host_max_pool_2d(float *X, float *Y, int n_rows, int n_cols, int kx, int ky) {
    // padding=1, stride=1
    for (int o_y = 0; o_y < n_rows; o_y++) {
        for (int o_x = 0; o_x < n_cols; o_x++) {

          float output_element = PAD_VALUE;
          float current_element;
          unsigned int addr;

          // input dimensions
          for (int i_y = o_y; i_y < o_y + ky; i_y++) {
            for (int i_x = o_x; i_x < o_x + kx; i_x++) {

              addr = i_y * n_cols + i_x;

              current_element =
                  // padding
                  (i_x >= 0 && i_x < n_cols && i_y >= 0 && i_y < n_rows) ?
                  X[addr]
                  : PAD_VALUE;

              output_element = max(output_element, current_element);
            }
          }

          addr = o_y * n_cols + o_x;
          Y[addr] = output_element;
        }
    }
}


__global__ void max_pool_2d(float *X, float *Y, int n_rows, int n_cols, int kx, int ky) {
    // padding=1, stride=1

    unsigned int o_col = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int o_row = blockDim.y * blockIdx.y + threadIdx.y;

    float out_element = PAD_VALUE;
    float current_element;

    unsigned int addr;

    if ((o_col >= n_cols) || (o_row >= n_rows)) {
        return;
    }

    for (int i_col = o_col; i_col < o_col + kx; i_col++) {
        for (int i_row = o_row; i_row < o_row + ky; i_row++) {
            current_element = 
                // padding
                (i_col < n_cols && i_row < n_rows)
                ? X[i_row * n_cols + i_col]
                : PAD_VALUE;

            if (current_element > out_element)
                out_element = current_element;
        }
    }

    addr = o_row * n_cols + o_col;
    Y[addr] = out_element;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
       printf("Please pass 2 arguments to the program: the first being the number of rows and the second being the number of columns.\n" ); 
       return -1;
    }

    // set matrix size
    int n_rows = atoi(argv[1]);
    int n_cols = atoi(argv[2]);

    int n_elements = n_rows * n_cols;
    int n_bytes = n_elements * sizeof(float);

    const unsigned int kx = 2;
    const unsigned int ky = 2;

    // alloc memory host-side
    float *h_X = (float *) malloc(n_bytes);
    float *h_hY = (float *) malloc(n_bytes);
    float *h_dY = (float *) malloc(n_bytes);
    float *d_X;
    float *d_Y;

    gpuErrchk(cudaDeviceReset());

    double start_time = getTimeStamp();

    cudaHostRegister(h_X, n_bytes, 0);
    cudaHostRegister(h_dY, n_bytes, 0);
    gpuErrchk(cudaMalloc((void **) &d_X, n_bytes));
    gpuErrchk(cudaMalloc((void **) &d_Y, n_bytes));

    initData(h_X, n_rows, n_cols);
    // printf("X:\n");
    // printMatrix(h_X, n_rows, n_cols);

    gpuErrchk(cudaMemcpy(d_X, h_X, n_bytes, cudaMemcpyHostToDevice));

    dim3 block(32, 32); // configure
    dim3 grid((n_cols + block.x - 1) / block.x, (n_rows + block.y - 1) / block.y);
    max_pool_2d<<<grid, block>>>(d_X, d_Y, n_rows, n_cols, kx, ky);

    gpuErrchk(cudaMemcpy(h_dY, d_Y, n_bytes, cudaMemcpyDeviceToHost));

    double end_time = getTimeStamp();
    int total_time_ms = (int) ceil((end_time - start_time) * 1000);

    host_max_pool_2d(h_X, h_hY, n_rows, n_cols, kx, ky);
    // printf("Y:\n");
    // printMatrix(h_hY, n_rows, n_cols);

    for (int i = 0; i < n_elements; i++) {
        if (fabs(h_hY[i] - h_dY[i]) >= MAX_TOL) {
            printf(
                "Error: Result mismatch at offset: %d. Expected: %f, Got: %f\n",
                i,
                h_hY[i],
                h_dY[i]
            );
            return -1;
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
