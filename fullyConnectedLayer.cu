#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 22400
#define OUTPUT_SIZE 256

/*You can use the following for any CUDA function that returns cudaError_t type*/
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

void initArray(float *A, int n) {
    for (int i = 0; i < n; i++) {
        A[i] = rand() % 10;
    }
}

void initWeights(float *W, int inSize, int outSize) {
    for (int i = 0; i < inSize; i++) {
        for (int j = 0; j < outSize; j++) {
            int offset = (i * outSize) + j;
            W[offset] = rand() % 10;
        }
    }
}

void printArray(float *A, int n) {
    for (int i = 0; i < n; i++) {
        printf("%f  ", A[i]);
    }
    printf("\n");
}

void printWeights(float *W, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int offset = i * m + j;
            printf("%f  ", W[offset]);
        }
        printf("\n");
    }
}

void host_FC(float *output, float *input, float *W, float *b, int inSize, int outSize) {
    for (int j = 0; j < outSize; j++) {
        for (int i = 0; i < inSize; i++) {
            int offset = i * outSize + j;
            output[j] += input[i] * W[offset];
        }
    }
    for (int j = 0; j < outSize; j++) {
        output[j] += b[j];
    }
}

// __constant__ float device_input[INPUT_SIZE];
// __constant__ float device_b[OUTPUT_SIZE];
__global__ void FC(float *output, float *input, float *W, float *b, int inSize, int outSize) {
// __global__ void FC(float *output, float *W, int inSize, int outSize) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if ((j >= outSize)) {
        return;
    }

    float sum = 0.0;
    for (int i = 0; i < inSize; i++) {
        int offset = i * outSize + j;
        sum += input[i] * W[offset];
        // sum += device_input[i] * W[offset];
    }

    output[j] = sum + b[j];
    // output[j] = sum + device_b[j];
}

int main() {
    // alloc memory host-side
    float *h_in = (float *) malloc (INPUT_SIZE * sizeof(float));
    float *h_W = (float *) malloc (INPUT_SIZE * OUTPUT_SIZE * sizeof(float));
    float *h_b = (float *) malloc (OUTPUT_SIZE * sizeof(float));
    float *h_out = (float *) malloc (OUTPUT_SIZE * sizeof(float)); // host result
    float *h_dout = (float *) malloc (OUTPUT_SIZE * sizeof(float)); // host result

    cudaHostRegister(h_in, INPUT_SIZE * sizeof(float), 0);
    cudaHostRegister(h_W, INPUT_SIZE * OUTPUT_SIZE * sizeof(float), 0);
    cudaHostRegister(h_b, OUTPUT_SIZE * sizeof(float), 0);

    initArray(h_in, INPUT_SIZE);
    initWeights(h_W, INPUT_SIZE, OUTPUT_SIZE);
    initArray(h_b, OUTPUT_SIZE);

    gpuErrchk(cudaDeviceReset());
    
    // alloc memory device side
    float *d_in;
    float *d_W;
    float *d_b;
    float *d_out;
    gpuErrchk( cudaMalloc( (void **) &d_in, INPUT_SIZE * sizeof(float) ) );
    gpuErrchk( cudaMalloc( (void **) &d_W, INPUT_SIZE * OUTPUT_SIZE * sizeof(float) ) );
    gpuErrchk( cudaMalloc( (void **) &d_b, OUTPUT_SIZE * sizeof(float) ) );
    gpuErrchk( cudaMalloc( (void **) &d_out, OUTPUT_SIZE * sizeof(float) ) );

    // transfer data to device
    gpuErrchk( cudaMemcpy(d_in, h_in, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_W, h_W, INPUT_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_b, h_b, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice) );
    // gpuErrchk( cudaMemcpyToSymbol(device_input, h_in, INPUT_SIZE * sizeof(float)) );
    // gpuErrchk( cudaMemcpyToSymbol(device_b, h_b, OUTPUT_SIZE * sizeof(float)) );

    double start_time = getTimeStamp();

	// invoke kernel
    dim3 block(32, 32); // configure
    dim3 grid((OUTPUT_SIZE+block.x-1)/block.x, (INPUT_SIZE+block.y-1)/block.y);
    FC<<<64, 1024>>>(d_out, d_in, d_W, d_b, INPUT_SIZE, OUTPUT_SIZE);
    // FC<<<grid, block>>>(d_out, d_W, INPUT_SIZE, OUTPUT_SIZE);
    gpuErrchk( cudaDeviceSynchronize() );

    // copy data back
    gpuErrchk( cudaMemcpy(h_dout, d_out, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost) );

    double end_time = getTimeStamp();
    int total_time_ms = (int) ceil ((end_time-start_time)*1000);

    host_FC(h_out, h_in, h_W, h_b, INPUT_SIZE, OUTPUT_SIZE);
    // printArray(h_in, INPUT_SIZE);
    // printf("\n");
    // printWeights(h_W, INPUT_SIZE, OUTPUT_SIZE);
    // printf("\n");
    // printArray(h_b, OUTPUT_SIZE);
    // printf("\n");
    // printArray(h_out, OUTPUT_SIZE);
    // printf("\n");
    // printArray(h_dout, OUTPUT_SIZE);
    // printf("\n");

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (h_out[i] != h_dout[i]) {
            printf("Error: CPU result and GPU result mismatch at offset: %d.\n", i);
            return 0;
        }
    }

    printf("%d\n", total_time_ms);

    cudaHostUnregister(h_in);
    cudaHostUnregister(h_W);
    cudaHostUnregister(h_b);

    // free gpu resources
    gpuErrchk( cudaFree(d_in) );
    gpuErrchk( cudaFree(d_W) );
    gpuErrchk( cudaFree(d_b) );
    gpuErrchk( cudaFree(d_out) );
    gpuErrchk( cudaDeviceReset() );

    return 0;

}

