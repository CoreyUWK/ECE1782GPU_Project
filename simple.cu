/*
* ECE1782 - Fall 2021 - Project
* nvcc -arch sm_52 -Xptxas="-v" final.cu

Input: 56x100
Convolution: 8x8 => output 64 channels | Stride=1, Padding=Same | ReLu
Max Pool: 2x2 | Stride=1, Padding=Same
Convolution: 4x4 => output 64 channels | Stride=1, Padding=Same | ReLu 
Max Pool: 2x2 | Stride=1, Padding=Same
Convolution: 2x2 => output 64 channels | Stride=1, Padding=Same | ReLu
Flatten: to 256 nodes
Full Connected Linear: in=256, out=3 | ReLu and softmax 
*/
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include "cnn_weights.cu"
//#define PRINTDATA 1

#define INPUT_WIDTH 56
#define INPUT_HEIGHT 100

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

__constant__ float device_cov1_b;
__constant__ float device_cov1_filter1[COV1_FILTER_N][COV1_FILTER_N];
__constant__ float device_cov1_filter2[COV1_FILTER_N][COV1_FILTER_N];
__constant__ float device_cov1_filter3[COV1_FILTER_N][COV1_FILTER_N];
__constant__ float device_cov1_filter4[COV1_FILTER_N][COV1_FILTER_N];
__constant__ float device_cov1_filter5[COV1_FILTER_N][COV1_FILTER_N];
__constant__ float device_cov1_filter6[COV1_FILTER_N][COV1_FILTER_N];
__constant__ float device_cov1_filter7[COV1_FILTER_N][COV1_FILTER_N];
__constant__ float device_cov1_filter8[COV1_FILTER_N][COV1_FILTER_N];


/*
https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html

W = input dimension
F = filter dimension
P = padding amount
S = strid amount

OuputSize = (W - F + 2P)/S + 1

In our case: Padding=Same and Stride=1 => so will pad enough so that output size equals input size
out_width = (56 - F + 2P)/1 + 1 = 57 - F + 2P
56 = 57 - F + 2P => (F - 1)/2 = P

out_height = (100 - F + 2P)/1 + 1 = 101 - F + 2P
100 = 101 - F + 2P => (F - 1)/2 = P

For F = 8x8
P = (8 - 1)/2 = 3.5 = 

*/
__device__ void device_CNN(float *in, float *out, float *filter, int filterSize, int padding) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int offset = y * INPUT_WIDTH + x; 

    float conv = 0;
    for (int i = 0; i < filterSize; ++i) {
        for (int j = 0; j < filterSize; ++j) { 
            cov += in[offset + i * INPUT_WIDTH + j] * cov1_filter[i * filterSize + j];
        }
    }

    out[]
}

// Generate Input Data
void initData(float *in, int width, int height, int padding) {
    int offset;
    const float magicNum = 1.1;

    for (int i = padding; i < height - padding; ++i) {
        for (int j = padding; j < width - padding; ++j) {
            offset = i * width + j;
            // printf("(%d,%d)=%d ", i, j, offset);
            in[offset] = ((i+j) % 10) * magicNum; //TODO: Update value to be accurate
        }
    }
}

void Print2D(float *m, int width, int height) {
    for (int i = 0, row=0; i < height; ++i, row += width) { // Row
        for (int j = 0; j < width; ++j) { // Col
            printf("%.6f\t", m[row + j]);
        }
        printf("\n");
    }
}

int getPadding(int filter) {
    return filter / 2;
}

int allocInput(int width, int height, int padding, float **out) {
    size_t input_num_elements = (width + padding) * (height + padding);
    size_t bytes = input_num_elements * sizeof(float);

    *out = (float *)malloc(bytes);
    if (NULL == *out) {
        return -1;
    }

    initData(*out, width + padding, height + padding, padding);
    
    return bytes;
}

void setUpCNNFilters() {
    cudaMemcpyToSymbol(device_cov1_b, host_cov1_b, sizeof(float));
    cudaMemcpyToSymbol(device_cov1_filter1, host_cov1_filter1, COV1_FILTER_N*COV1_FILTER_N*sizeof(float));
    cudaMemcpyToSymbol(device_cov1_filter2, host_cov1_filter2, COV1_FILTER_N*COV1_FILTER_N*sizeof(float));
    cudaMemcpyToSymbol(device_cov1_filter3, host_cov1_filter3, COV1_FILTER_N*COV1_FILTER_N*sizeof(float));
    cudaMemcpyToSymbol(device_cov1_filter4, host_cov1_filter4, COV1_FILTER_N*COV1_FILTER_N*sizeof(float));
    cudaMemcpyToSymbol(device_cov1_filter5, host_cov1_filter5, COV1_FILTER_N*COV1_FILTER_N*sizeof(float));
    cudaMemcpyToSymbol(device_cov1_filter6, host_cov1_filter6, COV1_FILTER_N*COV1_FILTER_N*sizeof(float));
    cudaMemcpyToSymbol(device_cov1_filter7, host_cov1_filter7, COV1_FILTER_N*COV1_FILTER_N*sizeof(float));
    cudaMemcpyToSymbol(device_cov1_filter8, host_cov1_filter8, COV1_FILTER_N*COV1_FILTER_N*sizeof(float));

}

int main( int argc, char *argv[])
{    
    // Allocate intial input to CNN with padding
    float *h_input = NULL;
    int padding = getPadding(COV1_FILTER_N);
    int bytes = allocInput(INPUT_WIDTH, INPUT_HEIGHT, padding, &h_input);
    if (bytes == -1) {
        printf("Error: Failed to allocte host memory for input");
        return 1;
    }
    
#ifdef PRINTDATA
    printf("Input:\n");
    Print2D(h_input, INPUT_WIDTH + padding, INPUT_HEIGHT + padding);
#endif
    
    gpuErrchk(cudaDeviceReset());

    // Pinning host memory so pages are not paged to disk for DMA to work
    gpuErrchk(cudaHostRegister(h_input, bytes, 0));

    dim3 block((INPUT_WIDTH + padding < 32) ? INPUT_WIDTH + padding : 32, (INPUT_HEIGHT + padding < 32) ? INPUT_HEIGHT + padding : 32); 
    dim3 grid( (INPUT_WIDTH + padding + block.x-1) / block.x, 
               (INPUT_HEIGHT + padding + block.y-1) / block.y);

    const int CNN_Layers = 5;
    const int Cov_Channels = 64;
    const int NumMatrixPerCNN = 1 + Cov_Channels*5; // Number of input sized matrix used in layer
    float *d_input; 

//================= Timing Begins ========================
    double start_time=getTimeStamp();

    setUpCNNFilters();

    // Get Memory for Data Input
    gpuErrchk(cudaMalloc((void **)&d_input, bytes));
    // Copy over input
    gpuErrchk(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));



    for (int layer = 0; layer < CNN_Layers; ++layer) {
        

        for (int i = 0, offset=0; i < Cov_Channels; ++i, offset += size2D) {
            // Get Memory for Layer channel process
                gpuErrchk(cudaMalloc((void **)&d_input[i], bytes));

            // Wait for previous layer to complete for channel
            int prevI = i-1;
            int prevStreamId = (i % (NumStream - 1)) + 1;
            //printf("%d:%d ", i, prevStreamId);
            gpuErrchk(cudaStreamWaitEvent(stream[prevStreamId], events[i]));
    
            // Get Memory for A
            if (queueA[endA] == -1) {
                gpuErrchk(cudaMalloc((void **)&d_a[prevI], size2DBytes));
                lookupA[prevI] = prevI;
            }
            else {
                lookupA[prevI] = queueA[endA];
                queueA[endA] = -1;
                if (0 < endA) --endA;
            }

            // Process Prev Slice
#ifdef SHAREDMEMORY
            kernel_device<<<grid, block, sharedMem, stream[prevStreamId]>>>(d_b[lookupB[prevI]], d_a[lookupA[prevI]], 
                    (1 < i) ? d_b[lookupB[i-2]] : NULL, d_b[lookupB[i]], n); //, prevI
#else
            kernel_device<<<grid, block, 0, stream[prevStreamId]>>>(d_b[lookupB[prevI]], d_a[lookupA[prevI]], 
                    (1 < i) ? d_b[lookupB[i-2]] : NULL, d_b[lookupB[i]], n);
#endif

            // Copy back Processed Slice
            gpuErrchk(cudaMemcpyAsync(&h_da[offset - size2D], d_a[lookupA[prevI]], size2DBytes, cudaMemcpyDeviceToHost, 
                stream[prevStreamId]));

            // Release memory when done
            params[i].i = i; params[i].prevI = prevI;
            cudaStreamAddCallback(stream[prevStreamId], callbackProcessFinished, (void*)&params[i], 0);
        }
    }

    for (int i = 1; i < NumStream; ++i) {
        gpuErrchk(cudaStreamSynchronize(stream[i]));
        gpuErrchk(cudaStreamDestroy(stream[i]));
    }
    //gpuErrchk(cudaDeviceSynchronize());

    //TODO: Data transfer from device to host
    #ifdef EXTR_TIMING
    double cpy_from_time = getTimeStamp();
    #endif

    double end_time=getTimeStamp();
//================= Timing Ends ========================    
    int total_time_ms = (int)ceil((end_time-start_time)*1000);
    #ifdef EXTR_TIMING
    int alloc_time_ms = (int)ceil((cpy_to_time-alloc_time)*1000);
    int cpy_to_time_ms = (int)ceil((kernel_time-cpy_to_time)*1000);
    int kernel_time_ms = (int)ceil((cpy_from_time-kernel_time)*1000);
    int cpy_from_time_ms = (int)ceil((end_time-cpy_from_time)*1000);
    #endif
    
    //TODO: free allocated resources
    gpuErrchk(cudaHostUnregister(h_b));
    gpuErrchk(cudaHostUnregister(h_da));

    for (int i = 0; i < n; ++i) {
        gpuErrchk(cudaEventDestroy(events[i]));
    }

    //TODO: Computing the sum (not included in timing)
    double sumDevice = calculateSum(h_da, n);
#ifdef PRINTDATA
    printf("Output Data GPU a:\n");
    PrintMatrix(h_da, n);
#endif


#ifdef PRINTDATA
    printf("Output Data Host a:\n");
    PrintMatrix(h_ha, n);
#endif

#ifdef ASSERTRESULTS
    if (n == 100) printf("Exp: 17861235.145611\n");
    else if (n == 200) printf("Exp: 145351171.783584\n");
    else if (n == 300) printf("Exp: 493349760.596508\n");
    else if (n == 400) printf("Exp: 1172737007.706970\n");
    else if (n == 500) printf("Exp: 2294392919.237560\n");
    else if (n == 600) printf("Exp: 3969197501.310867\n");
    else if (n == 700) printf("Exp: 6308030765.186127\n");
    else if (n == 800) printf("Exp: 9421772714.654696\n");
#endif
    
    printf("%lf %d\n", sumDevice, total_time_ms);
#ifdef EXTR_TIMING
    printf("%d %d %d %d\n", alloc_time_ms, cpy_to_time_ms, kernel_time_ms,
        cpy_from_time_ms);
#endif

    for (int i = 0; i < n; ++i) {
        if (lookupB[i] == i) {
            gpuErrchk(cudaFree(d_b[i]));
        }
        if (lookupA[i] == i) {
            gpuErrchk(cudaFree(d_a[i]));
        }
    }

    gpuErrchk(cudaDeviceReset());
*/
    free(h_input);

    return 0;
}