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

#define INPUT_WIDTH 100
#define INPUT_HEIGHT 56

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
__device__ void device_CNN(float *inCh, float *outCh, float *filter, int filterSize) {
    // Position relative to global memory of 2D matrix
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= INPUT_WIDTH || y >= INPUT_HEIGHT) {
        return;
    }

    const int offset_GM = y * INPUT_WIDTH + x;

    // P=max(F−S,0)
    const int totalPaddingHeight = filterSize - 1;
    const int totalPaddingWidth = filterSize - 1;
    const int topPadding = totalPaddingHeight / 2;
    const int leftPadding = totalPaddingWidth / 2;
    //const int bottomPadding = totalPaddingHeight - topPadding;
    //const int rightPadding = totalPaddingWidth - leftPadding;
    //printf("%d %d %d %d", topPadding, leftPadding, bottomPadding, rightPadding);
    /*
    1 2 3 4 5 6  filter=4x4  => TotPadH=3 => topPad=1 
    7 8 9 1 2 3                 TotPadW=3    leftPad=1
    4 5 6 7 8 9                              botPad=2
    1 2 3 4 5 6                              rightPad=2

    0 0 0 0 0 0 0 0 0
    0 1 2 3 4 5 6 0 0
    0 7 8 9 1 2 3 0 0
    0 4 5 6 7 8 9 0 0 
    0 1 2 3 4 5 6 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0

    for (0,0)= "1" (-1,-1)+(-1,0)+(-1,1)+(-1,2) +
               "2" (0,-1)+(0,0)+(0,1)+(0,2) +
               "3" (1,-1)+(1,0)+(1,1)+(1,2) +
               "4" (2,-1)+(2,0)+(2,1)+(2,2)

    for (3,5)= "1" (3-1,5-1)+(3-1,5)+(3-1,5+1)+(3-1,5+2) +
               "2" (3,5-1)+(3,5)+(3,5+1)+(3,5+2) +
               "3" (3+1,5-1)+(3+1,5)+(3+1,5+1)+(3+1,5+2) +
               "4" (3+2,5-1)+(3+2,5)+(3+2,5+1)+(3+2,5+2)

             = "1" (2,4)+(2,5)+(2,6)+(2,7) +
               "2" (3,4)+(3,5)+(3,6)+(3,7) +
               "3" (4,4)+(4,5)+(4,6)+(4,7) + -> all zero (pad)
               "4" (5,4)+(5,5)+(5,6)+(5,7)   -> all zero (pad)
    */

    //TODO: reduce repeated computations by storing
    //Maybe make loop condition handle outside matrix area instead of continue
    int cnnOffset = offset_GM - topPadding*INPUT_WIDTH - leftPadding;
    float conv = 0;
    for (int i = 0; i < filterSize; ++i) {
        if (y - topPadding + i < 0) continue;
        if (y - topPadding + i >= INPUT_HEIGHT) {
            //printf("%d %d %d\n", y, topPadding, i);
            break;
        }
        for (int j = 0; j < filterSize; ++j) {
            int offset = cnnOffset + i * INPUT_WIDTH + j;
            if (x - leftPadding + j < 0) continue;
            if (x - leftPadding + j >= INPUT_WIDTH) break;
            conv += inCh[cnnOffset + i * INPUT_WIDTH + j] * filter[i * filterSize + j];
            //printf("%d %d\n", i, j);
        }
    }

    outCh[offset_GM] = conv;
}

/* if can put input into shared memory then can do inplace on input
__device__ void device_CNN_SHMEM(float *inCh, float *filter, int filterSize) {
    
    
    // Position relative to global memory of 2D matrix
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int offset_GM = y * INPUT_WIDTH + x;

    // P=max(F−S,0)
    const int totalPaddingHeight = filterSize - 1;
    const int totalPaddingWidth = filterSize - 1;
    const int topPadding = totalPaddingHeight / 2;
    const int leftPadding = totalPaddingWidth / 2;
    const int bottomPadding = totalPaddingHeight - topPadding;
    const int rightPadding = totalPaddingWidth - leftPadding;
    //printf("%d %d %d %d", topPadding, leftPadding, bottomPadding, rightPadding);
    /*
    1 2 3 4 5 6  filter=4x4  => TotPadH=3 => topPad=1 
    7 8 9 1 2 3                 TotPadW=3    leftPad=1
    4 5 6 7 8 9                              botPad=2
    1 2 3 4 5 6                              rightPad=2

    0 0 0 0 0 0 0 0 0
    0 1 2 3 4 5 6 0 0
    0 7 8 9 1 2 3 0 0
    0 4 5 6 7 8 9 0 0 
    0 1 2 3 4 5 6 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0

    for (0,0)= "1" (-1,-1)+(-1,0)+(-1,1)+(-1,2) +
               "2" (0,-1)+(0,0)+(0,1)+(0,2) +
               "3" (1,-1)+(1,0)+(1,1)+(1,2) +
               "4" (2,-1)+(2,0)+(2,1)+(2,2)

    for (3,5)= "1" (3-1,5-1)+(3-1,5)+(3-1,5+1)+(3-1,5+2) +
               "2" (3,5-1)+(3,5)+(3,5+1)+(3,5+2) +
               "3" (3+1,5-1)+(3+1,5)+(3+1,5+1)+(3+1,5+2) +
               "4" (3+2,5-1)+(3+2,5)+(3+2,5+1)+(3+2,5+2)

             = "1" (2,4)+(2,5)+(2,6)+(2,7) +
               "2" (3,4)+(3,5)+(3,6)+(3,7) +
               "3" (4,4)+(4,5)+(4,6)+(4,7) + -> all zero (pad)
               "4" (5,4)+(5,5)+(5,6)+(5,7)   -> all zero (pad)
    */

    //TODO: reduce repeated computations by storing
    //Maybe make loop condition handle outside matrix area instead of continue
   /* int cnnOffset = offset_GM - topPadding*INPUT_WIDTH - leftPadding;
    float conv = 0;
    for (int i = 0; i < filterSize; ++i) {
        for (int j = 0; j < filterSize; ++j) {
            int offset = cnnOffset + i * INPUT_WIDTH + j;
            if (x - leftPadding + j < 0 || x - leftPadding + j >= INPUT_WIDTH || 
                y - topPadding + i < 0 || y - topPadding + i >= INPUT_HEIGHT) 
                continue;
            conv += inCh[cnnOffset + i * INPUT_WIDTH + j] * filter[i * filterSize + j];
            //printf("%d %d\n", i, j);
        }
    }

    outCh[offset_GM] = conv;
}*/


__global__ void kernel(float *inCh, float *outCh, float *filter, int filterSize) {
    device_CNN(inCh, outCh, filter, filterSize);
}

// Generate Input Data
void initData(float *in, int width, int height, int padding) {
    int offset;
    const float magicNum = 1.1;

    for (int i = padding; i < height - padding; ++i) {
        for (int j = padding; j < width - padding; ++j) {
            offset = i * width + j;
            // printf("(%d,%d)=%d ", i, j, offset);
            in[offset] = 1.0; //((i+j) % 10) * magicNum; //TODO: Update value to be accurate
        }
    }
}

void Print2D(float *m, int width, int height) {
    for (int i = 0, row=0; i < height; ++i, row += width) { // Row
        printf("%d:\t", i);
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
    gpuErrchk(cudaMemcpyToSymbol(device_cov1_b, &host_cov1_b, sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(device_cov1_filter1, host_cov1_filter1, COV1_FILTER_N*COV1_FILTER_N*sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(device_cov1_filter2, host_cov1_filter2, COV1_FILTER_N*COV1_FILTER_N*sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(device_cov1_filter3, host_cov1_filter3, COV1_FILTER_N*COV1_FILTER_N*sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(device_cov1_filter4, host_cov1_filter4, COV1_FILTER_N*COV1_FILTER_N*sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(device_cov1_filter5, host_cov1_filter5, COV1_FILTER_N*COV1_FILTER_N*sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(device_cov1_filter6, host_cov1_filter6, COV1_FILTER_N*COV1_FILTER_N*sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(device_cov1_filter7, host_cov1_filter7, COV1_FILTER_N*COV1_FILTER_N*sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(device_cov1_filter8, host_cov1_filter8, COV1_FILTER_N*COV1_FILTER_N*sizeof(float)));
}

int main( int argc, char *argv[])
{    
    // Allocate intial input to CNN with padding
    float *h_input = NULL;
    int padding = 0; //getPadding(3);
    int bytes = allocInput(INPUT_WIDTH, INPUT_HEIGHT, padding, &h_input);
    if (bytes == -1) {
        printf("Error: Failed to allocte host memory for input");
        return 1;
    }

    float *h_output = NULL;
    bytes = allocInput(INPUT_WIDTH, INPUT_HEIGHT, padding, &h_output);
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
    gpuErrchk(cudaHostRegister(h_output, bytes, 0));

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

    float *d_output;
    gpuErrchk(cudaMalloc((void **)&d_output, bytes));
    float *filterAddr;
    gpuErrchk(cudaGetSymbolAddress((void**)&filterAddr, device_cov1_filter1));
    kernel<<<grid, block>>>(d_input, d_output, filterAddr, COV1_FILTER_N);
    gpuErrchk(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    /*for (int layer = 0; layer < CNN_Layers; ++layer) {
        

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
    }*/
    gpuErrchk(cudaDeviceSynchronize());

    double end_time=getTimeStamp();
//================= Timing Ends ========================    
    int total_time_ms = (int)ceil((end_time-start_time)*1000);
    
    //TODO: free allocated resources
    gpuErrchk(cudaHostUnregister(h_input));
    gpuErrchk(cudaHostUnregister(h_output));

    /*for (int i = 0; i < n; ++i) {
        gpuErrchk(cudaEventDestroy(events[i]));
    }*/

#ifdef PRINTDATA
    printf("Output Data GPU a:\n");
    Print2D(h_output, INPUT_WIDTH + padding, INPUT_HEIGHT + padding);
#endif
   
    printf("Time: %d\n", total_time_ms);

    /*for (int i = 0; i < n; ++i) {
        if (lookupB[i] == i) {
            gpuErrchk(cudaFree(d_b[i]));
        }
        if (lookupA[i] == i) {
            gpuErrchk(cudaFree(d_a[i]));
        }
    }*/
    gpuErrchk(cudaFree(d_input));
    gpuErrchk(cudaFree(d_output));

    gpuErrchk(cudaDeviceReset());

    free(h_input);
    free(h_output);

    return 0;
}