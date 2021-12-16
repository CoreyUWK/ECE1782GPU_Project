/*
* ECE1782 - Fall 2021 - Project
* nvcc -arch sm_52 -Xptxas="-v" final.cu
nvcc simple.cu -ccbin g++ --std=c++11 -lpthread

Input: 56x100
Convolution: 8x8 => output 64 channels | Stride=1, Padding=Same | ReLu
Max Pool: 2x2 | Stride=1, Padding=Same
Convolution: 4x4 => output 64 channels | Stride=1, Padding=Same | ReLu 
Max Pool: 2x2 | Stride=1, Padding=Same
Convolution: 2x2 => output 64 channels | Stride=1, Padding=Same | ReLu
Flatten: to 256 nodes
Full Connected Linear: in=256, out=3 | ReLu and softmax

Convolution Design Ideas:
1.1) Store filters in constant memory
- issue: all filters do not fit in constant memory
( ((8*8)*64 + 64) + ((4*4)*64*64 + 64) + ((2*2)*64*64) + 64) ) *4byte = 344,832bytes= 344.8KB > 64KB
1.2) Need alternative to constant memory 
-> maybe pass to kernels GMem and then copy to shared memory with avaliable threads

2) Convolution function takes input matrix and produces output matrix
2.1) if injecting padding then will need to copy input matrix with padding
all 56x100 will become + max padding sizes, and kernel will have to know where to start from based on necessary padding for filter size
2.2) if not injecting padding, handle with if checks.
However, threads will not be indexed top left of convolution with filter
- actually can't do inline since can only sync threads per block, not entire 2d matrix, so will have sync issues (data not correct result)
- unless each thread copy filter area into shared memory or registers, process it, and then sync and write out to original input

3) for multi input channel, perform all filter convolution in input per thread, then write out to ouput (inline or not)

*/
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <mutex>
#include <algorithm>
#include "../src/cnn_weights.h"
#include "../src/utils.cu"

//#define PRINTDATA 1
//#define EnableLock 1

//#define SHMEM 1
//#define DebugSHMEM 1
//#define DebugSHMEM_Data 1

#define INPUT_WIDTH 100//2048//100
#define INPUT_HEIGHT 56//2048//56

#define NUM_STREAM 64+64

// MAXPOOL Config
#define PAD_VALUE -INFINITY
#define MAX_TOL 1e-3
#define POOL_SIZE 2
#define STRIDE POOL_SIZE

// Currently a thread per pooling, but thread no reading coalesed
// could read coalesed by copying to shared memory and then reorder in shared memory linearly
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


__global__ void device_CNN(int in_cols, int in_rows, float *inCh, float *outCh, float b, float *filter, int filterSize,
    bool isSingle, bool isFirst, bool isLast,
    int totalPaddingHeight, int totalPaddingWidth, int topPadding, int leftPadding) {
    // Position relative to global memory of 2D matrix
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= in_cols || y >= in_rows) {
        return;
    }
    
    const int offset_GM = y * in_cols + x;

    //TODO: reduce repeated computations by storing
    //Maybe make loop condition handle outside matrix area instead of continue
    int topCheck = y - topPadding;
    int leftCheck = x - leftPadding;
    int cnnOffset = offset_GM - topPadding*in_cols - leftPadding;
    float conv = 0;
    for (int i = 0, offset=cnnOffset, filterOffset=0, topChecki=topCheck; 
        i < filterSize; ++i, offset += in_cols, filterOffset += filterSize, ++topChecki) {
        if (topChecki < 0) continue;
        else if (topChecki >= in_rows) break;
        for (int j = 0; j < filterSize; ++j) {
            int leftCheckj = leftCheck + j;
            if (leftCheckj < 0) continue;
            if (leftCheckj >= in_cols) break;
            conv += inCh[offset + j] * filter[filterOffset + j];
        }
    }

    if (isSingle) {
        outCh[offset_GM] = relu(conv + b);
    }
    else if (isFirst) {
        outCh[offset_GM] = conv;
    }
    else if (isLast) {
        conv += outCh[offset_GM] + b;
        outCh[offset_GM] = relu(conv);
    }
    else {
        outCh[offset_GM] += conv;
    }
}


float* allocHostBlockHelper(std::vector<float*> &h_freeBlocks, std::mutex &h_freeBlocksMutex, int bytes) {
    float *mem = NULL;

#ifdef EnableLock
    h_freeBlocksMutex.lock();
#endif
    if (!h_freeBlocks.empty()) {
        mem = h_freeBlocks.back();
        h_freeBlocks.pop_back();
    }
#ifdef EnableLock
    h_freeBlocksMutex.unlock();
#endif
    if (mem == NULL) {
        mem = allocHostBlock(bytes);
    }

    return mem;
}

float* allocDeviceBlockHelper(std::vector<float*> &d_freeBlocks, std::mutex &d_freeBlocksMutex, int bytes) {
    float *mem = NULL;

#ifdef EnableLock
    d_freeBlocksMutex.lock();
#endif
    if (!d_freeBlocks.empty()) {
        mem = d_freeBlocks.back();
        d_freeBlocks.pop_back();
    }
#ifdef EnableLock
    d_freeBlocksMutex.unlock();
#endif
    if (mem == NULL) {
        mem = allocDeviceBlock(bytes);
    }

    return mem;
}

void setUpCNNFilters(float *host_cov_b, float * host_cov_filter, cudaStream_t stream) {
    gpuErrchk( cudaMemcpyToSymbolAsync(device_cov1_b, host_cov_b, COV1_FILTER_OUT_CH*sizeof(float), 
        0, cudaMemcpyHostToDevice, stream) );

    gpuErrchk( cudaMemcpyToSymbolAsync(device_cov1_filter, host_cov_filter, 
        COV1_FILTER_IN_CH * COV1_FILTER_OUT_CH * COV1_FILTER_N*COV1_FILTER_N*sizeof(float), 
        0, cudaMemcpyHostToDevice, stream));
}

void setupFilterCh(int ch, float *host_cov_b, float *host_cov_filter, cudaStream_t &stream) {
    int size = sizeof(float);
    /*gpuErrchk( cudaMemcpyToSymbolAsync(device_cov1_b, host_cov_b, size, 
        ch*size, cudaMemcpyHostToDevice, stream) );*/

    size = COV1_FILTER_N*COV1_FILTER_N*sizeof(float);
    gpuErrchk( cudaMemcpyToSymbolAsync(device_cov1_filter, host_cov_filter, size, 
        ch * size, cudaMemcpyHostToDevice, stream) );
}

void layer_cov(int in_cols, int in_rows, cudaStream_t *streams, cudaEvent_t &event,
    float *h_input, float *d_input, float **d_output) {

    int bytes = in_cols*in_rows*sizeof(float);

    dim3 block((in_cols < 32) ? in_cols : 32, 
                (in_rows < 32) ? in_rows : 32); 
    dim3 grid( (in_cols + block.x-1) / block.x, 
               (in_rows + block.y-1) / block.y);

    int totalPaddingHeight, totalPaddingWidth;
    int topPadding, leftPadding, bottomPadding, rightPadding;
    getConvPadding(COV1_FILTER_N, totalPaddingHeight, 
        totalPaddingWidth, topPadding, leftPadding,
        bottomPadding, rightPadding);

    float *filterAddr;
    gpuErrchk(cudaGetSymbolAddress((void**)&filterAddr, device_cov1_filter));

    for (int i=0; i < COV1_FILTER_OUT_CH; ++i) {
        // Copy over input
        if (i == 0) {
            // Performing async for implementation of multiple CNNs running in parallel for server
    
            // Copy over input
            gpuErrchk(cudaMemcpyAsync(d_input, h_input, bytes, cudaMemcpyHostToDevice, streams[0]));
            gpuErrchk(cudaEventRecord(event, streams[0]));
        }
        // Every stream needs to wait for input
        if (i > 0 && i < 64) // input cpy and first filter run on same stream so skip on first stream 
            gpuErrchk(cudaStreamWaitEvent(streams[i], event, 0));

        // Setup all filters and b
        setupFilterCh(i, &host_cov1_b[i], &host_cov1_filter[0][i][0][0], streams[i]);

        // Get output memory
        if (d_output[i] == NULL) {
            gpuErrchk(cudaMalloc((void **)&d_output[i], bytes));
        }
        device_CNN<<<grid, block, 0, streams[i]>>>(in_cols, in_rows, d_input, d_output[i], host_cov1_b[i], filterAddr + i*COV1_FILTER_N*COV1_FILTER_N, COV1_FILTER_N,
            true, false, false, 
            totalPaddingHeight, totalPaddingWidth, topPadding, leftPadding);
    }

    // TODO: fix not needing this
    // need as when function ends stack cleared, so params dropped but then accessed by callback still
//    gpuErrchk(cudaDeviceSynchronize());
}

void layer_maxPool(int in_cols, int in_rows, float **d_input, 
    int &out_cols, int &out_rows, float **d_output, cudaStream_t *streams) {
    
    out_cols = (in_cols - 1) / STRIDE + 1;
    out_rows = (in_rows - 1) / STRIDE + 1;
    //printf("Out Size: %d,%d\n", out_rows, out_cols);

    int out_elements = out_rows * out_cols;
    int out_bytes = out_elements * sizeof(float);

    dim3 block((out_cols < 32) ? out_cols : 32, (out_rows < 32) ? out_rows : 32); 
    dim3 grid( (out_cols + block.x-1) / block.x, 
               (out_rows + block.y-1) / block.y);

    // Get output memory
    for (int ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaMalloc((void **)&d_output[ch], out_bytes));
        max_pool_2d<<<grid, block, 0, streams[ch]>>>(d_input[ch], in_rows, in_cols, d_output[ch], out_rows, out_cols);
    }

    //gpuErrchk(cudaDeviceSynchronize());
    /*for (int ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaStreamSynchronize(streams[ch]));
        gpuErrchk(cudaStreamDestroy(streams[ch]));
    }*/
}

void layer_covMulti(int in_cols, int in_rows, cudaStream_t *streams, cudaEvent_t *events,
    float **d_input, float **d_output, int filterSize) {

    int bytes = in_cols*in_rows*sizeof(float);

    dim3 block((in_cols < 32) ? in_cols : 32, 
                (in_rows < 32) ? in_rows : 32); 
    dim3 grid( (in_cols + block.x-1) / block.x, 
               (in_rows + block.y-1) / block.y);

    int totalPaddingHeight, totalPaddingWidth;
    int topPadding, leftPadding, bottomPadding, rightPadding;
    getConvPadding(filterSize, totalPaddingHeight, 
        totalPaddingWidth, topPadding, leftPadding,
        bottomPadding, rightPadding);

    float *filterAddr;
    gpuErrchk(cudaGetSymbolAddress((void**)&filterAddr, device_cov1_filter));

    for (int inCh=0; inCh < COV2_FILTER_IN_CH; ++inCh) {
        gpuErrchk(cudaEventRecord(events[inCh], streams[inCh]));

        for (int outCh=0; outCh < COV2_FILTER_OUT_CH; ++outCh) {
            // Every stream needs to wait for input
            gpuErrchk(cudaStreamWaitEvent(streams[COV2_FILTER_IN_CH + outCh], events[inCh], 0));

            // Setup all filters and b
            setupFilterCh(outCh, &host_cov1_b[outCh], &host_cov1_filter[0][outCh][0][0], streams[COV2_FILTER_IN_CH+outCh]);

            // Get output memory
            if (inCh == 0) {
                gpuErrchk(cudaMalloc((void **)&d_output[outCh], bytes));
            }
            device_CNN<<<grid, block, 0, streams[COV2_FILTER_IN_CH + outCh]>>>(in_cols, in_rows, d_input[inCh], d_output[outCh], host_cov1_b[outCh], filterAddr + outCh*COV1_FILTER_N*COV1_FILTER_N, filterSize,
                false, (inCh==0)?true:false, (inCh==63)?true:false, 
                totalPaddingHeight, totalPaddingWidth, topPadding, leftPadding);
        }
    }

    // TODO: fix not needing this
    // need as when function ends stack cleared, so params dropped but then accessed by callback still
//    gpuErrchk(cudaDeviceSynchronize());
}


int main( int argc, char *argv[])
{   
    int blockSize = INPUT_HEIGHT*INPUT_WIDTH;
    int bytes = blockSize * sizeof(float);
    
    cudaStream_t streams[NUM_STREAM];

    // Setup filter values
    std::fill(&host_cov1_b[0], &host_cov1_b[0] + COV1_FILTER_OUT_CH, 1.0);
    std::fill(&host_cov1_filter[0][0][0][0], &host_cov1_filter[0][0][0][0] + COV1_FILTER_IN_CH * COV1_FILTER_OUT_CH * COV1_FILTER_N * COV1_FILTER_N, 1.0);

    gpuErrchk(cudaDeviceReset());

    // Allocate intial input to CNN
    float *h_input = allocHostBlock(bytes);
    if (h_input == NULL) {
        printf("Error: Failed to allocte host memory for input");
        return 1;
    }
    float value = 1.0;
    initData(h_input, INPUT_WIDTH, INPUT_HEIGHT, 0, &value);

#ifdef PRINTDATA
    printf("Input:\n");
    Print2D(h_input, INPUT_WIDTH, INPUT_HEIGHT);
#endif

    // Pinning host memory so pages are not paged to disk for DMA to work
    gpuErrchk(cudaHostRegister(h_input, bytes, 0));

    for (int i = NUM_STREAM - 1; i >= 0; --i) {
        gpuErrchk(cudaStreamCreate(&streams[i]));
    }

    cudaEvent_t events[64+64];
    int out_col = INPUT_WIDTH, out_row = INPUT_HEIGHT;

    // Perform First convolution layer
    float *d_in[COV1_FILTER_OUT_CH]; 
    float *d_out[COV1_FILTER_OUT_CH];
    for (int i=0; i < COV1_FILTER_OUT_CH; ++i) {
        d_out[i] = NULL;
        gpuErrchk(cudaEventCreate(&events[i]));
        gpuErrchk(cudaEventCreate(&events[i + 64]));
    }
//================= Timing Begins ========================
    double start_time=getTimeStamp();
    
    float *d_input = allocDeviceBlock(bytes);
    if (d_input == NULL) {
        printf("Error: Failed to allocte host memory for input");
        return 1;
    }
    
    layer_cov(INPUT_WIDTH, INPUT_HEIGHT, streams, events[0],
        h_input, d_input, d_out);

    // Input Not needed anymore by device
    gpuErrchk(cudaHostUnregister(h_input));
    free(h_input);
    gpuErrchk(cudaFree(d_input));

    // Perform first max pooling on the output from the first convolution layer
    std::copy(d_out, d_out + COV1_FILTER_OUT_CH, d_in);
    layer_maxPool(INPUT_WIDTH, INPUT_HEIGHT, d_in, 
        out_col, out_row, d_out, streams);

    // Input not needed anymore by device
    for (int ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaFree(d_in[ch]));
        
    }

    // Perform second convolution layer
    std::copy(d_out, d_out + COV2_FILTER_OUT_CH, d_in);
    layer_covMulti(out_col, out_row, streams, events,
        d_in, d_out, COV2_FILTER_N);

    // Input not needed anymore by device
    for (int ch=0; ch < COV2_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaFree(d_in[ch]));
    }

    // Perform second max pooling on the output from the second convolution layer
    std::copy(d_out, d_out + COV2_FILTER_OUT_CH, d_in);
    layer_maxPool(out_col, out_row, d_in, 
        out_col, out_row, d_out, &streams[64]);

    // Input not needed anymore by device
    for (int ch=0; ch < COV2_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaFree(d_in[ch]));
        std::swap(streams[ch], streams[COV2_FILTER_OUT_CH + ch]);
    }

    // Perform third convolution layer
    std::copy(d_out, d_out + COV3_FILTER_OUT_CH, d_in);
    layer_covMulti(out_col, out_row, streams, &events[64], d_in, d_out, COV3_FILTER_N);

    // Input not needed anymore by device
    for (int ch=0; ch < COV3_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaFree(d_in[ch]));
    }

    // Need to wait for stream to complete copy
    for (int i = 0; i < NUM_STREAM; ++i) {
        gpuErrchk(cudaStreamSynchronize(streams[i]));
        gpuErrchk(cudaStreamDestroy(streams[i]));
    }

    double end_time=getTimeStamp();
//================= Timing Ends ========================    
    int total_time_ms = (int)ceil((end_time-start_time)*1000);
    
#ifdef PRINTDATA
    // Allocate host output to print results for debugging
    float *h_output[COV1_FILTER_OUT_CH];
    bytes = out_col*out_row*sizeof(float);
    for (int ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
        h_output[ch] = allocHostBlock(bytes);
        if (h_output[ch] == NULL) {
            printf("Error: Failed to allocte host memory for output ch %d", ch);
            //TODO: need to clean up allocated upto this point
            return 1;
        }
    
        gpuErrchk(cudaHostRegister(h_output[ch], bytes, 0));
    
        gpuErrchk(cudaMemcpy(h_output[ch], d_out[ch], out_col*out_row*sizeof(float), cudaMemcpyDeviceToHost));

        // Need to wait for stream to complete copy
        gpuErrchk(cudaHostUnregister(h_output[ch]));
        
        printf("Output ch%d:\n", ch);
        Print2D(h_output[ch], out_col, out_row);

        free(h_output[ch]);
    }
#endif

    printf("Total Time: %d\n", total_time_ms);

    //gpuErrchk(cudaDeviceSynchronize());

    // Clean up device blocks
    for (int ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaFree(d_out[ch]));
        gpuErrchk(cudaEventDestroy(events[ch]));
        gpuErrchk(cudaEventDestroy(events[ch + 64]));
    }

    gpuErrchk(cudaDeviceReset());

    return 0;
}
