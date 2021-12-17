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
( ((8*8)*64 + 64) + ((4*4)*64*64 + 64) + ((2*2)*64*64) + 64) ) *4byte =
344,832bytes= 344.8KB > 64KB 1.2) Need alternative to constant memory
-> maybe pass to kernels GMem and then copy to shared memory with avaliable
threads

2) Convolution function takes input matrix and produces output matrix
2.1) if injecting padding then will need to copy input matrix with padding
all 56x100 will become + max padding sizes, and kernel will have to know where
to start from based on necessary padding for filter size 2.2) if not injecting
padding, handle with if checks. However, threads will not be indexed top left of
convolution with filter
- actually can't do inline since can only sync threads per block, not entire 2d
matrix, so will have sync issues (data not correct result)
- unless each thread copy filter area into shared memory or registers, process
it, and then sync and write out to original input

3) for multi input channel, perform all filter convolution in input per thread,
then write out to ouput (inline or not)

*/
#include "../src/alloc_helpers.cu"
#include "../src/cnn_weights.h"
#include "../src/utils.cu"
#include <algorithm>
#include <math.h>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>

//#define PRINTDATA 1
#define EnableLock 1

//#define SHMEM 1
//#define DebugSHMEM 1
//#define DebugSHMEM_Data 1

#define INPUT_WIDTH 100 // 2048//100
#define INPUT_HEIGHT 56 // 2048//56

#define NUM_STREAM 200

/*
https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html

W = input dimension
F = filter dimension
P = padding amount
S = strid amount

OuputSize = (W - F + 2P)/S + 1

In our case: Padding=Same and Stride=1 => so will pad enough so that output size
equals input size out_width = (56 - F + 2P)/1 + 1 = 57 - F + 2P 56 = 57 - F + 2P
=> (F - 1)/2 = P

out_height = (100 - F + 2P)/1 + 1 = 101 - F + 2P
100 = 101 - F + 2P => (F - 1)/2 = P

For F = 8x8
P = (8 - 1)/2 = 3.5 =

*/
__device__ void device_CNN(float *inCh, float *outCh, float b, float *filter,
                           int filterSize, int totalPaddingHeight,
                           int totalPaddingWidth, int topPadding,
                           int leftPadding) {
    // Position relative to global memory of 2D matrix
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= INPUT_WIDTH || y >= INPUT_HEIGHT) {
        return;
    }

    const int offset_GM = y * INPUT_WIDTH + x;

    // P=max(F−S,0)
    // const int totalPaddingHeight = filterSize - 1;
    // const int totalPaddingWidth = filterSize - 1;
    // const int topPadding = totalPaddingHeight / 2;
    // const int leftPadding = totalPaddingWidth / 2;
    // const int bottomPadding = totalPaddingHeight - topPadding;
    // const int rightPadding = totalPaddingWidth - leftPadding;
    // printf("%d %d %d %d", topPadding, leftPadding, bottomPadding,
    // rightPadding);
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

    // TODO: reduce repeated computations by storing
    // Maybe make loop condition handle outside matrix area instead of continue
    int topCheck = y - topPadding;
    int leftCheck = x - leftPadding;
    int cnnOffset = offset_GM - topPadding * INPUT_WIDTH - leftPadding;
    float conv = 0;
    for (int i = 0, offset = cnnOffset, filterOffset = 0; i < filterSize;
         ++i, offset += INPUT_WIDTH, filterOffset += filterSize) {
        int topChecki = topCheck + i;
        if (topChecki < 0)
            continue;
        if (topChecki >= INPUT_HEIGHT) {
            // printf("%d %d %d\n", y, topPadding, i);
            break;
        }
        for (int j = 0; j < filterSize; ++j) {
            int leftCheckj = leftCheck + j;
            if (leftCheckj < 0)
                continue;
            if (leftCheckj >= INPUT_WIDTH)
                break;
            conv += inCh[offset + j] * filter[filterOffset + j];
            // printf("%d %d\n", i, j);
        }
    }
    // printf("%f ", conv);
    outCh[offset_GM] = conv + b;
}

#ifdef SHMEM
// Need to used shared memory to store filters as doesn't fit into constant
// memory
__device__ void device_CNN_SHMEM(float *inCh, float *outCh, float b,
                                 float *filter, int filterSize,
                                 int totalPaddingHeight, int totalPaddingWidth,
                                 int topPadding, int bottomPadding,
                                 int leftPadding, int rightPadding) {

    /* Shared Memory Layout:
    000...000
    000...000
    000111000 <- place block of inCh here, but will have to place padding area
    for block edges handling 000111000 000111000 000111000 000...000 000...000
    */
    extern __shared__ float sData[];
    float *sInCh = sData;

    // Position relative to global memory of 2D matrix
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    const int offset_GM = y * INPUT_WIDTH + x;

    // SHMEM offsets (includes padding)
    int shmemWidth = totalPaddingWidth + blockDim.x;
    int shmemRow = (threadIdx.y + topPadding) * shmemWidth;
    int shmemRow_plus_x = shmemRow + threadIdx.x;
    int shmemRow_pos_padded = shmemRow_plus_x + leftPadding;

    // Initialize shared memory to zero - when block is on edge of grid
    // TODO: try to make more efficent by including in below code if checks
    // As here threads are doing duplicate work
    sInCh[shmemRow_pos_padded - topPadding * shmemWidth - leftPadding] =
#ifdef DebugSHMEM_Data
        -1;
#else
        0;
#endif
    sInCh[shmemRow_pos_padded - topPadding * shmemWidth] =
#ifdef DebugSHMEM_Data
        -1;
#else
        0;
#endif
    sInCh[shmemRow_pos_padded - topPadding * shmemWidth + rightPadding] =
#ifdef DebugSHMEM_Data
        -1;
#else
        0;
#endif
    sInCh[shmemRow_pos_padded + bottomPadding * shmemWidth - leftPadding] =
#ifdef DebugSHMEM_Data
        -1;
#else
        0;
#endif
    sInCh[shmemRow_pos_padded + bottomPadding * shmemWidth] =
#ifdef DebugSHMEM_Data
        -1;
#else
        0;
#endif
    sInCh[shmemRow_pos_padded + bottomPadding * shmemWidth + rightPadding] =
#ifdef DebugSHMEM_Data
        -1;
#else
        0;
#endif
    __syncthreads();

    // Now setup Shared memory with data
    if (threadIdx.y >= topPadding &&
        threadIdx.y < (blockDim.y - bottomPadding)) { // this could be an else
        // Set left-overs on top left corner
        if (x >= 2 * leftPadding &&
            y >=
                2 * topPadding && // Basically not block 0 (but if checking
                                  // blockIdx would have to split this into two)
            threadIdx.y < 3 * topPadding &&
            threadIdx.x < 2 * leftPadding && leftPadding <= threadIdx.x) {
            sInCh[shmemRow_plus_x - leftPadding - 2 * topPadding * shmemWidth] =
#ifdef DebugSHMEM_Data
                6;
#else
                inCh[offset_GM - 2 * leftPadding -
                     2 * topPadding * INPUT_WIDTH];
#endif
        }
        // Set left-overs on bottom left corner
        else if (x >= 2 * leftPadding &&
                 (y >= (INPUT_HEIGHT - 3 * bottomPadding) ||
                  threadIdx.y >= (blockDim.y - 3 * bottomPadding)) &&
                 y < (INPUT_HEIGHT - 2 * bottomPadding) &&
                 leftPadding <= threadIdx.x && threadIdx.x < 2 * leftPadding) {
            sInCh[shmemRow_plus_x - leftPadding +
                  2 * rightPadding * shmemWidth] =
#ifdef DebugSHMEM_Data
                8;
#else
                inCh[offset_GM - 2 * leftPadding +
                     2 * bottomPadding * blockDim.x];
#endif
        }
    }

    if (x >= INPUT_WIDTH || y >= INPUT_HEIGHT) {
        return;
    }

    // const int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    // int totalPadding = totalPaddingHeight * blockDim.x + totalPaddingWidth *
    // blockDim.y + // sides
    //                totalPaddingHeight*totalPaddingWidth; // Corners
    // float *sfilter = sData + blockDim.x * blockDim.y + totalPadding;

    // Every Thread in block copies it's value
    sInCh[shmemRow_pos_padded] =
#ifdef DebugSHMEM_Data
        1;
#else
        inCh[offset_GM];
#endif
    // Set top padding using all threads in topPadding number of rows
    if (blockIdx.y != 0 && threadIdx.y < topPadding) { // y >= topPadding
        sInCh[shmemRow_pos_padded - topPadding * shmemWidth] =
#ifdef DebugSHMEM_Data
            2;
#else
            inCh[offset_GM - topPadding * INPUT_WIDTH];
#endif
    }
    // Set bottom padding using all threads in bottomPadding number of rows
    else if (y < (INPUT_HEIGHT - bottomPadding) &&
             threadIdx.y >=
                 (blockDim.y -
                  bottomPadding)) { // blockIdx.y != lastY => could pass #
                                    // blocks to kernel or use static #define
                                    // size -> maybe helps performance try
        sInCh[shmemRow_pos_padded + bottomPadding * shmemWidth] =
#ifdef DebugSHMEM_Data
            3;
#else
            inCh[offset_GM + bottomPadding * INPUT_WIDTH];
#endif
    }
    // Use remaining threads for left-over area (left, right, corners +
    // top/bottom padding extra on sides) left-over threads = INPUT_HEIGHT -
    // topPadding - bottomPadding
    else if (threadIdx.y >= topPadding &&
             threadIdx.y <
                 (blockDim.y - bottomPadding)) { // this could be an else
        // Set Left padding
        if (y < (INPUT_HEIGHT - bottomPadding) && x >= leftPadding &&
            threadIdx.x < leftPadding) {
            sInCh[shmemRow_plus_x] =
#ifdef DebugSHMEM_Data
                4;
#else
                inCh[offset_GM - leftPadding];
#endif
        }
        // Set Right padding
        else if (y < (INPUT_HEIGHT - bottomPadding) &&
                 x < (INPUT_WIDTH - rightPadding) &&
                 threadIdx.x >= blockDim.x - rightPadding) {
            sInCh[shmemRow_pos_padded + rightPadding] =
#ifdef DebugSHMEM_Data
                5;
#else
                inCh[offset_GM + rightPadding];
#endif
        }
        // Set left-overs on top left corner
        /*else if (x >= 2*leftPadding && y >= 2*topPadding && // Basically not
        block 0 (but if checking blockIdx would have to split this into two)
            threadIdx.y < 3*topPadding && threadIdx.x < 2*leftPadding &&
            leftPadding <= threadIdx.x) {
            sInCh[shmemRow_plus_x - leftPadding - 2*topPadding*shmemWidth] =
        6;//in[offset_GM - 2*leftPadding - 2*topPadding*INPUT_WIDTH];
        }*/
        // Set left-overs on top right corner
        else if (x <= (INPUT_WIDTH - 2 * rightPadding) && y >= 2 * topPadding &&
                 threadIdx.y < 3 * topPadding &&
                 (blockDim.x - rightPadding) >= threadIdx.x &&
                 threadIdx.x >= (blockDim.x - 2 * rightPadding)) {
            sInCh[shmemRow_pos_padded + 2 * rightPadding -
                  2 * topPadding * shmemWidth] =
#ifdef DebugSHMEM_Data
                7;
#else
                inCh[offset_GM + 2 * rightPadding -
                     2 * topPadding * blockDim.x];
#endif
        }
        // Set left-overs on bottom left corner
        /*else if (x >= 2*leftPadding && (y >= (INPUT_HEIGHT - 3*bottomPadding)
        || threadIdx.y >= (blockDim.y - 3*bottomPadding)) && y < (INPUT_HEIGHT -
        2*bottomPadding) && leftPadding <= threadIdx.x && threadIdx.x <
        2*leftPadding) { sInCh[shmemRow_plus_x - leftPadding +
        2*rightPadding*shmemWidth] = 8;
        }*/
        // Set left-overs on bottom right corner
        else if (x <= (INPUT_WIDTH - 2 * rightPadding) &&
                 (y >= (INPUT_HEIGHT - 3 * bottomPadding) ||
                  threadIdx.y >= (blockDim.y - 3 * bottomPadding)) &&
                 y < (INPUT_HEIGHT - 2 * bottomPadding) &&
                 (blockDim.x - rightPadding) >= threadIdx.x &&
                 threadIdx.x >= (blockDim.x - 2 * rightPadding)) {
            sInCh[shmemRow_pos_padded + 2 * rightPadding +
                  2 * bottomPadding * shmemWidth] =
#ifdef DebugSHMEM_Data
                9;
#else
                inCh[offset_GM + 2 * rightPadding +
                     2 * bottomPadding * blockDim.x];
#endif
        }
    }

    __syncthreads(); // TODO: try only syncing threads used in filter area (or
                     // see if helps with performance)

#ifdef DebugSHMEM
    // printf("%d,%d->%d\n", x,y, shmemRow_pos_padded);
    if (blockIdx.x == 3 && blockIdx.y == 1 && threadIdx.x == 0 &&
        threadIdx.y == 0) {
        printf("Top: %d, Left:%d Right:%d Bottom:%d\n", topPadding, leftPadding,
               rightPadding, bottomPadding);
        printf("SHMEM:\n");
        for (int i = 0, row = 0; i < totalPaddingHeight + blockDim.y;
             ++i, row += shmemWidth) {
            for (int j = 0; j < shmemWidth; ++j) {
                // printf("%d,%d=%f ", i,j, sInCh[row + j]);
                printf("%.0f\t", sInCh[row + j]);
            }
            printf("\n");
        }
    }
#endif

    int cnnOffset = shmemRow_plus_x - topPadding * shmemWidth;

    float conv = 0;
    for (int i = 0, shmemRowOffset = cnnOffset, filterRowOffset = 0;
         i < filterSize;
         ++i, shmemRowOffset += shmemWidth, filterRowOffset += filterSize) {
        for (int j = 0; j < filterSize; ++j) {
            // printf("%d ", shmemRowOffset);
            conv += sInCh[shmemRowOffset + j] * filter[filterRowOffset + j];
        }
    }

    outCh[offset_GM] = conv + b;
}
#endif

__global__ void kernel(float *inCh, float *outCh, float b, float *filter,
                       int filterSize, int totalPaddingHeight,
                       int totalPaddingWidth, int topPadding, int bottomPadding,
                       int leftPadding, int rightPadding) {
#ifdef SHMEM
    device_CNN_SHMEM(inCh, outCh, b, filter, filterSize, totalPaddingHeight,
                     totalPaddingWidth, topPadding, bottomPadding, leftPadding,
                     rightPadding);
#else
    device_CNN(inCh, outCh, b, filter, filterSize, totalPaddingHeight,
               totalPaddingWidth, topPadding, leftPadding);
#endif
}

void setUpCNNFilters(float *host_cov_b, float *host_cov_filter,
                     cudaStream_t stream) {
    gpuErrchk(cudaMemcpyToSymbolAsync(device_cov1_b, host_cov_b,
                                      COV1_FILTER_OUT_CH * sizeof(float), 0,
                                      cudaMemcpyHostToDevice, stream));

    gpuErrchk(cudaMemcpyToSymbolAsync(device_cov1_filter, host_cov_filter,
                                      COV1_FILTER_IN_CH * COV1_FILTER_OUT_CH *
                                          COV1_FILTER_N * COV1_FILTER_N *
                                          sizeof(float),
                                      0, cudaMemcpyHostToDevice, stream));
}

void setupFilterCh(int ch, float *host_cov_b, float *host_cov_filter,
                   cudaStream_t stream) {
    int size = sizeof(float);
    gpuErrchk(cudaMemcpyToSymbolAsync(device_cov1_b, host_cov_b, size,
                                      ch * size, cudaMemcpyHostToDevice,
                                      stream));

    size = COV1_FILTER_N * COV1_FILTER_N * sizeof(float);
    gpuErrchk(cudaMemcpyToSymbolAsync(device_cov1_filter, host_cov_filter, size,
                                      ch * size, cudaMemcpyHostToDevice,
                                      stream));
}

struct Params {
    std::vector<float *> *d_freeBlocks;
    std::mutex *d_freeBlocksMutex;
    std::vector<int> *freeStreams;
    std::mutex *freeStreamsMutex;
    int streamIdx;
    float *inBlock;
    int *count;
#ifdef PRINTDATA
    float *hOutBlock;
    int outCh;
#endif
};

void callbackProcessFinished(cudaStream_t stream, cudaError_t status,
                             void *arg) {
    Params *params = (Params *)arg;
    // printf("call: %d %d\n", *params->count, params->streamIdx);
    // Reuse stream
#ifdef EnableLock
    params->freeStreamsMutex->lock();
#endif
    params->freeStreams->push_back(params->streamIdx);
#ifdef EnableLock
    params->freeStreamsMutex->unlock();
#endif
    // printf("callend: %d\n", *params->count);

    // Reuse memory
#ifdef EnableLock
    params->d_freeBlocksMutex->lock();
#endif
    ++(*params->count);
    if (*params->count == COV1_FILTER_OUT_CH) {
        params->d_freeBlocks->push_back(params->inBlock);
    }
#ifdef EnableLock
    params->d_freeBlocksMutex->unlock();
#endif

#ifdef PRINTDATA
    printf("Output ch%d:\n", params->outCh);
    Print2D(params->hOutBlock, INPUT_WIDTH, INPUT_HEIGHT);
    // free(&params->hOutBlock[params->outCh]);
#endif
}

void layer1_cov1(int bytes, dim3 grid, dim3 block, cudaStream_t *streams,
                 float *h_input, float *d_input, float **d_output,
#ifdef PRINTDATA
                 float **h_output,
#endif
                 std::vector<float *> *d_freeBlocks,
                 std::mutex *d_freeBlocksMutex, std::vector<int> *freeStreams,
                 std::mutex *freeStreamsMutex) {

    int totalPaddingHeight, totalPaddingWidth;
    int topPadding, leftPadding, bottomPadding, rightPadding;
    getConvPadding(COV1_FILTER_N, totalPaddingHeight, totalPaddingWidth,
                   topPadding, leftPadding, bottomPadding, rightPadding);

    const int shmemSize = (INPUT_HEIGHT + totalPaddingHeight) *
                          (INPUT_WIDTH + totalPaddingWidth) * sizeof(float);

    cudaEvent_t event;
    gpuErrchk(cudaEventCreate(&event));

    Params params[COV1_FILTER_OUT_CH];
    int count = 0;
    for (int i = 0; i < COV1_FILTER_OUT_CH; ++i) {
        params[i].freeStreams = freeStreams;
        params[i].freeStreamsMutex = freeStreamsMutex;
        params[i].d_freeBlocks = d_freeBlocks;
        params[i].d_freeBlocksMutex = d_freeBlocksMutex;
        params[i].count = &count;
        params[i].inBlock = d_input;
    }

    float *filterAddr;
    gpuErrchk(cudaGetSymbolAddress((void **)&filterAddr, device_cov1_filter));

    bool got = false;
    int streamIdx = 0;
    for (int i = 0; i < COV1_FILTER_OUT_CH; ++i) {
        // Get Stream
        streamIdx = -1;
        while (streamIdx == -1) {
#ifdef EnableLock
            freeStreamsMutex->lock();
#endif
            if (!freeStreams->empty()) {
                streamIdx = freeStreams->back();
                freeStreams->pop_back();
            }
#ifdef EnableLock
            freeStreamsMutex->unlock();
#endif
        }
        // Copy over input
        if (i == 0) {
            // Performing async for implementation of multiple CNNs running in
            // parallel for server

            // Copy over input
            gpuErrchk(cudaMemcpyAsync(d_input, h_input, bytes,
                                      cudaMemcpyHostToDevice,
                                      streams[streamIdx]));
            gpuErrchk(cudaEventRecord(event, streams[streamIdx]));
        }
        // Every stream needs to wait for input
        if (i > 0) // input cpy and first filter run on same stream so skip on
                   // first stream
            gpuErrchk(cudaStreamWaitEvent(streams[streamIdx], event, 0));

        // Setup all filters and b
        setupFilterCh(i, &host_cov1_b[i], &host_cov1_filter[0][i][0][0],
                      streams[streamIdx]);

        // Get output memory
        got = false;
#ifdef EnableLock
        d_freeBlocksMutex->lock();
#endif
        if (!d_freeBlocks->empty()) {
            got = true;
            d_output[i] = d_freeBlocks->back();
            d_freeBlocks->pop_back();
        }
#ifdef EnableLock
        d_freeBlocksMutex->unlock();
#endif
        if (!got) {
            gpuErrchk(cudaMalloc((void **)&d_output[i], bytes));
        }

#ifdef SHMEM
        kernel<<<grid, block, shmemSize, streams[streamIdx]>>>(
            d_input, d_output[i], host_cov1_b[i],
            filterAddr + i * COV1_FILTER_N * COV1_FILTER_N, COV1_FILTER_N,
            totalPaddingHeight, totalPaddingWidth, topPadding, bottomPadding,
            leftPadding, rightPadding);
#else
        kernel<<<grid, block, 0, streams[streamIdx]>>>(
            d_input, d_output[i], host_cov1_b[i],
            filterAddr + i * COV1_FILTER_N * COV1_FILTER_N, COV1_FILTER_N,
            totalPaddingHeight, totalPaddingWidth, topPadding, bottomPadding,
            leftPadding, rightPadding);
#endif

#ifdef PRINTDATA
        // If want output then need to copy back to host memory
        gpuErrchk(cudaMemcpyAsync(h_output[i], d_output[i], bytes,
                                  cudaMemcpyDeviceToHost, streams[streamIdx]));
        params[i].hOutBlock = h_output[i];
        params[i].outCh = i;
#endif

        params[i].streamIdx = streamIdx;
        gpuErrchk(cudaStreamAddCallback(streams[streamIdx],
                                        callbackProcessFinished,
                                        (void *)&params[i], 0));
        // printf("i:%d\n", i);
    }
    // TODO: fix not needing this
    // need as when function ends stack cleared, so params dropped but then
    // accessed by callback still
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaEventDestroy(event));

    /*d_freeBlocksMutex.lock();
    // Add device input to list of free blocks
    d_freeBlocks.push_back(d_input);
    d_freeBlocksMutex.unlock();*/
}

int main(int argc, char *argv[]) {
    // TODO: maybe don't need mutex and change vector to queue
    std::vector<float *> d_freeBlocks;
    std::mutex d_freeBlocksMutex;

    std::vector<float *> h_freeBlocks;
    std::mutex h_freeBlocksMutex;

    std::vector<int> freeStreams;
    std::mutex freeStreamsMutex;

    int blockSize = INPUT_HEIGHT * INPUT_WIDTH;
    int bytes = blockSize * sizeof(float);

    cudaStream_t streams[NUM_STREAM]; //[COV1_FILTER_OUT_CH];

    // Setup filter values
    std::fill(&host_cov1_b[0], &host_cov1_b[0] + COV1_FILTER_OUT_CH, 1.0);
    std::fill(&host_cov1_filter[0][0][0][0],
              &host_cov1_filter[0][0][0][0] + COV1_FILTER_IN_CH *
                                                  COV1_FILTER_OUT_CH *
                                                  COV1_FILTER_N * COV1_FILTER_N,
              1.0);

    gpuErrchk(cudaDeviceReset());

    // Allocate intial input to CNN
    float *h_input =
        allocHostBlockHelper(h_freeBlocks, h_freeBlocksMutex, bytes);
    if (h_input == NULL) {
        printf("Error: Failed to allocte host memory for input");
        return 1;
    }
    float value = 1.0;
    initData(h_input, INPUT_WIDTH, INPUT_HEIGHT, 0, &value);
    float *d_input =
        allocDeviceBlockHelper(d_freeBlocks, d_freeBlocksMutex, bytes);
    if (d_input == NULL) {
        printf("Error: Failed to allocte host memory for input");
        return 1;
    }

#ifdef PRINTDATA
    // Allocate host output to print results for debugging
    float *h_output[COV1_FILTER_OUT_CH];
    for (int ch = 0; ch < COV1_FILTER_OUT_CH; ++ch) {
        h_output[ch] =
            allocHostBlockHelper(h_freeBlocks, h_freeBlocksMutex, bytes);
        if (h_output[ch] == NULL) {
            printf("Error: Failed to allocte host memory for output ch %d", ch);
            // TODO: need to clean up allocated upto this point
            return 1;
        }
    }
    // Pinning host memory so pages are not paged to disk for DMA to work
    for (int ch = 0; ch < COV1_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaHostRegister(h_output[ch], bytes, 0));
    }

    printf("Input:\n");
    Print2D(h_input, INPUT_WIDTH, INPUT_HEIGHT);
#endif

    // Pinning host memory so pages are not paged to disk for DMA to work
    gpuErrchk(cudaHostRegister(h_input, bytes, 0));

    dim3 block((INPUT_WIDTH < 32) ? INPUT_WIDTH : 32,
               (INPUT_HEIGHT < 32) ? INPUT_HEIGHT : 32);
    dim3 grid((INPUT_WIDTH + block.x - 1) / block.x,
              (INPUT_HEIGHT + block.y - 1) / block.y);

    for (int i = NUM_STREAM - 1; i >= 0; --i) {
        gpuErrchk(cudaStreamCreate(&streams[i]));
        freeStreams.push_back(i);
    }

    //================= Timing Begins ========================
    double start_time = getTimeStamp();

    /* For simple testing of convolution kernel
    setUpCNNFilters();
    double constMemFilter_time = getTimeStamp();
    float *d_output;
    gpuErrchk(cudaMalloc((void **)&d_output, bytes));
    float *filterAddr;
    gpuErrchk(cudaGetSymbolAddress((void**)&filterAddr, device_cov1_filter));
    kernel<<<grid, block>>>(d_input, d_output, filterAddr, COV1_FILTER_N);
    gpuErrchk(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));*/

    // Perform First convolution layer
    float *d_cov1_out[COV1_FILTER_OUT_CH];

    layer1_cov1(bytes, grid, block, streams, h_input, d_input, d_cov1_out,
#ifdef PRINTDATA
                h_output,
#endif
                &d_freeBlocks, &d_freeBlocksMutex, &freeStreams,
                &freeStreamsMutex);

    // Need to wait for stream to complete copy
    // gpuErrchk(cudaHostUnregister(h_output[ch]));
    // for (int i = 0; i < NUM_STREAM; ++i) {
    // printf("Sync %d ", i);
    // gpuErrchk(cudaStreamSynchronize(streams[i]));
    //}
    // gpuErrchk(cudaDeviceSynchronize());

    double end_time = getTimeStamp();
    //================= Timing Ends ========================
    int total_time_ms = (int)ceil((end_time - start_time) * 1000);
    // int constMemFilter_time_ms = (int)ceil((constMemFilter_time -
    // start_time)*1000);

    printf("Total Time: %d\n", total_time_ms);
    // printf("Filter Cpy Time: %d\n", constMemFilter_time_ms);

    // Input Not needed anymore by device
    gpuErrchk(cudaHostUnregister(h_input));
    // free(h_input);

    // Add device output to list of free blocks
    for (int i = 0; i < COV1_FILTER_OUT_CH; ++i) {
        d_freeBlocks.push_back(d_cov1_out[i]);
    }

    // Clean up streams
    while (!freeStreams.empty()) {
        int streamIdx = freeStreams.back();
        freeStreams.pop_back();
        // printf("%d ", streamIdx);
        gpuErrchk(cudaStreamDestroy(streams[streamIdx]));
    }

    // Clean up device blocks
    while (!d_freeBlocks.empty()) {
        float *block = d_freeBlocks.back();
        d_freeBlocks.pop_back();
        gpuErrchk(cudaFree(block));
    }

    gpuErrchk(cudaDeviceReset());

    return 0;
}
