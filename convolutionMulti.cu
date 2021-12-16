/*
* ECE1782 - Fall 2021 - Project
* nvcc -arch sm_52 -Xptxas="-v" final.cu

nvcc convolutionMulti.cu -Xptxas="-v" --use_fast_math

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
#include "cnn_weights.cu"
#include <cmath>
//#include "cublas_v2.h"
#include "config.h"
#include "utils.cu"
#include "layers/linear.cu"
#include "layers/max_pool_2d.cu"

#define PRINTDATA 1
#define SHMEM 1
//#define DebugSHMEM 1
//#define DebugSHMEM_Data 1

#define INPUT_WIDTH 100//2048//100
#define INPUT_HEIGHT 56//2048//56

// Linear Config
#define INPUT_SIZE1 22400
#define OUTPUT_SIZE1 256
#define INPUT_SIZE2 256
#define OUTPUT_SIZE2 3


// Currently a thread per pooling, but thread no reading coalesed
// could read coalesed by copying to shared memory and then reorder in shared memory linearly

__constant__ float *device_output[COV1_FILTER_OUT_CH];

#ifdef SHMEM
__global__ void device_CNN_Multi_SHMEM(int in_cols, int in_rows, float *inCh, int filterSize, bool isSingle, bool isFirst, bool isLast,
    int totalPaddingHeight, int totalPaddingWidth, int topPadding, int bottomPadding, int leftPadding, int rightPadding) {
    /* Shared Memory Layout:
    000...000 
    000...000
    000111000 <- place block of inCh here, but will have to place padding area for block edges handling 
    000111000
    000111000
    000111000
    000...000
    000...000
    */
    extern __shared__ float sData[];
    float *sInCh = sData;

    // Position relative to global memory of 2D matrix
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    const int offset_GM = y * in_cols + x;

    // SHMEM offsets (includes padding)
    int shmemWidth = totalPaddingWidth + blockDim.x;
    int shmemRow = (threadIdx.y + topPadding) * shmemWidth; 
    int shmemRow_plus_x = shmemRow + threadIdx.x;
    int shmemRow_pos_padded = shmemRow_plus_x + leftPadding;
    
    // Initialize shared memory to zero - when block is on edge of grid
    // TODO: try to make more efficent by including in below code if checks
    // As here threads are doing duplicate work
    sInCh[shmemRow_pos_padded - topPadding*shmemWidth - leftPadding] = 
#ifdef DebugSHMEM_Data
    -1;
#else
    0;
#endif
    sInCh[shmemRow_pos_padded - topPadding*shmemWidth] = 
#ifdef DebugSHMEM_Data
    -1;
#else
    0;
#endif
    sInCh[shmemRow_pos_padded - topPadding*shmemWidth + rightPadding] =
#ifdef DebugSHMEM_Data
    -1;
#else
    0;
#endif
    sInCh[shmemRow_pos_padded + bottomPadding*shmemWidth - leftPadding] =
#ifdef DebugSHMEM_Data
    -1;
#else
    0;
#endif
    sInCh[shmemRow_pos_padded + bottomPadding*shmemWidth] = 
#ifdef DebugSHMEM_Data
    -1;
#else
    0;
#endif
    sInCh[shmemRow_pos_padded + bottomPadding*shmemWidth + rightPadding] =
#ifdef DebugSHMEM_Data
    -1;
#else
    0;
#endif
    __syncthreads();
    
    // Now setup Shared memory with data
    if (threadIdx.y >= topPadding && threadIdx.y < (blockDim.y - bottomPadding)) { // this could be an else
        // Set left-overs on top left corner
        if (x >= 2*leftPadding && y >= 2*topPadding && // Basically not block 0 (but if checking blockIdx would have to split this into two)
            threadIdx.y < 3*topPadding && threadIdx.x < 2*leftPadding &&
            leftPadding <= threadIdx.x) {
            sInCh[shmemRow_plus_x - leftPadding - 2*topPadding*shmemWidth] = 
#ifdef DebugSHMEM_Data
            6;
#else
            inCh[offset_GM - 2*leftPadding - 2*topPadding*in_cols];
#endif
        }
        // Set left-overs on bottom left corner
        else if (x >= 2*leftPadding && (y >= (in_rows - 3*bottomPadding) || threadIdx.y >= (blockDim.y - 3*bottomPadding)) && y < (in_rows - 2*bottomPadding) &&
            leftPadding <= threadIdx.x && threadIdx.x < 2*leftPadding) {
            sInCh[shmemRow_plus_x - leftPadding + 2*rightPadding*shmemWidth] = 
#ifdef DebugSHMEM_Data
            8;
#else
            inCh[offset_GM - 2*leftPadding  + 2*bottomPadding*blockDim.x];
#endif            
        }
    }

    if (x >= in_cols || y >= in_rows) {
        return;
    }
    
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
        inCh[offset_GM - topPadding*in_cols];
#endif
    }
    // Set bottom padding using all threads in bottomPadding number of rows
    else if (y < (in_rows - bottomPadding) && threadIdx.y >= (blockDim.y - bottomPadding)) { //blockIdx.y != lastY => could pass # blocks to kernel or use static #define size -> maybe helps performance try
        sInCh[shmemRow_pos_padded + bottomPadding * shmemWidth] = 
#ifdef DebugSHMEM_Data
        3;
#else
        inCh[offset_GM + bottomPadding*in_cols];
#endif
    }
    // Use remaining threads for left-over area (left, right, corners + top/bottom padding extra on sides)
    // left-over threads = INPUT_HEIGHT - topPadding - bottomPadding 
    else if (threadIdx.y >= topPadding && threadIdx.y < (blockDim.y - bottomPadding)) { // this could be an else
        // Set Left padding 
        if (y < (in_rows - bottomPadding) && x >= leftPadding && threadIdx.x < leftPadding) {
            sInCh[shmemRow_plus_x] = 
#ifdef DebugSHMEM_Data
            4;
#else
            inCh[offset_GM - leftPadding];
#endif
        }
        // Set Right padding
        else if (y < (in_rows - bottomPadding) && x < (in_cols - rightPadding) && threadIdx.x >= blockDim.x - rightPadding) {
            sInCh[shmemRow_pos_padded + rightPadding] = 
#ifdef DebugSHMEM_Data
            5;
#else
            inCh[offset_GM + rightPadding];
#endif
        }
        // Set left-overs on top left corner
        /*else if (x >= 2*leftPadding && y >= 2*topPadding && // Basically not block 0 (but if checking blockIdx would have to split this into two)
            threadIdx.y < 3*topPadding && threadIdx.x < 2*leftPadding &&
            leftPadding <= threadIdx.x) {
            sInCh[shmemRow_plus_x - leftPadding - 2*topPadding*shmemWidth] = 6;//in[offset_GM - 2*leftPadding - 2*topPadding*INPUT_WIDTH];
        }*/
        // Set left-overs on top right corner
        else if (x <= (in_cols - 2*rightPadding) && y >= 2*topPadding &&
            threadIdx.y < 3*topPadding && 
            (blockDim.x - rightPadding) >= threadIdx.x && threadIdx.x >= (blockDim.x - 2*rightPadding) ) {
            sInCh[shmemRow_pos_padded + 2*rightPadding - 2*topPadding*shmemWidth] = 
#ifdef DebugSHMEM_Data
            7; 
#else
            inCh[offset_GM + 2*rightPadding - 2*topPadding*blockDim.x];
#endif
        }
        // Set left-overs on bottom left corner
        /*else if (x >= 2*leftPadding && (y >= (INPUT_HEIGHT - 3*bottomPadding) || threadIdx.y >= (blockDim.y - 3*bottomPadding)) && y < (INPUT_HEIGHT - 2*bottomPadding) &&
            leftPadding <= threadIdx.x && threadIdx.x < 2*leftPadding) {
            sInCh[shmemRow_plus_x - leftPadding + 2*rightPadding*shmemWidth] = 8;            
        }*/
        // Set left-overs on bottom right corner
        else if (x <= (in_cols - 2*rightPadding) && 
            (y >= (in_rows - 3*bottomPadding) || threadIdx.y >= (blockDim.y - 3*bottomPadding)) && y < (in_rows - 2*bottomPadding) &&
            (blockDim.x - rightPadding) >= threadIdx.x && threadIdx.x >= (blockDim.x - 2*rightPadding)) {
            sInCh[shmemRow_pos_padded + 2*rightPadding + 2*bottomPadding*shmemWidth] = 
#ifdef DebugSHMEM_Data
            9;
#else
            inCh[offset_GM + 2*rightPadding + 2*bottomPadding*blockDim.x];
#endif            
        }
    }

    __syncthreads(); //TODO: try only syncing threads used in filter area (or see if helps with performance)

#ifdef DebugSHMEM
    //printf("%d,%d->%d\n", x,y, shmemRow_pos_padded);
    if (blockIdx.x == 3 && blockIdx.y == 1 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("Top: %d, Left:%d Right:%d Bottom:%d\n", topPadding, leftPadding, rightPadding, bottomPadding);
        printf("SHMEM:\n");
        for (int i=0, row=0; i < totalPaddingHeight + blockDim.y; ++i, row += shmemWidth) {
            for (int j=0; j < shmemWidth; ++j) {
                //printf("%d,%d=%f ", i,j, sInCh[row + j]);
                printf("%.0f\t", sInCh[row + j]);
            }
            printf("\n");
        }
    }
#endif

    int cnnOffset = shmemRow_plus_x - topPadding*shmemWidth;
    float cacheIn[COV1_FILTER_N * COV1_FILTER_N];
    float *f = &device_cov1_filter[0][0][0][0];
    int totalFilterSize = filterSize*filterSize;
    int i = 0;
    for (int row=0, shmemRowOffset=cnnOffset; i < filterSize; ++i, row += filterSize, shmemRowOffset += shmemWidth) {
        for (int j = 0; j < filterSize; ++j) {
            cacheIn[row + j] = sInCh[shmemRowOffset + j];
        }
    }

    for (int ch=0, filterChOffset = 0; ch < COV1_FILTER_OUT_CH; ++ch, filterChOffset += totalFilterSize) {
        float conv = 0.0;
        for (i = 0; i < totalFilterSize; ++i) {
            conv += cacheIn[i] * *(f + filterChOffset + i);
        }
        if (isSingle) {
            device_output[ch][offset_GM] = relu(conv + device_cov1_b[ch]);
        }
        else if (isFirst) {
            device_output[ch][offset_GM] = conv;
        }
        else if (isLast) {
            conv += device_output[ch][offset_GM] + device_cov1_b[ch];
            device_output[ch][offset_GM] = relu(conv);
        }
        else {
            device_output[ch][offset_GM] += conv;
        }
    }
}
#else
//template <int filterSize> // Can't template cause multiple functions with some at 77 registers
__global__ void device_CNN_Multi_v1(int in_cols, int in_rows, float *inCh, int filterSize, bool isSingle, bool isFirst, bool isLast,
    int totalPaddingHeight, int totalPaddingWidth, int topPadding, int bottomPadding, int leftPadding, int rightPadding) {
    // Position relative to global memory of 2D matrix
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= in_cols || y >= in_rows) {
        return;
    }
    int totalFilterSize = filterSize*filterSize;

    const int offset_GM = y * in_cols + x;
    int cnnOffset = offset_GM - topPadding*in_cols - leftPadding;
    float cacheIn[COV1_FILTER_N * COV1_FILTER_N]; //NOTE requires constant size arg, else compiler complains (so constant memory), could make template
    int boundryH = y - topPadding;
    int boundryW = x - leftPadding;
    int i = 0;
    for (int row=0; i < filterSize; ++i, row += filterSize, cnnOffset += in_cols, ++boundryH) {
        for (int j = 0; j < filterSize; ++j) {
            if (boundryH < 0 || boundryH >= in_rows || 
                (boundryW + j < 0) || (boundryW + j >= in_cols)) {
                cacheIn[row + j] = 0;
            }               
            else {
                cacheIn[row + j] = inCh[cnnOffset + j];
            }
        }
    }

    float *f = &device_cov1_filter[0][0][0][0];
    for (int ch=0, filterChOffset = 0; ch < COV1_FILTER_OUT_CH; ++ch, filterChOffset += totalFilterSize) {
        float conv = 0.0;
        for (i = 0; i < totalFilterSize; ++i) {
            conv += cacheIn[i] * *(f + filterChOffset + i);
        }
        if (isSingle) {
            device_output[ch][offset_GM] = relu(conv + device_cov1_b[ch]);
        }
        else if (isFirst) {
            device_output[ch][offset_GM] = conv;
        }
        else if (isLast) {
            conv += device_output[ch][offset_GM] + device_cov1_b[ch];
            device_output[ch][offset_GM] = relu(conv);
        }
        else {
            device_output[ch][offset_GM] += conv;
        }
    }
}

__global__ void device_CNN_Multi_v2(int in_cols, int in_rows, float *inCh, int filterSize, bool isSingle, bool isFirst, bool isLast,
    int totalPaddingHeight, int totalPaddingWidth, int topPadding, int bottomPadding, int leftPadding, int rightPadding) {
    // Position relative to global memory of 2D matrix
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= in_cols || y >= in_rows) {
        return;
    }

    const int offset_GM = y * in_cols + x;
    int cnnOffset = offset_GM - topPadding*in_cols - leftPadding;

    //float covResult[64]; // To many registers so have to use GMEM of output
    float readVal;
    int boundryHeight = y - topPadding;
    int boundryWidth = x - leftPadding;
    int filterTotalSize = filterSize*filterSize;
    float *f = &device_cov1_filter[0][0][0][0];
    int ch, filterChOffset;

    for (int i = 0, row=0; i < filterSize; ++i, row += filterSize, cnnOffset += in_cols, ++boundryHeight) {
        if (boundryHeight < 0 || boundryHeight >= in_rows) continue;
        for (int j = 0; j < filterSize; ++j) {
            if ((boundryWidth + j < 0) || (boundryWidth + j >= in_cols)) continue;
            
            readVal = inCh[cnnOffset + j];
            for (ch=0, filterChOffset=0; ch < 64; ++ch, filterChOffset += filterTotalSize) {
                //covResult[ch] += readVal * *(f + filterChOffset + row + j);
                //covResult[ch] = __fmaf_rd(readVal, *(f + filterChOffset + row + j), covResult[ch]);
                if (i == 0 && isFirst) {
                    device_output[ch][offset_GM] = readVal * *(f + filterChOffset + row + j);
                }
                else {
                    device_output[ch][offset_GM] += readVal * *(f + filterChOffset + row + j);
                }
            }
        }
    }

    if (isLast) {
        for (ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
            device_output[ch][offset_GM] = relu(device_output[ch][offset_GM] + device_cov1_b[ch]);
            /*else if (isFirst) {
                device_output[ch][offset_GM] = device_output[ch][offset_GM];
            }*/
            /*else if (isLast) {
                covResult[ch] += device_output[ch][offset_GM] + device_cov1_b[ch];
                device_output[ch][offset_GM] = relu(covResult[ch]);
            }*/
            /*else {
                device_output[ch][offset_GM] += covResult[ch];
            }*/
        }
    }
}
#endif


/*__global__ void kernel_multi(int in_cols, int in_rows, float *inCh, int filterSize, bool isSingle, bool isFirst, bool isLast,
    int totalPaddingHeight, int totalPaddingWidth, int topPadding, int bottomPadding, int leftPadding, int rightPadding) {
    printf("%d %d %d %d %d %d", totalPaddingHeight, totalPaddingWidth, topPadding, bottomPadding, leftPadding, rightPadding);
#ifdef SHMEM
    device_CNN_Multi_SHMEM(in_cols, in_rows, inCh, filterSize, isSingle, isFirst, isLast,
        totalPaddingHeight, totalPaddingWidth, topPadding, bottomPadding, leftPadding, rightPadding);
#else  
    device_CNN_Multi(in_cols, in_rows);/*, inCh, filterSize, isSingle, isFirst, isLast,
        totalPaddingHeight, totalPaddingWidth, topPadding, bottomPadding, leftPadding, rightPadding);*/
/*#endif
}*/

__global__ void flatten(float *d_conv_out, float *d_flattened, int channel, int size) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x >= size) {
        return;
    }

    int offset = channel*size;

    d_flattened[offset+x] = d_conv_out[x];
}


float* softmax(int size, float* z)
{
    float max = 0;
    for (int i = 0; i < size; i++) {
        if (z[i] > max) {
            max = z[i];
        }
    }
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += expf(z[i]-max);
    }
    float* buff = new float[size];
    for (int i = 0; i < size; i++){
        buff[i] = expf(z[i]-max) / sum;
    }
    return buff;
}



void setUpCNNFilters(float *host_cov_b, float * host_cov_filter) {
    gpuErrchk( cudaMemcpyToSymbol(device_cov1_b, host_cov_b, COV1_FILTER_OUT_CH*sizeof(float), 
        0, cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpyToSymbol(device_cov1_filter, host_cov_filter, 
        COV1_FILTER_IN_CH * COV1_FILTER_OUT_CH * COV1_FILTER_N*COV1_FILTER_N*sizeof(float), 
        0, cudaMemcpyHostToDevice));
}


void layer1_cov1_multi(int bytes, dim3 grid, dim3 block,
    float *h_input, float *d_input, float *d_output[COV1_FILTER_OUT_CH]) {
    
    float *filterAddr;
    gpuErrchk(cudaGetSymbolAddress((void**)&filterAddr, device_cov1_filter));

    // Copy over input
    gpuErrchk(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Setup all filters and b
    setUpCNNFilters(host_cov1_b, &host_cov1_filter[0][0][0][0]);
    
    // Get output memory
    for (int ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaMalloc((void **)&d_output[ch], bytes));
        // printf("Addr%d: %p %p", ch, &d_output[ch], d_output[ch]);
        gpuErrchk( cudaMemcpyToSymbol(device_output[0], &d_output[ch], sizeof(float*),
            ch*sizeof(float*), cudaMemcpyHostToDevice) );
    }

    int totalPaddingHeight, totalPaddingWidth;
    int topPadding, leftPadding, bottomPadding, rightPadding;
    getConvPadding(COV1_FILTER_N, totalPaddingHeight, totalPaddingWidth,
        topPadding, leftPadding, bottomPadding, rightPadding);
    //printf("%d %d %d %d %d %d ", totalPaddingHeight, totalPaddingWidth,
    //    topPadding, leftPadding, bottomPadding, rightPadding);
#ifdef SHMEM
    const int shmemSize = (INPUT_HEIGHT + totalPaddingHeight) * (INPUT_WIDTH + totalPaddingWidth) * sizeof(float);
    device_CNN_Multi_SHMEM<<<grid, block, shmemSize>>>(INPUT_WIDTH, INPUT_HEIGHT, d_input, COV1_FILTER_N, true, false, false,
        totalPaddingHeight, totalPaddingWidth, topPadding, bottomPadding, leftPadding, rightPadding);   
#else
    device_CNN_Multi_v1<<<grid, block>>>(INPUT_WIDTH, INPUT_HEIGHT, d_input, COV1_FILTER_N, true, false, false,
                totalPaddingHeight, totalPaddingWidth, topPadding, bottomPadding, leftPadding, rightPadding);
#endif

    gpuErrchk(cudaDeviceSynchronize());
}

void layer1_maxPool_multi(int in_cols, int in_rows, float **d_input, 
    int &out_cols, int &out_rows, float **d_output) {
    
    out_cols = (in_cols - 1) / STRIDE + 1;
    out_rows = (in_rows - 1) / STRIDE + 1;
    //printf("Out Size: %d,%d\n", out_rows, out_cols);

    int out_elements = out_rows * out_cols;
    int out_bytes = out_elements * sizeof(float);

    //cudaStream_t streams[COV1_FILTER_OUT_CH]; //[COV1_FILTER_OUT_CH];
    dim3 block((out_cols < 32) ? out_cols : 32, (out_rows < 32) ? out_rows : 32); 
    dim3 grid( (out_cols + block.x-1) / block.x, 
               (out_rows + block.y-1) / block.y);

    // Get output memory
    for (int ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
        //gpuErrchk(cudaStreamCreate(&streams[ch]));
        gpuErrchk(cudaMalloc((void **)&d_output[ch], out_bytes));
        max_pool_2d<<<grid, block/*, 0, streams[ch]*/>>>(d_input[ch], in_rows, in_cols, d_output[ch], out_rows, out_cols);
    }

    gpuErrchk(cudaDeviceSynchronize());
    /*for (int ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaStreamSynchronize(streams[ch]));
        gpuErrchk(cudaStreamDestroy(streams[ch]));
    }*/
}

void layer2_cov_multi(int filterSize, int in_cols, int in_rows, float **d_input, float **d_output) {

    //float *filterAddr;
    //gpuErrchk(cudaGetSymbolAddress((void**)&filterAddr, device_cov1_filter));
    
    int bytes = in_cols * in_rows * sizeof(float);
    dim3 block((in_cols < 32) ? in_cols : 32, (in_rows < 32) ? in_rows : 32); 
    dim3 grid( (in_cols + block.x-1) / block.x, 
               (in_rows + block.y-1) / block.y);


    // Get output memory
    for (int ch=0; ch < COV2_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaMalloc((void **)&d_output[ch], bytes));
        // printf("Addr%d: %p %p", ch, &d_output[ch], d_output[ch]);
        gpuErrchk( cudaMemcpyToSymbol(device_output[0], &d_output[ch], sizeof(float*),
            ch*sizeof(float*), cudaMemcpyHostToDevice) );
    }

    int totalPaddingHeight, totalPaddingWidth;
    int topPadding, leftPadding, bottomPadding, rightPadding;
    getConvPadding(filterSize, totalPaddingHeight, totalPaddingWidth,
        topPadding, leftPadding, bottomPadding, rightPadding);
    //printf("%d %d %d %d\n", totalPaddingHeight, totalPaddingWidth, topPadding, leftPadding);

#ifdef SHMEM
    const int shmemSize = (in_rows + totalPaddingHeight) * (in_cols + totalPaddingWidth) * sizeof(float);
#endif
    for (int ch=0; ch < COV2_FILTER_OUT_CH; ++ch) {
        // Setup all filters and b
        setUpCNNFilters(host_cov1_b, &host_cov1_filter[0][0][0][0]); //TODO UPdate for each iteration

#ifdef SHMEM
        device_CNN_Multi_SHMEM<<<grid, block, shmemSize>>>(in_cols, in_rows, d_input[ch], filterSize, false, (ch == 0) ? true : false, (ch == (COV2_FILTER_OUT_CH-1)) ? true : false,
                totalPaddingHeight, totalPaddingWidth, topPadding, bottomPadding, leftPadding, rightPadding);   
#else
        device_CNN_Multi_v1<<<grid, block>>>(in_cols, in_rows, d_input[ch], filterSize, false, (ch == 0) ? true : false, (ch == (COV2_FILTER_OUT_CH-1)) ? true : false,
                totalPaddingHeight, totalPaddingWidth, topPadding, bottomPadding, leftPadding, rightPadding);
#endif
        gpuErrchk(cudaDeviceSynchronize());
    }
}


int main( int argc, char *argv[])
{   
    // TODO: maybe don't need mutex and change vector to queue
    int blockSize = INPUT_HEIGHT*INPUT_WIDTH;
    int bytes = blockSize * sizeof(float);
    
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
    // Allocate host output to print results for debugging
    float *h_output[COV1_FILTER_OUT_CH];
    for (int ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
        h_output[ch] = allocHostBlock(bytes);
        if (h_output[ch] == NULL) {
            printf("Error: Failed to allocte host memory for output ch %d", ch);
            //TODO: need to clean up allocated upto this point
            return 1;
        }
    }
    // Pinning host memory so pages are not paged to disk for DMA to work
    for (int ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaHostRegister(h_output[ch], bytes, 0));
    }

    printf("Input:\n");
    Print2D(h_input, INPUT_WIDTH, INPUT_HEIGHT);
#endif
    
    // Pinning host memory so pages are not paged to disk for DMA to work
    gpuErrchk(cudaHostRegister(h_input, bytes, 0));

    dim3 block((INPUT_WIDTH < 32) ? INPUT_WIDTH : 32, (INPUT_HEIGHT < 32) ? INPUT_HEIGHT : 32); 
    dim3 grid( (INPUT_WIDTH + block.x-1) / block.x, 
               (INPUT_HEIGHT + block.y-1) / block.y);

    int out_col = INPUT_WIDTH, out_row = INPUT_HEIGHT;
    float *d_in[COV1_FILTER_OUT_CH]; 
    float *d_out[COV1_FILTER_OUT_CH];

    // linear layers setup
    // alloc memory host-side for weights and biases needed in linear layers
    float *h_W1 = (float *) malloc (INPUT_SIZE1 * OUTPUT_SIZE1 * sizeof(float));
    float *h_b1 = (float *) malloc (OUTPUT_SIZE1 * sizeof(float));
    float *h_W2 = (float *) malloc (INPUT_SIZE2 * OUTPUT_SIZE2 * sizeof(float));
    float *h_b2 = (float *) malloc (OUTPUT_SIZE2 * sizeof(float));
    float *h_linear_out = (float *) malloc (OUTPUT_SIZE2 * sizeof(float)); // host output of linear layers
    // pinning host memory
    gpuErrchk(cudaHostRegister(h_W1, INPUT_SIZE1 * OUTPUT_SIZE1 * sizeof(float), 0));
    gpuErrchk(cudaHostRegister(h_b1, OUTPUT_SIZE1 * sizeof(float), 0));
    gpuErrchk(cudaHostRegister(h_W2, INPUT_SIZE2 * OUTPUT_SIZE2 * sizeof(float), 0));
    gpuErrchk(cudaHostRegister(h_b2, OUTPUT_SIZE2 * sizeof(float), 0));
    // init weights and biases
    initWeights(h_W1, INPUT_SIZE1, OUTPUT_SIZE1);
    initBias(h_b1, OUTPUT_SIZE1);
    initWeights(h_W2, INPUT_SIZE2, OUTPUT_SIZE2);
    initBias(h_b2, OUTPUT_SIZE2);
    
//================= Timing Begins ========================
    double start_time=getTimeStamp();

    float *d_input = allocDeviceBlock(bytes);
    if (d_input == NULL) {
        printf("Error: Failed to allocte host memory for input");
        return 1;
    }

    // Perform First convolution layer
    layer1_cov1_multi(bytes, grid, block, h_input, d_input, d_out);

    // Input Not needed anymore by device
    gpuErrchk(cudaHostUnregister(h_input));
    free(h_input);
    gpuErrchk(cudaFree(d_input));

    // Perform first max pooling on the output from the first convolution layer
    std::copy(d_out, d_out + COV1_FILTER_OUT_CH, d_in);
    layer1_maxPool_multi(INPUT_WIDTH, INPUT_HEIGHT, d_in, 
        out_col, out_row, d_out);

    // Input not needed anymore by device
    for (int ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaFree(d_in[ch]));
    }

    // Perform second convolution layer
    std::copy(d_out, d_out + COV2_FILTER_OUT_CH, d_in);
    layer2_cov_multi(COV2_FILTER_N, out_col, out_row, d_in, d_out);

    // Input not needed anymore by device
    for (int ch=0; ch < COV2_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaFree(d_in[ch]));
    }

    // Perform second max pooling on the output from the second convolution layer
    std::copy(d_out, d_out + COV2_FILTER_OUT_CH, d_in);
    layer1_maxPool_multi(out_col, out_row, d_in, 
        out_col, out_row, d_out);

    // Input not needed anymore by device
    for (int ch=0; ch < COV2_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaFree(d_in[ch]));
    }

    // Perform third convolution layer
    std::copy(d_out, d_out + COV3_FILTER_OUT_CH, d_in);
    layer2_cov_multi(COV3_FILTER_N, out_col, out_row, d_in, d_out);

    // Input not needed anymore by device
    for (int ch=0; ch < COV3_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaFree(d_in[ch]));
    }

    // flatten third convolution layer output
    float *d_flattened;
    gpuErrchk( cudaMalloc( (void **) &d_flattened, COV3_FILTER_OUT_CH*out_col*out_row * sizeof(float) ) );
    for(int ch = 0; ch < COV3_FILTER_OUT_CH; ch++) {
        flatten<<<64, 1024>>>(d_out[ch], d_flattened, ch, out_row*out_col);
    }
    
    // linear layers
    // alloc weights and bias memory device side
    float *d_W1;
    float *d_b1;
    float *d_W2;
    float *d_b2;
    float *d_linear_out1;
    float *d_linear_out2;
    gpuErrchk( cudaMalloc( (void **) &d_W1, INPUT_SIZE1 * OUTPUT_SIZE1 * sizeof(float) ) );
    gpuErrchk( cudaMalloc( (void **) &d_b1, OUTPUT_SIZE1 * sizeof(float) ) );
    gpuErrchk( cudaMalloc( (void **) &d_W2, INPUT_SIZE2 * OUTPUT_SIZE2 * sizeof(float) ) );
    gpuErrchk( cudaMalloc( (void **) &d_b2, OUTPUT_SIZE2 * sizeof(float) ) );
    gpuErrchk( cudaMalloc( (void **) &d_linear_out1, OUTPUT_SIZE1 * sizeof(float) ) );
    gpuErrchk( cudaMalloc( (void **) &d_linear_out2, OUTPUT_SIZE2 * sizeof(float) ) );

    // transfer weights and biases to device
    gpuErrchk( cudaMemcpy(d_W1, h_W1, INPUT_SIZE1 * OUTPUT_SIZE1 * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_b1, h_b1, OUTPUT_SIZE1 * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_W2, h_W2, INPUT_SIZE2 * OUTPUT_SIZE2 * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_b2, h_b2, OUTPUT_SIZE2 * sizeof(float), cudaMemcpyHostToDevice) );

    // linear layer 1: input: d_flattened (1x22400) output: (1x256)
    linear<<<64, 1024>>>(d_linear_out1, d_flattened, d_W1, d_b1, INPUT_SIZE1, OUTPUT_SIZE1, false);
    // linear layer 2: input: d_linear_out1 (1x256) output: (1x3)
    linear<<<64, 1024>>>(d_linear_out2, d_linear_out1, d_W2, d_b2, INPUT_SIZE2, OUTPUT_SIZE2, true);
    gpuErrchk( cudaDeviceSynchronize() );

    // copy data back
    gpuErrchk( cudaMemcpy(h_linear_out, d_linear_out2, OUTPUT_SIZE2 * sizeof(float), cudaMemcpyDeviceToHost) );

    // softmax on the output of second linear layer
    h_linear_out = softmax(OUTPUT_SIZE2, h_linear_out);

    double end_time=getTimeStamp();
//================= Timing Ends ========================    
    int total_time_ms = (int)ceil((end_time-start_time)*1000);
    //int constMemFilter_time_ms = (int)ceil((constMemFilter_time - start_time)*1000);
    
// #ifdef PRINTDATA
//     for (int ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
//         gpuErrchk(cudaMemcpy(h_output[ch], d_out[ch], out_col*out_row*sizeof(float), cudaMemcpyDeviceToHost));
//     //}
//     //for (int ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
//         // Need to wait for stream to complete copy
//         gpuErrchk(cudaHostUnregister(h_output[ch]));
        
//         printf("Output ch%d:\n", ch);
//         Print2D(h_output[ch], out_col, out_row);

//         free(h_output[ch]);
//     }
// #endif

#ifdef PRINTDATA
    // print flattened output
    int flattenedBytes = COV3_FILTER_OUT_CH*out_col*out_row*sizeof(float);
    float *h_flattened = (float *) malloc (flattenedBytes);
    gpuErrchk(cudaMemcpy(h_flattened, d_flattened, flattenedBytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < COV3_FILTER_OUT_CH; i++) {
        printf("Output ch%d:\n", i);
        for (int j = 0; j < out_row; j++) {
            for (int k = 0; k < out_col; k++) {
                printf("%f ", h_flattened[k + j*out_col + i*out_row*out_col]);
            }
            printf("\n");
        }
    }    
    printf("\n");

    // print final output
    printf("Softmax output:\n");
    for (int j = 0; j < OUTPUT_SIZE2; j++) {
        printf("%f ", h_linear_out[j]);
    }
    printf("\n");
#endif
    
    for (int ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaFree(d_out[ch]));
    }

    printf("Total Time: %d\n", total_time_ms);
    //printf("Filter Cpy Time: %d\n", constMemFilter_time_ms);

    gpuErrchk(cudaDeviceReset());

    return 0;
}
