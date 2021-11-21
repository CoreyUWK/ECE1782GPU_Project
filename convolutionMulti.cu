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

//#define PRINTDATA 1
#define SHMEM 1
//#define DebugSHMEM 1
//#define DebugSHMEM_Data 1

#define INPUT_WIDTH 100//2048//100
#define INPUT_HEIGHT 56//2048//56

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

__constant__ float *device_output[COV1_FILTER_OUT_CH];
__device__ void device_CNN_Multi(float *inCh, int filterSize) {
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
    
    int cnnOffset = offset_GM - topPadding*INPUT_WIDTH - leftPadding;
    float cacheIn[COV1_FILTER_N * COV1_FILTER_N];
    for (int i = 0, row=0, inOffset=cnnOffset; i < filterSize; ++i, row += filterSize, cnnOffset += INPUT_WIDTH) {
        for (int j = 0; j < filterSize; ++j) {
            //int offset = cnnOffset + i * INPUT_WIDTH + j;
            if ((y - topPadding + i < 0) || (x - leftPadding + j < 0) || 
                (y - topPadding + i >= INPUT_HEIGHT) || (x - leftPadding + j >= INPUT_WIDTH)) {
                cacheIn[row + j] = 0;
            }               
            else {
                cacheIn[row + j] = inCh[cnnOffset + j];
            }
        }
    }

    int filterChOffset = 0;
    for (int ch=0; ch < 64; ++ch, filterChOffset += COV1_FILTER_N*COV1_FILTER_N) {
        //TODO: reduce repeated computations by storing
        //Maybe make loop condition handle outside matrix area instead of continue
        float conv = 0.0;
        int filterOffset = filterChOffset;
        for (int i = 0, row=0; i < filterSize; ++i, filterOffset += filterSize, row += COV1_FILTER_N) {
            for (int j = 0; j < filterSize; ++j) {
                conv += cacheIn[row + j] * device_cov1_filter[0][ch][i][j]; //[filterOffset + j];
            }
        }
        /*if (x == 0 && y == 0) {
            printf("Addr%d: %p %p ", ch, device_output[ch], &device_output[ch]);
        }*/
        device_output[ch][offset_GM] = conv + device_cov1_b[ch];
    }
}

#ifdef SHMEM
__device__ void device_CNN_Multi_SHMEM(float *inCh, int filterSize,
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

    const int offset_GM = y * INPUT_WIDTH + x;

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
            inCh[offset_GM - 2*leftPadding - 2*topPadding*INPUT_WIDTH];
#endif
        }
        // Set left-overs on bottom left corner
        else if (x >= 2*leftPadding && (y >= (INPUT_HEIGHT - 3*bottomPadding) || threadIdx.y >= (blockDim.y - 3*bottomPadding)) && y < (INPUT_HEIGHT - 2*bottomPadding) &&
            leftPadding <= threadIdx.x && threadIdx.x < 2*leftPadding) {
            sInCh[shmemRow_plus_x - leftPadding + 2*rightPadding*shmemWidth] = 
#ifdef DebugSHMEM_Data
            8;
#else
            inCh[offset_GM - 2*leftPadding  + 2*bottomPadding*blockDim.x];
#endif            
        }
    }

    if (x >= INPUT_WIDTH || y >= INPUT_HEIGHT) {
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
        inCh[offset_GM - topPadding*INPUT_WIDTH];
#endif
    }
    // Set bottom padding using all threads in bottomPadding number of rows
    else if (y < (INPUT_HEIGHT - bottomPadding) && threadIdx.y >= (blockDim.y - bottomPadding)) { //blockIdx.y != lastY => could pass # blocks to kernel or use static #define size -> maybe helps performance try
        sInCh[shmemRow_pos_padded + bottomPadding * shmemWidth] = 
#ifdef DebugSHMEM_Data
        3;
#else
        inCh[offset_GM + bottomPadding*INPUT_WIDTH];
#endif
    }
    // Use remaining threads for left-over area (left, right, corners + top/bottom padding extra on sides)
    // left-over threads = INPUT_HEIGHT - topPadding - bottomPadding 
    else if (threadIdx.y >= topPadding && threadIdx.y < (blockDim.y - bottomPadding)) { // this could be an else
        // Set Left padding 
        if (y < (INPUT_HEIGHT - bottomPadding) && x >= leftPadding && threadIdx.x < leftPadding) {
            sInCh[shmemRow_plus_x] = 
#ifdef DebugSHMEM_Data
            4;
#else
            inCh[offset_GM - leftPadding];
#endif
        }
        // Set Right padding
        else if (y < (INPUT_HEIGHT - bottomPadding) && x < (INPUT_WIDTH - rightPadding) && threadIdx.x >= blockDim.x - rightPadding) {
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
        else if (x <= (INPUT_WIDTH - 2*rightPadding) && y >= 2*topPadding &&
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
        else if (x <= (INPUT_WIDTH - 2*rightPadding) && 
            (y >= (INPUT_HEIGHT - 3*bottomPadding) || threadIdx.y >= (blockDim.y - 3*bottomPadding)) && y < (INPUT_HEIGHT - 2*bottomPadding) &&
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
    for (int i=0, row=0, shmemRowOffset=cnnOffset, filterRowOffset=0; i < filterSize; ++i, row += filterSize, shmemRowOffset += shmemWidth, filterRowOffset += filterSize) {
        for (int j = 0; j < filterSize; ++j) {
            cacheIn[row + j] = sInCh[shmemRowOffset + j];
        }
    }

    int filterChOffset = 0;
    for (int ch=0; ch < 64; ++ch, filterChOffset += COV1_FILTER_N*COV1_FILTER_N) {
        //TODO: reduce repeated computations by storing
        //Maybe make loop condition handle outside matrix area instead of continue
        float conv = 0.0;
        int filterOffset = filterChOffset;
        for (int i = 0, row=0; i < filterSize; ++i, filterOffset += filterSize, row += COV1_FILTER_N) {
            for (int j = 0; j < filterSize; ++j) {
                conv += cacheIn[row + j] * device_cov1_filter[0][ch][i][j]; //[filterOffset + j];
            }
        }
        /*if (x == 0 && y == 0) {
            printf("Addr%d: %p %p ", ch, device_output[ch], &device_output[ch]);
        }*/
        device_output[ch][offset_GM] = conv + device_cov1_b[ch];
    }
}
#endif


__global__ void kernel_multi(float *inCh, int filterSize,
    int totalPaddingHeight, int totalPaddingWidth, int topPadding, int bottomPadding, int leftPadding, int rightPadding) {
#ifdef SHMEM
    device_CNN_Multi_SHMEM(inCh, filterSize,
        totalPaddingHeight, totalPaddingWidth, topPadding, bottomPadding, leftPadding, rightPadding);
#else  
    device_CNN_Multi(inCh, filterSize);
#endif
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

float* allocHostBlock(int bytes) {
    float *mem = NULL;
    mem = (float *)malloc(bytes);
    return mem;
}

float* allocDeviceBlock(int bytes) {
    float *mem = NULL;
    gpuErrchk(cudaMalloc((void **)&mem, bytes));
    return mem;
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

    const int totalPaddingHeight = COV1_FILTER_N - 1;
    const int totalPaddingWidth = COV1_FILTER_N - 1;
    const int topPadding = totalPaddingHeight / 2;
    const int leftPadding = totalPaddingWidth / 2;
    const int bottomPadding = totalPaddingHeight - topPadding;
    const int rightPadding = totalPaddingWidth - leftPadding;

#ifdef SHMEM
    const int shmemSize = (INPUT_HEIGHT + totalPaddingHeight) * (INPUT_WIDTH + totalPaddingWidth) * sizeof(float);
    kernel_multi<<<grid, block, shmemSize>>>(d_input, COV1_FILTER_N, 
        totalPaddingHeight, totalPaddingWidth, topPadding, bottomPadding, leftPadding, rightPadding);   
#else
    kernel_multi<<<grid, block>>>(d_input, COV1_FILTER_N,
                totalPaddingHeight, totalPaddingWidth, topPadding, bottomPadding, leftPadding, rightPadding);
#endif
    gpuErrchk(cudaDeviceSynchronize());


    // If want output then need to copy back to host memory
    //gpuErrchk(cudaMemcpyAsync(&out[i], &d_output[i], bytes, cudaMemcpyDeviceToHost, streams[i]));

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
    initData(h_input, INPUT_WIDTH, INPUT_HEIGHT, 0);
    float *d_input = allocDeviceBlock(bytes);
    if (d_input == NULL) {
        printf("Error: Failed to allocte host memory for input");
        return 1;
    }

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

//================= Timing Begins ========================
    double start_time=getTimeStamp();

    // Perform First convolution layer
    float *d_cov1_out[COV1_FILTER_OUT_CH];
    layer1_cov1_multi(bytes, grid, block, h_input, d_input, d_cov1_out);

    // Input Not needed anymore by device
    gpuErrchk(cudaHostUnregister(h_input));
    free(h_input);

#ifdef PRINTDATA
    for (int ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaMemcpy(h_output[ch], d_cov1_out[ch], bytes, cudaMemcpyDeviceToHost));
    }
    for (int ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
        // Need to wait for stream to complete copy
        gpuErrchk(cudaHostUnregister(h_output[ch]));
        
        printf("Output ch%d:\n", ch);
        Print2D(h_output[ch], INPUT_WIDTH, INPUT_HEIGHT);

        free(h_output[ch]);
    }

#endif

    double end_time=getTimeStamp();
//================= Timing Ends ========================    
    int total_time_ms = (int)ceil((end_time-start_time)*1000);
    //int constMemFilter_time_ms = (int)ceil((constMemFilter_time - start_time)*1000);
    
    printf("Total Time: %d\n", total_time_ms);
    //printf("Filter Cpy Time: %d\n", constMemFilter_time_ms);

    gpuErrchk(cudaDeviceReset());

    return 0;
}