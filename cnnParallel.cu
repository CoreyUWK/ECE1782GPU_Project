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
#include "cnn_weights.cu"

//#define PRINTDATA 1 // Print Ouput Results
//#define EnableLock 1
//#define GET_TIMING_BREAKDOWN 1 // Enable CNN timing breakdown print

//#define SHMEM 1   // Enable shared memory for convolution
//#define DebugSHMEM 1
//#define DebugSHMEM_Data 1

// Input Matrix Dimensions
#define INPUT_WIDTH 100//2048
#define INPUT_HEIGHT 56//2048

#define NUM_STREAM 64+64

// MAXPOOL Config
#define PAD_VALUE -INFINITY
#define MAX_TOL 1e-3
#define POOL_SIZE 2
#define STRIDE POOL_SIZE // 1

// Linear Config
#define ENABLE_LINEAR_LAYER 1
#if STRIDE == POOL_SIZE
#define INPUT_SIZE1 22400 // 64x(14x25)
#else // STRIDE == 1
#define INPUT_SIZE1 358400 // 64x(56x100)
#endif
#define OUTPUT_SIZE1 256
#define INPUT_SIZE2 256
#define OUTPUT_SIZE2 3

#include "utils.cu"

// Currently a thread per pooling, but thread no reading coalesed
// could read coalesed by copying to shared memory and then reorder in shared memory linearly
__global__ void max_pool_2d(float *in, int in_rows, int in_cols, float *out, int out_rows, int out_cols,
        int px_pre, int py_pre) {
    unsigned int o_col = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int o_row = blockDim.y * blockIdx.y + threadIdx.y;

    float out_element = PAD_VALUE;
    float current_element;
    unsigned int addr;

    if (o_col >= out_cols || o_row >= out_rows) {
        return;
    }

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

__global__ void flatten(float *d_conv_out, float *d_flattened, int channel, int size) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x >= size) {
        return;
    }

    int offset = channel*size;

    d_flattened[offset+x] = d_conv_out[x];
}

__global__ void linear(float *output, float *input, float *W, float *b, int inSize, int outSize, bool isFinal) {
    // Get output index/row = what to calculate for
    int outNeuron = blockDim.x * blockIdx.x + threadIdx.x; 

    // This shouldn't get hit as passing exact
    if (outNeuron >= outSize) {
        return;
    }

    /* Weight Layout:
    Threads read first row, then next iteration threads read next row, ...
         v v v v      v
    Out: 1 2 3 4 ... 256
    IN:  1 2 3 4 ... 256
         ... ... ... ...
         Last Input Row

    This allows for threads to read memory contigously */
    float sum = 0.0;
    for (int i = 0, offset = outNeuron; i < inSize; ++i, offset += outSize) {
        sum += input[i] * W[offset];
    }

    // final linear layer don't use relu
    if (isFinal) {
        output[outNeuron] = sum + b[outNeuron];
    }
    else {
        output[outNeuron] = relu(sum + b[outNeuron]);
    }
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
    if (NULL != host_cov_b) {
        gpuErrchk( cudaMemcpyToSymbolAsync(device_cov1_b, host_cov_b, size, 
            ch*size, cudaMemcpyHostToDevice, stream) );
    }

    if (NULL != host_cov_filter) {
        size = COV1_FILTER_N*COV1_FILTER_N*sizeof(float);
        gpuErrchk( cudaMemcpyToSymbolAsync(device_cov1_filter, host_cov_filter, size, 
            ch * size, cudaMemcpyHostToDevice, stream) );
    }
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
        if (i > 0) // input cpy and first filter run on same stream so skip on first stream 
            gpuErrchk(cudaStreamWaitEvent(streams[i], event));

        // Setup all filters and b
        setupFilterCh(i, NULL, &host_cov1_filter[0][i][0][0], streams[i]);

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

    // Implement padding=same from tensorflow
    int px_pre = (in_cols % STRIDE == 0) ? max(POOL_SIZE - STRIDE, 0) : max(POOL_SIZE - (in_cols % STRIDE), 0);
    int py_pre = (in_rows % STRIDE == 0) ? max(POOL_SIZE - STRIDE, 0) : max(POOL_SIZE - (in_rows % STRIDE), 0);
    px_pre /= 2;
    py_pre /= 2;

    // Get output memory
    for (int ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaMalloc((void **)&d_output[ch], out_bytes));
        max_pool_2d<<<grid, block, 0, streams[ch]>>>(d_input[ch], in_rows, in_cols, d_output[ch], out_rows, out_cols,
            px_pre, py_pre);
    }

    //gpuErrchk(cudaDeviceSynchronize());
    /*for (int ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaStreamSynchronize(streams[ch]));
        gpuErrchk(cudaStreamDestroy(streams[ch]));
    }*/
}

void layer_covMulti(int in_cols, int in_rows, cudaStream_t *streams, cudaEvent_t *events,
    float **d_input, float **d_output, int filterSize, 
    float * host_cov_filter, float *host_cov_b) {

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

    float *host_filterAddr = host_cov_filter;
    int totalFilterSize = filterSize*filterSize; 
    int totalInChSize = totalFilterSize * COV2_FILTER_OUT_CH;
    for (int inCh=0; inCh < COV2_FILTER_IN_CH; ++inCh) {
        gpuErrchk(cudaEventRecord(events[inCh], streams[inCh]));

        for (int outCh=0; outCh < COV2_FILTER_OUT_CH; ++outCh, host_filterAddr += totalFilterSize) {
            // Every stream needs to wait for input
            gpuErrchk(cudaStreamWaitEvent(streams[COV2_FILTER_IN_CH + outCh], events[inCh]));

            // Setup all filters and b
            setupFilterCh(outCh, NULL, host_filterAddr, streams[COV2_FILTER_IN_CH+outCh]);

            // Get output memory
            if (inCh == 0) {
                gpuErrchk(cudaMalloc((void **)&d_output[outCh], bytes));
            }
            device_CNN<<<grid, block, 0, streams[COV2_FILTER_IN_CH + outCh]>>>(in_cols, in_rows, d_input[inCh], d_output[outCh], 
                host_cov_b[outCh], filterAddr + outCh*COV1_FILTER_N*COV1_FILTER_N, filterSize,
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

    std::fill(&host_cov2_b[0], &host_cov2_b[0] + COV2_FILTER_OUT_CH, 1.0);
    std::fill(&host_cov2_filter[0][0][0][0], &host_cov2_filter[0][0][0][0] + COV2_FILTER_IN_CH * COV2_FILTER_OUT_CH * COV2_FILTER_N * COV2_FILTER_N, 1.0);
    
    std::fill(&host_cov3_b[0], &host_cov3_b[0] + COV3_FILTER_OUT_CH, 1.0);
    std::fill(&host_cov3_filter[0][0][0][0], &host_cov3_filter[0][0][0][0] + COV3_FILTER_IN_CH * COV3_FILTER_OUT_CH * COV3_FILTER_N * COV3_FILTER_N, 1.0);

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

    float *d_in[COV1_FILTER_OUT_CH]; 
    float *d_out[COV1_FILTER_OUT_CH];
    for (int i=0; i < COV1_FILTER_OUT_CH; ++i) {
        d_out[i] = NULL;
        gpuErrchk(cudaEventCreate(&events[i]));
        gpuErrchk(cudaEventCreate(&events[i + 64]));
    }

#ifdef ENABLE_LINEAR_LAYER
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
    std::fill(h_W1, h_W1 + INPUT_SIZE1 * OUTPUT_SIZE1, 1.0);
    std::fill(h_b1, h_b1 + OUTPUT_SIZE1, 1.0);
    std::fill(h_W2, h_W2 + INPUT_SIZE2 * OUTPUT_SIZE2, 1.0);
    std::fill(h_b2, h_b2 + OUTPUT_SIZE2, 1.0);
    float *d_W1;
    float *d_b1;
    float *d_W2;
    float *d_b2;
    float *d_linear_out1;
    float *d_linear_out2;
#endif
//================= Timing Begins ========================
    double start_time=getTimeStamp();
    
    float *d_input = allocDeviceBlock(bytes);
    if (d_input == NULL) {
        printf("Error: Failed to allocte host memory for input");
        return 1;
    }

#ifdef GET_TIMING_BREAKDOWN
    double in_alloc_time=getTimeStamp();
#endif
    
    layer_cov(INPUT_WIDTH, INPUT_HEIGHT, streams, events[0],
        h_input, d_input, d_out);

#ifdef GET_TIMING_BREAKDOWN
    gpuErrchk(cudaDeviceSynchronize());
    double cov1_time=getTimeStamp();
#endif

#ifdef Free_Memory
    // Input Not needed anymore by device
    gpuErrchk(cudaHostUnregister(h_input));
    free(h_input);
    gpuErrchk(cudaFree(d_input));
#endif 

#ifdef GET_TIMING_BREAKDOWN
    double free_input_time=getTimeStamp();
#endif

    // Perform first max pooling on the output from the first convolution layer
    std::copy(d_out, d_out + COV1_FILTER_OUT_CH, d_in);
    layer_maxPool(INPUT_WIDTH, INPUT_HEIGHT, d_in, 
        out_col, out_row, d_out, streams);

#ifdef GET_TIMING_BREAKDOWN
    gpuErrchk(cudaDeviceSynchronize());
    double maxpool1_time=getTimeStamp();
#endif

#ifdef Free_Memory
    // Input not needed anymore by device
    for (int ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaFree(d_in[ch]));
    }
#endif

#ifdef GET_TIMING_BREAKDOWN
    double free_cov1_time=getTimeStamp();
#endif

    // Perform second convolution layer
    std::copy(d_out, d_out + COV2_FILTER_OUT_CH, d_in);
    layer_covMulti(out_col, out_row, streams, events,
        d_in, d_out, COV2_FILTER_N, 
        &host_cov2_filter[0][0][0][0], host_cov2_b);

#ifdef GET_TIMING_BREAKDOWN
    gpuErrchk(cudaDeviceSynchronize());
    double cov2_time=getTimeStamp();
#endif

#ifdef Free_Memory
    // Input not needed anymore by device
    for (int ch=0; ch < COV2_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaFree(d_in[ch]));
    }
#endif 

#ifdef GET_TIMING_BREAKDOWN
    double free_maxpool1_time=getTimeStamp();
#endif

    // Perform second max pooling on the output from the second convolution layer
    std::copy(d_out, d_out + COV2_FILTER_OUT_CH, d_in);
    layer_maxPool(out_col, out_row, d_in, 
        out_col, out_row, d_out, &streams[64]);

#ifdef GET_TIMING_BREAKDOWN
    gpuErrchk(cudaDeviceSynchronize());
    double maxpool2_time=getTimeStamp();
#endif

#ifdef Free_Memory
    // Input not needed anymore by device
    for (int ch=0; ch < COV2_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaFree(d_in[ch]));
        std::swap(streams[ch], streams[COV2_FILTER_OUT_CH + ch]);
    }
#endif 

#ifdef GET_TIMING_BREAKDOWN
    double free_cov2_time=getTimeStamp();
#endif

    // Perform third convolution layer
    std::copy(d_out, d_out + COV3_FILTER_OUT_CH, d_in);
    layer_covMulti(out_col, out_row, streams, &events[64], 
        d_in, d_out, COV3_FILTER_N,
        &host_cov3_filter[0][0][0][0], host_cov3_b);

#ifdef GET_TIMING_BREAKDOWN
    gpuErrchk(cudaDeviceSynchronize());
    double cov3_time=getTimeStamp();
#endif

#ifdef Free_Memory
    // Input not needed anymore by device
    for (int ch=0; ch < COV3_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaFree(d_in[ch]));
    }
#endif 

#ifdef GET_TIMING_BREAKDOWN
    double free_maxpool2_time=getTimeStamp();
#endif

    gpuErrchk(cudaDeviceSynchronize());

#ifdef ENABLE_LINEAR_LAYER
    // flatten third convolution layer output
    float *d_flattened;
    int flattenOutSize = out_col*out_row;
    gpuErrchk( cudaMalloc( (void **) &d_flattened, COV3_FILTER_OUT_CH * flattenOutSize * sizeof(float) ) );
    
    dim3 block((flattenOutSize < 1024) ? flattenOutSize : 1024); 
    dim3 grid( (flattenOutSize + block.x-1) / block.x);

    for(int ch = 0; ch < COV3_FILTER_OUT_CH; ch++) {
        flatten<<<grid, block>>>(d_out[ch], d_flattened, ch, flattenOutSize);
    }

#ifdef GET_TIMING_BREAKDOWN
    gpuErrchk( cudaDeviceSynchronize() );
    double flatten_time=getTimeStamp();
#endif

    // linear layers
    // alloc weights and bias memory device side
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

#ifdef GET_TIMING_BREAKDOWN
    double linear_layer_setup=getTimeStamp();
#endif

    // linear layer 1: input: d_flattened (1x22400) output: (1x256)
    linear<<<1, OUTPUT_SIZE1>>>(d_linear_out1, d_flattened, d_W1, d_b1, INPUT_SIZE1, OUTPUT_SIZE1, false);
    // linear layer 2: input: d_linear_out1 (1x256) output: (1x3)
    linear<<<1, OUTPUT_SIZE2>>>(d_linear_out2, d_linear_out1, d_W2, d_b2, INPUT_SIZE2, OUTPUT_SIZE2, true);

    gpuErrchk( cudaDeviceSynchronize() );

#ifdef GET_TIMING_BREAKDOWN
    double linear_layer_time=getTimeStamp();
#endif

    // Need to wait for stream to complete copy
    for (int i = 0; i < NUM_STREAM; ++i) {
        gpuErrchk(cudaStreamSynchronize(streams[i]));
        gpuErrchk(cudaStreamDestroy(streams[i]));
    }

    // copy data back
    gpuErrchk( cudaMemcpy(h_linear_out, d_linear_out2, OUTPUT_SIZE2 * sizeof(float), cudaMemcpyDeviceToHost) );

    // softmax on the output of second linear layer
    h_linear_out = softmax(OUTPUT_SIZE2, h_linear_out);
#endif

    double end_time=getTimeStamp();
//================= Timing Ends ========================    
    float total_time_ms = (end_time-start_time)*1000.0;
#ifdef GET_TIMING_BREAKDOWN
    float alloc_input_ms = (in_alloc_time-start_time)*1000.0;
    float cov1_time_ms = (cov1_time-in_alloc_time)*1000.0;
    float maxpool1_time_ms = (maxpool1_time-free_input_time)*1000.0;
    float cov2_time_ms = (cov2_time-free_cov1_time)*1000.0;
    float maxpool2_time_ms = (maxpool2_time-free_maxpool1_time)*1000.0;
    float cov3_time_ms = (cov3_time-free_cov2_time)*1000.0;
    float linear_layer_ms = (linear_layer_time-free_maxpool2_time)*1000.0;
#endif

#ifdef PRINTDATA
#ifdef ENABLE_LINEAR_LAYER
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
#else

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
#endif

    printf("Total Time: %f\n", total_time_ms);
#ifdef GET_TIMING_BREAKDOWN
    printf("Cov1 Time: %f\n", cov1_time_ms);
    printf("Maxpool1 Time: %f\n", maxpool1_time_ms);
    printf("Cov2 Time: %f\n", cov2_time_ms);
    printf("Maxpool2 Time: %f\n", maxpool2_time_ms);
    printf("Cov3 Time: %f\n", cov3_time_ms);
    printf("Linear Layer Time: %f\n", linear_layer_ms);
#endif

    // Clean up device blocks
    for (int ch=0; ch < COV1_FILTER_OUT_CH; ++ch) {
        gpuErrchk(cudaFree(d_out[ch]));
        gpuErrchk(cudaEventDestroy(events[ch]));
        gpuErrchk(cudaEventDestroy(events[ch + 64]));
    }

    gpuErrchk(cudaDeviceReset());

    return 0;
}