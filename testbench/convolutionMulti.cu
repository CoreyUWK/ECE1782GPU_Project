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
#include <cmath>
//#include "cublas_v2.h"
#include "../src/cnn_weights.h"
#include "../src/config.h"
#include "../src/utils.cu"
#include "../src/layers/conv_2d.cu"
#include "../src/layers/flatten.cu"
#include "../src/layers/linear.cu"
#include "../src/layers/max_pool_2d.cu"
#include "../src/layers/softmax.cu"


// Currently a thread per pooling, but thread no reading coalesed
// could read coalesed by copying to shared memory and then reorder in shared memory linearly



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
