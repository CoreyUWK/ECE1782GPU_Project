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

#define INPUT_WIDTH 56
#define INPUT_HEIGHT 100

#define COV1_FILTER_N 8 // 8x8
#define COV2_FILTER_N 4 // 4x4
#define COV3_FILTER_N 2 // 2x2


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

__constant__ float cov1_filter[COV1_FILTER_N]
__constant__ float cov2_filter[COV2_FILTER_N]
__constant__ float cov3_filter[COV3_FILTER_N]

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
__device__ void device_CNN(float *in, float *out, float *filter, int filterSize) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int offset = y * INPUT_WIDTH + x; 
    const int padding = filterSize / 2;

    float conv = 0;
    for (int i = 0; i < filterSize; ++i) {
        for (int j = 0; j < filterSize; ++j) {
            if ()

            cov += in[offset + ] * cov1_filter[i * filterSize + j];
        }
    }
}

__global__ void kernel_device(float* in, float *out, float *prevSlice, float *nextSlice, int n) 
{
    // Get positions in GMEM
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x >= n-1 || y >= n-1) {
        return;
    }

    const int row = (y * n);
    const int col = x;
    const int offset = row + col;
    const float magicNum = 0.8;
    float output = (NULL != prevSlice) ? prevSlice[offset] : 0; // prev 2D
    output += nextSlice[offset]; // next 2D
    output += (0 < y) ? in[offset - n] : 0; // up row
    output += in[offset + n]; // down row
    output += (0 < x) ? in[offset - 1] : 0; // left col
    output += in[offset + 1]; // right col

    //printf("offset:%d %d %p", offset, slice, out);
    out[offset] = magicNum * output;
}


struct Params {
    int *queueA; int *endA; int *lookupA;
    int *queueB; int *endB; int *lookupB;
    int i; int prevI;
};

void callbackProcessFinished(cudaStream_t stream, cudaError_t status, void *arg) {
    Params *params = (Params*)arg;
    int i = params->i;

    // Reuse memory - TODO: from threads could have contention            
    if (-1 != params->queueA[*(params->endA)]) ++(*(params->endA)); 
    params->queueA[*(params->endA)] = params->lookupA[params->prevI];

    // prev b is done as it has been copied to next(current)
    if (1 < i) {
        if (-1 != params->queueB[*(params->endB)]) ++(*(params->endB));
        params->queueB[*(params->endB)] = params->lookupB[i-2];
    }
}

// Generate Input Data
void initData(float *h_input) {
    int offset;
    const float magicNum = 1.1;

    for (int i = 0, row=0; i < INPUT_HEIGHT; ++i, row += INPUT_WIDTH) {
        for (int j = 0; j < INPUT_WIDTH; ++j) {
            offset = row + j;
            h_b[offset] = ((i+j) % 10) * magicNum; //TODO: Update value to be accurate
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


int main( int argc, char *argv[])
{    
    size_t input_num_elements = INPUT_WIDTH * INPUT_HEIGHT;
    size_t bytes = input_num_elements * sizeof(float);

    // Host memory allocation (not included in timing)
    // alloc memory host-side
    float *h_input = (float *)malloc(bytes); // Input
    if (NULL == h_input) {
        printf("Error: Failed to allocte host memory for input");
        return 1;
    }

    float *h_da = (float *)malloc(bytes); // Device Output
    if (NULL == h_da) {
        printf("Error: Failed to allocte host memory for A used by GPU");
        free(h_b);
        return 1;
    }

    // Input initialization (not included in timing)
    initData(h_input);
#ifdef PRINTDATA
    printf("Input:\n");
    Print2D(h_input, INPUT_WIDTH, INPUT_HEIGHT);
#endif

    gpuErrchk(cudaDeviceReset());

    // Pinning host memory so pages are not paged to disk for DMA to work
    gpuErrchk(cudaHostRegister(h_input, bytes, 0));

    dim3 block((INPUT_WIDTH-1 < 32) ? INPUT_WIDTH-1 : 32, (INPUT_HEIGHT-1 < 32) ? INPUT_HEIGHT-1 : 32); 
    dim3 grid( (INPUT_WIDTH-1 + block.x-1) / block.x, 
               (INPUT_HEIGHT-1 + block.y-1) / block.y);

    // Setup streams
    const int CNN_Layers = 5;
    const int Cov_Channels = 64;
    const int NumStream = 200; //(n + 1 < 17) ? n + 1 : 17;
    const int NumMatrixPerCNN = 1 + Cov_Channels*5; // Number of input sized matrix used in layer
    // TODO: Figure out how many matrix of input size is reused (note: I believe all layers of CNN use same size) 
    // TODO: Maybe make this more dynamic with std::vector or linked list
    float *d_input[NumMatrixPerCNN]; 
    int queue[NumMatrixPerCNN]; int end = 0; queue[end] = -1; int lookup[NumMatrixPerCNN]; lookup[0] = -1;
    cudaStream_t stream[NumStream];
    const int cpyToDeviceStream = 1;
    cudaEvent_t events[NumMatrixPerCNN];

    Params params[NumMatrixPerCNN];
    for (int i = 0; i<n;++i) {
        params[i].queue = queue; params[i].end = &end; params[i].lookup = lookup;
    }

    for (int i = 1; i < NumStream; ++i) {
        gpuErrchk(cudaStreamCreate(&stream[i]));
    }

    for (int i = 0; i < NumMatrixPerCNN; ++i) {
        gpuErrchk(cudaEventCreate(&events[i]));
    }

//================= Timing Begins ========================
    double start_time=getTimeStamp();

    // Get Memory for Data Input
    gpuErrchk(cudaMalloc((void **)&d_input[0], bytes));
    lookup[0] = 0;
    // Copy over input
    gpuErrchk(cudaMemcpyAsync(d_input[lookup[0]], h_input, bytes, cudaMemcpyHostToDevice, stream[cpyToDeviceStream]));
    // All streams on next layer per CNN needs to wait for input data
    // gpuErrchk(cudaEventRecord(events[0], stream[cpyToDeviceStream]));  
    /*for (int i = 1; i < NumStream; ++i) {
        gpuErrchk(cudaStreamSynchronize(stream[i]));
    }*/
    gpuErrchk(cudaDeviceSynchronize());

    for (int layer = 0; layer < CNN_Layers; ++layer) {
        for (int i = 1, offset=0; i <= Cov_Channels; ++i, offset += size2D) {
            // Get Memory for Layer channel process
            if (queue[end] == -1) {
                gpuErrchk(cudaMalloc((void **)&d_input[i], bytes));
                lookup[i] = i;
            }
            else {
                lookup[i] = queue[end];
                queue[end] = -1;
                if (0 < end) --end;
            }

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

    free(h_b);
    free(h_da);
    #ifdef VALIDATE
    free(h_ha);
    #endif

    return 0;
}