/*
* ECE1782 - Fall 2021 - Lab 2 - Sample Code
* 
* IMPORTANT NOTES:
* 1. For initialization use (((i + j + k)%10) *(float)1.1) [cast to float]
* 2. For kernel use (float)0.8  [cast to float]
* Sample Test Cases (sum)
n: 100   sum: 17861235.145611
n: 200   sum: 145351171.783584
n: 300   sum: 493349760.596508
n: 400   sum: 1172737007.706970
n: 500   sum: 2294392919.237560
n: 600   sum: 3969197501.310867
n: 700   sum: 6308030765.186127
n: 800   sum: 9421772714.654696

nvcc -arch sm_52 -Xptxas="-v" final.cu
*/
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>

//#define EXTR_TIMING 1
//#define PRINTDATA 1
//#define ASSERTRESULTS 1
//#define VALIDATE 1
#define SHAREDMEMORY 1

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

#ifdef SHAREDMEMORY
__global__ void kernel_device(float* in, float *out, float *prevSlice, float *nextSlice, int n) //, int callId
{
    extern __shared__ float sData[];

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

    const int nSMEM = blockDim.x + 3;
    const int rowSMEM = ((threadIdx.y + 1) * nSMEM);
    const int colSMEM = threadIdx.x + 1;
    const int offsetSMEM = rowSMEM + colSMEM;
    
    //printf("%d %d %d %d %d %d %d %d %d\n", threadIdx.x, threadIdx.y, blockDim.x, blockDim.y, rowSMEM, colSMEM, offsetSMEM, x, y);
    sData[offsetSMEM] = in[offset]; // Copy over data in 2D slice that is in block
    if (threadIdx.x == 0) {
        sData[offsetSMEM - 1] = (0 < x) ? in[offset - 1] : 0; // Copy over data just to left of block start
    }
    sData[offsetSMEM + 1] = in[offset + 1]; // Copy over data just to right of block start
    if (threadIdx.y == 0) {
        sData[offsetSMEM - nSMEM] = (0 < y) ? in[offset - n] : 0; // Copy data just above block start
    }
    sData[offsetSMEM + nSMEM] = in[offset + n]; // Copy data just below block end
    __syncthreads();

#ifdef PRINTDATA
    if (callId == 0 && x == n-2 && y == n-2) {
        printf("%d %d\n", x, y);
        for (int m=0; m<(blockDim.y + 2); ++m) {
            for (int j=0; j<nSMEM; ++j) {
                //printf("(%d,%d)=%.3f ", m, j, sData[m * nSMEM + j]);
                printf("%.6f\t", sData[m * nSMEM + j]);
            }
            printf("\n");
        }
    }
#endif
    float output = (NULL != prevSlice) ? prevSlice[offset] : 0; // prev 2D
    output += nextSlice[offset]; // next 2D
    output += sData[offsetSMEM - nSMEM]; //] (0 < y) ? in[offset - n] : 0; // up row
    output += sData[offsetSMEM + nSMEM]; //in[offset + n]; // down row
    output += sData[offsetSMEM - 1]; // (0 < x) ? in[offset - 1] : 0; // left col
    output += sData[offsetSMEM + 1]; //in[offset + 1]; // right col

    //printf("offset:%d %d %p", offset, slice, out);
    out[offset] = magicNum * output;
}
#else
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
#endif


#ifdef VALIDATE
void kernel_host(float *in, float *out, int n) {
    const int size2D = n*n;
    int offset;
    const float magicNum = 0.8;
    for (int k=0, matrix2D=0; k < n-1; ++k, matrix2D+=size2D) {
        for (int i=0, row=0; i < n-1; ++i, row+=n) {
            for (int j=0; j < n-1; ++j) {
                offset = matrix2D + row + j;
                float upRow = (0 < i) ? in[offset - n] : 0; // up row
                float leftCol = (0 < j) ? in[offset - 1] : 0; // left col
                float prev2D = (0 < k) ? in[offset - size2D] : 0; // prev 2D
                out[offset] = magicNum * (
                    prev2D +
                    in[offset + size2D] + // next 2D  
                    upRow + 
                    in[offset + n] + // down row
                    leftCol + // left col
                    in[offset + 1]); // right col
            }
        }
    }
}
#endif

double calculateSum(float *a, int n) {
    // sum over all elements of the cube of  a[i][j][k] * (((i+j+k)%10)?1:-1)
    double sum = 0;
    const int size2D = n*n;
    int offset;
    for (int k=0, matrix2D=0; k < n; ++k, matrix2D += size2D) {
        for (int i=0, row=0; i < n; ++i, row += n) {
            for (int j=0; j < n; ++j) {
                offset = matrix2D + row + j;
                sum += a[offset] * (( (i+j+k) % 10 ) ? 1 :-1);
            }
        }
    }
    return sum;
}

void initData(float *h_b, int n) {
    const int size2D = n*n;
    int offset;
    const float magicNum = 1.1;
    for (int k = 0, matrix2D=0; k < n; ++k, matrix2D+=size2D) {
        for (int i = 0, row=0; i < n; ++i, row+=n) {
            for (int j = 0; j < n; ++j) {
                offset = matrix2D + row + j;
                h_b[offset] = ((k+i+j) % 10) * magicNum;
           }
        }
    }
}

void PrintMatrix(float *m, int n) {
    const int size2D = n*n;
    for (int k = 0, matrix2D=0; k < n; ++k, matrix2D+=size2D) { // Next 2D
        for (int i = 0, row=0; i < n; ++i, row+=n) { // Row
            for (int j = 0; j < n; ++j) { // Col
                printf("%.6f\t", m[matrix2D + row + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
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

int main( int argc, char *argv[])
{    
    if( argc != 2) {
        printf( "Error: wrong number of args\n");
        exit(1);
    }

    int n = atoi(argv[1]);
    size_t number_of_elements = ((size_t)n)*n*n;
    size_t bytes = number_of_elements*sizeof(float);
    int size2D = n*n;
    int size2DBytes = size2D *sizeof(float);

    //TODO: Host memory allocation (not included in timing)
    // alloc memory host-side
    float *h_b = (float *)malloc(bytes); // Input
    if (NULL == h_b) {
        printf("Error: Failed to allocte host memory for B");
        return 1;
    }

    float *h_da = (float *)malloc(bytes); // Device Output
    if (NULL == h_da) {
        printf("Error: Failed to allocte host memory for A used by GPU");
        free(h_b);
        return 1;
    }

    #ifdef VALIDATE
    float *h_ha = (float *)malloc(bytes); // Host Output
    if (NULL == h_ha) {
        printf("Error: Failed to allocte host memory for A");
        free(h_b);
        free(h_da);
        return 1;
    }
    #endif

    //TODO: Matrix initialization (not included in timing)
    initData(h_b, n);
#ifdef PRINTDATA
    printf("Input Data b:\n");
    PrintMatrix(h_b, n);
#endif

#ifdef VALIDATE
    //TODO: Verify the sum is correct (run the CPU version of your code for DEBUG only, disable it when you submit)
    // Check result
    kernel_host(h_b, h_ha, n);
    double sumHost = calculateSum(h_ha, n);
    printf("Host Sum: %lf\n", sumHost);
#endif 

    gpuErrchk(cudaDeviceReset());

    // Pinning host memory so pages are not paged to disk for DMA to work
    gpuErrchk(cudaHostRegister(h_b, bytes, 0));
    gpuErrchk(cudaHostRegister(h_da, bytes, 0));

    dim3 block((n-1 < 32) ? n-1 : 32, (n-1 < 32) ? n-1 : 32); 
    dim3 grid( (n-1 + block.x-1) / block.x, 
               (n-1 + block.y-1) / block.y);
    const size_t sharedMem = (block.x + 3) * (block.y + 2) * sizeof(float);

    // Setup streams -> found if created stream per z=n, at n=800 run out of memory
    const int NumStream = 200; //(n + 1 < 17) ? n + 1 : 17;
    float *d_a[n]; float *d_b[n];
    int queueA[n]; int endA = 0; queueA[endA] = -1; int lookupA[n]; lookupA[0] = -1;
    int queueB[n]; int endB = 0; queueB[endB] = -1; int lookupB[n]; lookupB[0] = -1;
    cudaStream_t stream[NumStream];
    const int cpyStream = 1;
    cudaEvent_t events[n];

    Params params[n];
    for (int i = 0; i<n;++i) {
        params[i].queueA = queueA; params[i].endA = &endA; params[i].lookupA = lookupA;
        params[i].queueB = queueB; params[i].endB = &endB; params[i].lookupB = lookupB;
    }

    for (int i = 1; i < NumStream; ++i) {
        gpuErrchk(cudaStreamCreate(&stream[i]));
    }

    for (int i = 0; i < n; ++i) {
        gpuErrchk(cudaEventCreate(&events[i]));
    }

//================= Timing Begins ========================
    double start_time=getTimeStamp();

    #ifdef EXTR_TIMING
    double alloc_time = getTimeStamp();
    #endif

    /*Device allocations are included in timing*/
    
    //TODO: Data transfer from host to device
    #ifdef EXTR_TIMING
    double cpy_to_time = getTimeStamp();
    #endif

    //TODO: Kernel call
    #ifdef EXTR_TIMING
    double kernel_time = getTimeStamp();
    #endif

    for (int i = 0, offset=0; i < n; ++i, offset += size2D) {
        // Get Memory for B
        if (queueB[endB] == -1) {
            gpuErrchk(cudaMalloc((void **)&d_b[i], size2DBytes));
            lookupB[i] = i;
        }
        else {
            lookupB[i] = queueB[endB];
            queueB[endB] = -1;
            if (0 < endB) --endB;
        }

        // Copy over B
        gpuErrchk(cudaMemcpyAsync(d_b[lookupB[i]], &h_b[offset], size2DBytes, cudaMemcpyHostToDevice, stream[cpyStream]));
        gpuErrchk(cudaEventRecord(events[i], stream[cpyStream]));

        // Process previous slice once current B's has been copied (above)
        if (0 < i) {
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