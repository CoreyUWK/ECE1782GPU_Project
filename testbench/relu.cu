#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>

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

void initData(float *M, int nRows, int nCols) {
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            int offset = (i * nCols) + j;
            M[offset] = (i+j) % 2 == 0 ? -(i+j) : (i+j);
        }
    }
}

void printMatrix(float *M, int nRows, int nCols) {
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            int offset = i * nCols + j;
            printf("%d,%d:%f   ", i,j, M[offset]);
        }
        printf("\n");
    }
}

void host_relu(float *X, float *h_hX, int nRows, int nCols) {
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            int offset = i * nCols + j;
            h_hX[offset] = max(0.0, X[offset]);
        }
    }
}

__global__ void relu(float *X, int nRows, int nCols) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if ((col >= nCols) || (row >= nRows)) {
        return;
    }

    int offset = row * nCols + col;

    X[offset] = fmaxf(0.0, X[offset]);;
}

int main( int argc, char *argv[] ) {

    if (argc != 3) {
       printf( "Please pass 2 arguments to the program: the first being the number of rows and the second being the number of columns.\n" ); 
       return -1;
    }

    // set matrix size
    int nRows = atoi(argv[1]);
    int nCols = atoi(argv[2]);
    // printf("Num rows =  %d, num cols = %d \n", nRows, nCols);
    int nElements = nRows * nCols;
    int bytes = nElements * sizeof(float);

    // alloc memory host-side
    float *h_X = (float *) malloc (bytes);
    float *h_dX = (float *) malloc (bytes);
    float *h_hX = (float *) malloc (bytes); // host result

    cudaHostRegister(h_X, bytes, 0);
    cudaHostRegister(h_dX, bytes, 0);

    // init data
    initData(h_X, nRows, nCols);

    // printMatrix(h_X, nRows, nCols);
    // printf("\n");

    gpuErrchk(cudaDeviceReset());

    double start_time = getTimeStamp();
    
    // alloc memory device side
    float *d_X;
    gpuErrchk( cudaMalloc( (void **) &d_X, bytes ) );

	// transfer data to device
    gpuErrchk( cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice) );

	// invoke kernel
    dim3 block(32, 32); // configure
    dim3 grid((nCols+block.x-1)/block.x, (nRows+block.y-1)/block.y);
    relu<<<grid, block>>>(d_X, nRows, nCols);
    gpuErrchk( cudaDeviceSynchronize() );

    // copy data back
    gpuErrchk( cudaMemcpy(h_dX, d_X, bytes, cudaMemcpyDeviceToHost) );

    double end_time = getTimeStamp();
    int total_time_ms = (int) ceil ((end_time-start_time)*1000);

    // check result
    host_relu(h_X, h_hX, nRows, nCols);
    // printMatrix(h_hX, nRows, nCols);
    // printf("\n");
    // printMatrix(h_dX, nRows, nCols);
    for (int i = 0; i < nElements; i++) {
        if (h_hX[i] != h_dX[i]) {
            printf("Error: CPU result and GPU result mismatch at offset: %d.\n", i);
            return 0;
        }
    }

    printf("%d\n", total_time_ms);

    cudaHostUnregister(h_X);
    cudaHostUnregister(h_dX);

    // free gpu resources
    gpuErrchk( cudaFree(d_X) );
    gpuErrchk( cudaDeviceReset() );

	return 0;
}