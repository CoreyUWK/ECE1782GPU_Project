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

__device__ float relu(float val) {
    return fmaxf(0.0, val);
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

void getConvPadding(int filterSize, int &totalPaddingHeight, 
    int &totalPaddingWidth, int &topPadding, int &leftPadding,
    int &bottomPadding, int &rightPadding) {

    totalPaddingHeight = filterSize - 1;
    totalPaddingWidth = filterSize - 1;
    topPadding = totalPaddingHeight / 2;
    leftPadding = totalPaddingWidth / 2;
    bottomPadding = totalPaddingHeight - topPadding;
    rightPadding = totalPaddingWidth - leftPadding;
}

// Generate Input Data
void initData(float *in, int width, int height, int padding, float *value) {
    int offset;
    for (int i = padding; i < height - padding; ++i) {
        for (int j = padding; j < width - padding; ++j) {
            offset = i * width + j;
            // printf("(%d,%d)=%d ", i, j, offset);
            in[offset] = (value != NULL) ? *value : offset;
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

#define POOL_SIZE 2
#define STRIDE POOL_SIZE
void maxPoolOutSize(int in_rows, int in_cols, int &out_rows, int &out_cols) {
    out_rows = (in_rows - 1) / STRIDE + 1;
    out_cols = (in_cols - 1) / STRIDE + 1;
}