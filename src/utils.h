#ifndef UTILS_H
#define UTILS_H

/*You can use the following for any CUDA function that returns cudaError_t
 * type*/
#define gpuErrchk(ans)                                                         \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true);

/*Use the following to get a timestamp*/
double getTimeStamp();

__device__ float relu(float val);

float *allocHostBlock(int bytes);

float *allocDeviceBlock(int bytes);

void getConvPadding(int filterSize, int &totalPaddingHeight,
                    int &totalPaddingWidth, int &topPadding, int &leftPadding,
                    int &bottomPadding, int &rightPadding);

// Generate Input Data
void initData(float *in, int width, int height, int padding, float *value);

void Print2D(float *m, int width, int height);
void maxPoolOutSize(int in_rows, int in_cols, int &out_rows, int &out_cols);

void initBias(float *A, int n);

void initWeights(float *W, int inSize, int outSize);

#endif
