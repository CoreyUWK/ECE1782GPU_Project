/* 
To compile and measure time for the 3 convolution layers:
nvcc gemm_conv.cu -o gemm.o -lcublas && ./gemm.o
*/

// Including needed files and libraries
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <sys/time.h>
#include <math.h>
#include <vector>
#include <mutex>
#include <algorithm>
#include <cmath>
#include "utils.cu"

// Defining Cuda check error and safe call macros
#define CUDA_CHECK_ERROR
#define CudaSafeCall(err) __CudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError() __CudaCheckError(__FILE__, __LINE__)

__host__ void __CudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_CHECK_ERROR
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
                cudaGetErrorString(err));
        exit(-1);
    }
#endif
}

__host__ void __CudaCheckError(const char *file, const int line) {
#ifdef CUDA_CHECK_ERROR
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
                cudaGetErrorString(err));
        exit(-1);
    }
#endif
}

// Defining cublas handler
cublasHandle_t cubHandle;
// dummy constants used by cublas
const float alpha = 1.0f;
const float beta = 0.0f;


// input image of size 56 * 100 * 1 should be initialized in the main function
float image[56 * 100 * 1];
// Pointer needed to handle the output of each layer 
float *d_output;

/*
This function is used only for the input image to the CNN
It transform raw_input to the column to be multiplied with the filter matrix 
*/
__global__ void transform_image(float *input, const float *raw_input, const int width,const int height, const int filter_width)
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int start_i = thread_id / width - 1;
    int start_j = thread_id % width - 1;
    int hidden_width = filter_width * filter_width * 1 + 1;
    int global_offset = thread_id * hidden_width;

    int offset = 0;
        for (int i = start_i; i < start_i + filter_width; i++) {
            if (i < 0 || i == height)
                continue;
            for (int j = start_j; j < start_j + filter_width; j++) {
                if (j < 0 || j == width)
                    continue;
                input[global_offset + offset] = raw_input[ i * width + j];
                offset++;
            }
        }
    
    input[(thread_id + 1) * hidden_width - 1] = 1;
}



__global__ void transform(float *input, const float *raw_input, const int width,const int height,const int channels, const int filter_width)
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int start_i = thread_id / width - 1;
    int start_j = thread_id % width - 1;
    int hidden_width = filter_width * filter_width * channels + 1;
    int global_offset = thread_id * hidden_width;

    float relu;
    for (int c = 0; c < channels; c++) {
        int offset = 0;
        for (int i = start_i; i < start_i + filter_width; i++) {
            if (i < 0 || i == height)
                continue;
            for (int j = start_j; j < start_j + filter_width; j++) {
                if (j < 0 || j == width)
                    continue;
                relu = raw_input[(i * width + j) * channels + c];
                input[global_offset + c * filter_width * filter_width + offset] = relu < 0 ? 0 : relu;
                offset++;
            }
        }
    }
    input[(thread_id + 1) * hidden_width - 1] = 1;
}


/*
- inside the convolution function we first pass the dimensions of the input then the number of channels in input
	then the number of output channels and filter width
- based on those numbers, we calculate the number of weights, out size, filter size and hidden width
*/ 
void convolution(int width, int height, int channels, int num_filters, int filter_width)
{
    int num_weights = (filter_width * filter_width * channels + 1) * num_filters;
    int output_size = width * height * num_filters; // because we are doing same convolution
    int filter_size = filter_width * filter_width * channels;
    int hidden_width = filter_width * filter_width * channels + 1;
    float *weights = (float *)malloc(num_weights * sizeof(float)); // including biasses 
    for (int i = 0; i < num_filters; i++) {
        for (int j = 0; j < filter_size; j++)
            weights[j * num_filters + i]=1; // filling weights data with ones host side 
        weights[filter_size * num_filters + i]=1;
    }

    float *d_raw_input; // to hold raw input 
    float *d_input;  // to hold input 
    size_t input_size = width * height * hidden_width * sizeof(float); // expanded input size 
    CudaSafeCall(cudaMalloc(&d_input, input_size)); // allocating device side 
    CudaSafeCall(cudaMemset(d_input, 0, input_size)); // setting by 0 
    // expand original input to (width * height) * (filter_width * filter_width * channels + 1) with a 1 at last for bias
    if (channels == 1) { // for the first layer only we need to copy the image from host to device
        size_t raw_input_size = width * height * channels * sizeof(float);
        CudaSafeCall(cudaMalloc(&d_raw_input, raw_input_size)); // allocating size for row image 
        CudaSafeCall(cudaMemcpy(d_raw_input, image, raw_input_size, cudaMemcpyHostToDevice)); // transfering the image from CPU side
		transform_image <<<width, height>>> (d_input, d_raw_input, width,height, filter_width);
    }
    else  // For all layers expect for the first one
        transform <<< width, height >>> (d_input, d_output, width,height, channels,filter_width);
    CudaCheckError();
    CudaSafeCall(cudaDeviceSynchronize());

    float *d_weights; // pointer on device for weights 
    CudaSafeCall(cudaMalloc(&d_weights, num_weights * sizeof(float))); // allocating on device for weights 
    cudaFree(d_output); // Free output to be used again
    CudaSafeCall(cudaMalloc(&d_output, output_size * sizeof(float))); // allocating for the new out size 
	// next line copies weights from host to device to be used 
    cublasSetMatrix(num_filters, hidden_width, sizeof(float), weights, num_filters, d_weights, num_filters);
    // input * weights = ((width * height) * (filter_width * filter_width * channels + 1)) * ((filter_width * filter_width * channels + 1) * num_filters)
    cublasSgemm(cubHandle, CUBLAS_OP_N, CUBLAS_OP_N, num_filters, width * height, hidden_width,
                            &alpha, d_weights, num_filters, d_input, hidden_width,
                            &beta, d_output, num_filters);

    free(weights);
    if (channels == 1)
        cudaFree(d_raw_input);
    cudaFree(d_input);
    cudaFree(d_weights);
    gpuErrchk(cudaDeviceSynchronize());

}


int main(int argc, char **argv)
{
    // Here we could define the input image to be used
    cublasCreate(&cubHandle);

    double start_time=getTimeStamp();
    // Stacking layers to time them
    // This is done only to get timing performance 
    convolution(56,100, 1, 64,8);    
    convolution(28,50, 64, 64,5);
    convolution(14,25, 64, 64,3);
    double end_time=getTimeStamp();

    float total_time_ms = (end_time-start_time)*1000.0;
    printf("Total Time: %f\n", total_time_ms);

    // Destroying cublas handler
    cublasDestroy(cubHandle);

    return 0;
}
