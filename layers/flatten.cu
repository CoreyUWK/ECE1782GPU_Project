__global__ void flatten(float *d_conv_out, float *d_flattened, int channel, int size) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x >= size) {
        return;
    }

    int offset = channel*size;

    d_flattened[offset+x] = d_conv_out[x];
}
