/* This will not work as no way to sync threads in grid
__device__ void device_CNN_inline(float *inCh, float *filter, int filterSize) {
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

    //TODO: reduce repeated computations by storing
    //Maybe make loop condition handle outside matrix area instead of continue
    int cnnOffset = offset_GM - topPadding*INPUT_WIDTH - leftPadding;
    float conv = 0;
    for (int i = 0; i < filterSize; ++i) {
        if (y - topPadding + i < 0) continue;
        if (y - topPadding + i >= INPUT_HEIGHT) {
            //printf("%d %d %d\n", y, topPadding, i);
            break;
        }
        for (int j = 0; j < filterSize; ++j) {
            int offset = cnnOffset + i * INPUT_WIDTH + j;
            if (x - leftPadding + j < 0) continue;
            if (x - leftPadding + j >= INPUT_WIDTH) break;
            conv += inCh[cnnOffset + i * INPUT_WIDTH + j] * filter[i *
filterSize + j];
            //printf("%d %d\n", i, j);
        }
    }

    __syncthreads();
    inCh[offset_GM] = conv;
}*/
