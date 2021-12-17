#include "../cnn_weights.h"
#include "../utils.h"

__constant__ float *device_output[COV1_FILTER_OUT_CH];

#ifdef SHMEM
__global__ void device_CNN_Multi_SHMEM(
    int in_cols, int in_rows, float *inCh, int filterSize, bool isSingle,
    bool isFirst, bool isLast, int totalPaddingHeight, int totalPaddingWidth,
    int topPadding, int bottomPadding, int leftPadding, int rightPadding) {
    /* Shared Memory Layout:
    000...000
    000...000
    000111000 <- place block of inCh here, but will have to place padding area
    for block edges handling 000111000 000111000 000111000 000...000 000...000
    */
    extern __shared__ float sData[];
    float *sInCh = sData;

    // Position relative to global memory of 2D matrix
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    const int offset_GM = y * in_cols + x;

    // SHMEM offsets (includes padding)
    int shmemWidth = totalPaddingWidth + blockDim.x;
    int shmemRow = (threadIdx.y + topPadding) * shmemWidth;
    int shmemRow_plus_x = shmemRow + threadIdx.x;
    int shmemRow_pos_padded = shmemRow_plus_x + leftPadding;

    // Initialize shared memory to zero - when block is on edge of grid
    // TODO: try to make more efficent by including in below code if checks
    // As here threads are doing duplicate work
    sInCh[shmemRow_pos_padded - topPadding * shmemWidth - leftPadding] =
#ifdef DebugSHMEM_Data
        -1;
#else
        0;
#endif
    sInCh[shmemRow_pos_padded - topPadding * shmemWidth] =
#ifdef DebugSHMEM_Data
        -1;
#else
        0;
#endif
    sInCh[shmemRow_pos_padded - topPadding * shmemWidth + rightPadding] =
#ifdef DebugSHMEM_Data
        -1;
#else
        0;
#endif
    sInCh[shmemRow_pos_padded + bottomPadding * shmemWidth - leftPadding] =
#ifdef DebugSHMEM_Data
        -1;
#else
        0;
#endif
    sInCh[shmemRow_pos_padded + bottomPadding * shmemWidth] =
#ifdef DebugSHMEM_Data
        -1;
#else
        0;
#endif
    sInCh[shmemRow_pos_padded + bottomPadding * shmemWidth + rightPadding] =
#ifdef DebugSHMEM_Data
        -1;
#else
        0;
#endif
    __syncthreads();

    // Now setup Shared memory with data
    if (threadIdx.y >= topPadding &&
        threadIdx.y < (blockDim.y - bottomPadding)) { // this could be an else
        // Set left-overs on top left corner
        if (x >= 2 * leftPadding &&
            y >=
                2 * topPadding && // Basically not block 0 (but if checking
                                  // blockIdx would have to split this into two)
            threadIdx.y < 3 * topPadding &&
            threadIdx.x < 2 * leftPadding && leftPadding <= threadIdx.x) {
            sInCh[shmemRow_plus_x - leftPadding - 2 * topPadding * shmemWidth] =
#ifdef DebugSHMEM_Data
                6;
#else
                inCh[offset_GM - 2 * leftPadding - 2 * topPadding * in_cols];
#endif
        }
        // Set left-overs on bottom left corner
        else if (x >= 2 * leftPadding &&
                 (y >= (in_rows - 3 * bottomPadding) ||
                  threadIdx.y >= (blockDim.y - 3 * bottomPadding)) &&
                 y < (in_rows - 2 * bottomPadding) &&
                 leftPadding <= threadIdx.x && threadIdx.x < 2 * leftPadding) {
            sInCh[shmemRow_plus_x - leftPadding +
                  2 * rightPadding * shmemWidth] =
#ifdef DebugSHMEM_Data
                8;
#else
                inCh[offset_GM - 2 * leftPadding +
                     2 * bottomPadding * blockDim.x];
#endif
        }
    }

    if (x >= in_cols || y >= in_rows) {
        return;
    }

    // Every Thread in block copies it's value
    sInCh[shmemRow_pos_padded] =
#ifdef DebugSHMEM_Data
        1;
#else
        inCh[offset_GM];
#endif
    // Set top padding using all threads in topPadding number of rows
    if (blockIdx.y != 0 && threadIdx.y < topPadding) { // y >= topPadding
        sInCh[shmemRow_pos_padded - topPadding * shmemWidth] =
#ifdef DebugSHMEM_Data
            2;
#else
            inCh[offset_GM - topPadding * in_cols];
#endif
    }
    // Set bottom padding using all threads in bottomPadding number of rows
    else if (y < (in_rows - bottomPadding) &&
             threadIdx.y >=
                 (blockDim.y -
                  bottomPadding)) { // blockIdx.y != lastY => could pass #
                                    // blocks to kernel or use static #define
                                    // size -> maybe helps performance try
        sInCh[shmemRow_pos_padded + bottomPadding * shmemWidth] =
#ifdef DebugSHMEM_Data
            3;
#else
            inCh[offset_GM + bottomPadding * in_cols];
#endif
    }
    // Use remaining threads for left-over area (left, right, corners +
    // top/bottom padding extra on sides) left-over threads = INPUT_HEIGHT -
    // topPadding - bottomPadding
    else if (threadIdx.y >= topPadding &&
             threadIdx.y <
                 (blockDim.y - bottomPadding)) { // this could be an else
        // Set Left padding
        if (y < (in_rows - bottomPadding) && x >= leftPadding &&
            threadIdx.x < leftPadding) {
            sInCh[shmemRow_plus_x] =
#ifdef DebugSHMEM_Data
                4;
#else
                inCh[offset_GM - leftPadding];
#endif
        }
        // Set Right padding
        else if (y < (in_rows - bottomPadding) &&
                 x < (in_cols - rightPadding) &&
                 threadIdx.x >= blockDim.x - rightPadding) {
            sInCh[shmemRow_pos_padded + rightPadding] =
#ifdef DebugSHMEM_Data
                5;
#else
                inCh[offset_GM + rightPadding];
#endif
        }
        // Set left-overs on top left corner
        /*else if (x >= 2*leftPadding && y >= 2*topPadding && // Basically not
        block 0 (but if checking blockIdx would have to split this into two)
            threadIdx.y < 3*topPadding && threadIdx.x < 2*leftPadding &&
            leftPadding <= threadIdx.x) {
            sInCh[shmemRow_plus_x - leftPadding - 2*topPadding*shmemWidth] =
        6;//in[offset_GM - 2*leftPadding - 2*topPadding*INPUT_WIDTH];
        }*/
        // Set left-overs on top right corner
        else if (x <= (in_cols - 2 * rightPadding) && y >= 2 * topPadding &&
                 threadIdx.y < 3 * topPadding &&
                 (blockDim.x - rightPadding) >= threadIdx.x &&
                 threadIdx.x >= (blockDim.x - 2 * rightPadding)) {
            sInCh[shmemRow_pos_padded + 2 * rightPadding -
                  2 * topPadding * shmemWidth] =
#ifdef DebugSHMEM_Data
                7;
#else
                inCh[offset_GM + 2 * rightPadding -
                     2 * topPadding * blockDim.x];
#endif
        }
        // Set left-overs on bottom left corner
        /*else if (x >= 2*leftPadding && (y >= (INPUT_HEIGHT - 3*bottomPadding)
        || threadIdx.y >= (blockDim.y - 3*bottomPadding)) && y < (INPUT_HEIGHT -
        2*bottomPadding) && leftPadding <= threadIdx.x && threadIdx.x <
        2*leftPadding) { sInCh[shmemRow_plus_x - leftPadding +
        2*rightPadding*shmemWidth] = 8;
        }*/
        // Set left-overs on bottom right corner
        else if (x <= (in_cols - 2 * rightPadding) &&
                 (y >= (in_rows - 3 * bottomPadding) ||
                  threadIdx.y >= (blockDim.y - 3 * bottomPadding)) &&
                 y < (in_rows - 2 * bottomPadding) &&
                 (blockDim.x - rightPadding) >= threadIdx.x &&
                 threadIdx.x >= (blockDim.x - 2 * rightPadding)) {
            sInCh[shmemRow_pos_padded + 2 * rightPadding +
                  2 * bottomPadding * shmemWidth] =
#ifdef DebugSHMEM_Data
                9;
#else
                inCh[offset_GM + 2 * rightPadding +
                     2 * bottomPadding * blockDim.x];
#endif
        }
    }

    __syncthreads(); // TODO: try only syncing threads used in filter area (or
                     // see if helps with performance)

#ifdef DebugSHMEM
    // printf("%d,%d->%d\n", x,y, shmemRow_pos_padded);
    if (blockIdx.x == 3 && blockIdx.y == 1 && threadIdx.x == 0 &&
        threadIdx.y == 0) {
        printf("Top: %d, Left:%d Right:%d Bottom:%d\n", topPadding, leftPadding,
               rightPadding, bottomPadding);
        printf("SHMEM:\n");
        for (int i = 0, row = 0; i < totalPaddingHeight + blockDim.y;
             ++i, row += shmemWidth) {
            for (int j = 0; j < shmemWidth; ++j) {
                // printf("%d,%d=%f ", i,j, sInCh[row + j]);
                printf("%.0f\t", sInCh[row + j]);
            }
            printf("\n");
        }
    }
#endif

    int cnnOffset = shmemRow_plus_x - topPadding * shmemWidth;
    float cacheIn[COV1_FILTER_N * COV1_FILTER_N];
    float *f = &device_cov1_filter[0][0][0][0];
    int totalFilterSize = filterSize * filterSize;
    int i = 0;
    for (int row = 0, shmemRowOffset = cnnOffset; i < filterSize;
         ++i, row += filterSize, shmemRowOffset += shmemWidth) {
        for (int j = 0; j < filterSize; ++j) {
            cacheIn[row + j] = sInCh[shmemRowOffset + j];
        }
    }

    for (int ch = 0, filterChOffset = 0; ch < COV1_FILTER_OUT_CH;
         ++ch, filterChOffset += totalFilterSize) {
        float conv = 0.0;
        for (i = 0; i < totalFilterSize; ++i) {
            conv += cacheIn[i] * *(f + filterChOffset + i);
        }
        if (isSingle) {
            device_output[ch][offset_GM] = relu(conv + device_cov1_b[ch]);
        } else if (isFirst) {
            device_output[ch][offset_GM] = conv;
        } else if (isLast) {
            conv += device_output[ch][offset_GM] + device_cov1_b[ch];
            device_output[ch][offset_GM] = relu(conv);
        } else {
            device_output[ch][offset_GM] += conv;
        }
    }
}
#else
// template <int filterSize> // Can't template cause multiple functions with
// some at 77 registers
__global__ void device_CNN_Multi_v1(int in_cols, int in_rows, float *inCh,
                                    int filterSize, bool isSingle, bool isFirst,
                                    bool isLast, int totalPaddingHeight,
                                    int totalPaddingWidth, int topPadding,
                                    int bottomPadding, int leftPadding,
                                    int rightPadding) {
    // Position relative to global memory of 2D matrix
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= in_cols || y >= in_rows) {
        return;
    }
    int totalFilterSize = filterSize * filterSize;

    const int offset_GM = y * in_cols + x;
    int cnnOffset = offset_GM - topPadding * in_cols - leftPadding;
    float cacheIn[COV1_FILTER_N *
                  COV1_FILTER_N]; // NOTE requires constant size arg, else
                                  // compiler complains (so constant memory),
                                  // could make template
    int boundryH = y - topPadding;
    int boundryW = x - leftPadding;
    int i = 0;
    for (int row = 0; i < filterSize;
         ++i, row += filterSize, cnnOffset += in_cols, ++boundryH) {
        for (int j = 0; j < filterSize; ++j) {
            if (boundryH < 0 || boundryH >= in_rows || (boundryW + j < 0) ||
                (boundryW + j >= in_cols)) {
                cacheIn[row + j] = 0;
            } else {
                cacheIn[row + j] = inCh[cnnOffset + j];
            }
        }
    }

    float *f = &device_cov1_filter[0][0][0][0];
    for (int ch = 0, filterChOffset = 0; ch < COV1_FILTER_OUT_CH;
         ++ch, filterChOffset += totalFilterSize) {
        float conv = 0.0;
        for (i = 0; i < totalFilterSize; ++i) {
            conv += cacheIn[i] * *(f + filterChOffset + i);
        }
        if (isSingle) {
            device_output[ch][offset_GM] = relu(conv + device_cov1_b[ch]);
        } else if (isFirst) {
            device_output[ch][offset_GM] = conv;
        } else if (isLast) {
            conv += device_output[ch][offset_GM] + device_cov1_b[ch];
            device_output[ch][offset_GM] = relu(conv);
        } else {
            device_output[ch][offset_GM] += conv;
        }
    }
}

__global__ void device_CNN_Multi_v2(int in_cols, int in_rows, float *inCh,
                                    int filterSize, bool isSingle, bool isFirst,
                                    bool isLast, int totalPaddingHeight,
                                    int totalPaddingWidth, int topPadding,
                                    int bottomPadding, int leftPadding,
                                    int rightPadding) {
    // Position relative to global memory of 2D matrix
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= in_cols || y >= in_rows) {
        return;
    }

    const int offset_GM = y * in_cols + x;
    int cnnOffset = offset_GM - topPadding * in_cols - leftPadding;

    // float covResult[64]; // To many registers so have to use GMEM of output
    float readVal;
    int boundryHeight = y - topPadding;
    int boundryWidth = x - leftPadding;
    int filterTotalSize = filterSize * filterSize;
    float *f = &device_cov1_filter[0][0][0][0];
    int ch, filterChOffset;

    for (int i = 0, row = 0; i < filterSize;
         ++i, row += filterSize, cnnOffset += in_cols, ++boundryHeight) {
        if (boundryHeight < 0 || boundryHeight >= in_rows)
            continue;
        for (int j = 0; j < filterSize; ++j) {
            if ((boundryWidth + j < 0) || (boundryWidth + j >= in_cols))
                continue;

            readVal = inCh[cnnOffset + j];
            for (ch = 0, filterChOffset = 0; ch < 64;
                 ++ch, filterChOffset += filterTotalSize) {
                // covResult[ch] += readVal * *(f + filterChOffset + row + j);
                // covResult[ch] = __fmaf_rd(readVal, *(f + filterChOffset + row
                // + j), covResult[ch]);
                if (i == 0 && isFirst) {
                    device_output[ch][offset_GM] =
                        readVal * *(f + filterChOffset + row + j);
                } else {
                    device_output[ch][offset_GM] +=
                        readVal * *(f + filterChOffset + row + j);
                }
            }
        }
    }

    if (isLast) {
        for (ch = 0; ch < COV1_FILTER_OUT_CH; ++ch) {
            device_output[ch][offset_GM] =
                relu(device_output[ch][offset_GM] + device_cov1_b[ch]);
            /*else if (isFirst) {
                device_output[ch][offset_GM] = device_output[ch][offset_GM];
            }*/
            /*else if (isLast) {
                covResult[ch] += device_output[ch][offset_GM] +
            device_cov1_b[ch]; device_output[ch][offset_GM] =
            relu(covResult[ch]);
            }*/
            /*else {
                device_output[ch][offset_GM] += covResult[ch];
            }*/
        }
    }
}
#endif
