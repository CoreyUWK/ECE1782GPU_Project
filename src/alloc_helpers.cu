#include "config.h"
#include "utils.h"
#include <mutex>
#include <vector>

float *allocHostBlock(int bytes) {
    float *mem = NULL;
    mem = (float *)malloc(bytes);
    return mem;
}

float *allocDeviceBlock(int bytes) {
    float *mem = NULL;
    gpuErrchk(cudaMalloc((void **)&mem, bytes));
    return mem;
}

float *allocHostBlockHelper(std::vector<float *> &h_freeBlocks,
                            std::mutex &h_freeBlocksMutex, int bytes) {
    float *mem = NULL;

#ifdef EnableLock
    h_freeBlocksMutex.lock();
#endif
    if (!h_freeBlocks.empty()) {
        mem = h_freeBlocks.back();
        h_freeBlocks.pop_back();
    }
#ifdef EnableLock
    h_freeBlocksMutex.unlock();
#endif
    if (mem == NULL) {
        mem = allocHostBlock(bytes);
    }

    return mem;
}

float *allocDeviceBlockHelper(std::vector<float *> &d_freeBlocks,
                              std::mutex &d_freeBlocksMutex, int bytes) {
    float *mem = NULL;

#ifdef EnableLock
    d_freeBlocksMutex.lock();
#endif
    if (!d_freeBlocks.empty()) {
        mem = d_freeBlocks.back();
        d_freeBlocks.pop_back();
    }
#ifdef EnableLock
    d_freeBlocksMutex.unlock();
#endif
    if (mem == NULL) {
        mem = allocDeviceBlock(bytes);
    }

    return mem;
}
