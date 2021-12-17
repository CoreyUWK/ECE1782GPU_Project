#ifndef ALLOC_HELPERS_H
#define ALLOC_HELPERS_H

float *allocHostBlock(int bytes);
float *allocDeviceBlock(int bytes);

float *allocHostBlockHelper(std::vector<float *> &h_freeBlocks,
                            std::mutex &h_freeBlocksMutex, int bytes);

float *allocDeviceBlockHelper(std::vector<float *> &d_freeBlocks,
                              std::mutex &d_freeBlocksMutex, int bytes);
#endif
