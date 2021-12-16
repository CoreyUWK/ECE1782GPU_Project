#ifndef CONFIG_KERNEL_H
#define CONFIG_KERNEL_H

#define INPUT_WIDTH 100 // 2048//100
#define INPUT_HEIGHT 56 // 2048//56

// Linear Config
#define INPUT_SIZE1 22400
#define OUTPUT_SIZE1 256
#define INPUT_SIZE2 256
#define OUTPUT_SIZE2 3

// MAXPOOL Config
#define PAD_VALUE -INFINITY
#define MAX_TOL 1e-3
#define POOL_SIZE 2
#define STRIDE POOL_SIZE

#endif
