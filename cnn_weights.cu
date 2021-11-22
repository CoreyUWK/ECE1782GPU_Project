// NOTE: filter size doesn't seem to affect performance, tried 3 vs 8 and got same time
#define COV1_FILTER_N 8 // 8x8
#define COV1_FILTER_IN_CH 1
#define COV1_FILTER_OUT_CH 64 // Can set to 1 when debugging shared memory and printing data in kernel

#define COV2_FILTER_N 2 // 4x4
#define COV2_FILTER_IN_CH 64
#define COV2_FILTER_OUT_CH 64

#define COV3_FILTER_N 2 // 2x2
#define COV3_FILTER_IN_CH 64
#define COV3_FILTER_OUT_CH 64

__constant__ float device_cov1_b[COV1_FILTER_OUT_CH];
__constant__ float device_cov1_filter[COV1_FILTER_IN_CH][COV1_FILTER_OUT_CH][COV1_FILTER_N][COV1_FILTER_N];

// To much constant memory > 64KB
//__constant__ float device_cov2_b[COV2_FILTER_OUT_CH];
//__constant__ float device_cov2_filter[COV2_FILTER_IN_CH][COV2_FILTER_OUT_CH][COV2_FILTER_N][COV2_FILTER_N];

//__constant__ float device_cov3_b[COV3_FILTER_OUT_CH];
//__constant__ float device_cov3_filter[COV3_FILTER_IN_CH][COV3_FILTER_OUT_CH][COV3_FILTER_N][COV3_FILTER_N];


float host_cov1_b[COV1_FILTER_OUT_CH];
float host_cov1_filter[COV1_FILTER_IN_CH][COV1_FILTER_OUT_CH][COV1_FILTER_N][COV1_FILTER_N] = {
    { 
        { {1, 1}, {1, 1} }
    }
    
    /*{1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8}*/
};

float host_cov2_filter[COV2_FILTER_IN_CH][COV2_FILTER_OUT_CH][COV2_FILTER_N][COV2_FILTER_N] = {
    { 
        { {1, 1}, {1, 1} }
    }
    
    /*{1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8}*/
};

