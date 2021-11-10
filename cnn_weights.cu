#define COV1_FILTER_N 2 // 8x8
#define COV1_FILTER_IN_CH 1
#define COV1_FILTER_OUT_CH 64

#define COV2_FILTER_N 4 // 4x4
#define COV3_FILTER_N 2 // 2x2

__constant__ float device_cov1_b;
__constant__ float device_cov1_filter1[COV1_FILTER_N][COV1_FILTER_N];
__constant__ float device_cov1_filter2[COV1_FILTER_N][COV1_FILTER_N];
__constant__ float device_cov1_filter3[COV1_FILTER_N][COV1_FILTER_N];
__constant__ float device_cov1_filter4[COV1_FILTER_N][COV1_FILTER_N];
__constant__ float device_cov1_filter5[COV1_FILTER_N][COV1_FILTER_N];
__constant__ float device_cov1_filter6[COV1_FILTER_N][COV1_FILTER_N];
__constant__ float device_cov1_filter7[COV1_FILTER_N][COV1_FILTER_N];
__constant__ float device_cov1_filter8[COV1_FILTER_N][COV1_FILTER_N];

float host_cov1_b = 1;
float host_cov1_filter1[COV1_FILTER_N][COV1_FILTER_N] = {
     {1, 1},
     {1, 1}
    /*
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8}*/
};

float host_cov1_filter2[COV1_FILTER_N][COV1_FILTER_N] = {
     {1.0, 1.0},
     {1.0, 1.0}
    /*
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8}*/
};

float host_cov1_filter3[COV1_FILTER_N][COV1_FILTER_N] = {
     {1.0, 1.0},
     {1.0, 1.0}
    /*
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8}*/
};

float host_cov1_filter4[COV1_FILTER_N][COV1_FILTER_N] = {
     {1.0, 1.0},
     {1.0, 1.0}
    /*
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8}*/
};

float host_cov1_filter5[COV1_FILTER_N][COV1_FILTER_N] = {
     {1.0, 1.0},
     {1.0, 1.0}
    /*
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8}*/
};

float host_cov1_filter6[COV1_FILTER_N][COV1_FILTER_N] = {
     {1.0, 1.0},
     {1.0, 1.0}
    /*
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8}*/
};

float host_cov1_filter7[COV1_FILTER_N][COV1_FILTER_N] = {
     {1.0, 1.0},
     {1.0, 1.0}
    /*
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8}*/
};

float host_cov1_filter8[COV1_FILTER_N][COV1_FILTER_N] = {
     {1.0, 1.0},
     {1.0, 1.0}
    /*
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 2, 3, 4, 5, 6, 7, 8}*/
};

