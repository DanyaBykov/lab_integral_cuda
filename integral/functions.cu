#include "functions.cuh"
#include <cmath>
#include <cuda_runtime.h>

__device__ double atomicAddD(double* address, double value) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ double function_1 (const double x1, const double x2) {
    double result = 0.002;
    for (int i = -2; i < 3; i++) {
        for (int j = -2; j < 3; j++) {
            result += 1.0/(5*(i+2) + j + 3 + powf(x1 - 16*j, 6) + powf(x2 - 16*i, 6));
        }
    }
    result = powf(result, -1);
    return result;
}

__device__ double function_2 (const double x1, const double x2) {
    const double a = 20;
    const double b = 0.2;
    const double c = 2 * M_PI;
    double result = 0.0;
    result -= a * exp(-b * sqrt(0.5 * (powf(x1, 2) + powf(x2, 2))));
    result -= exp(0.5 * ((cos(c * x1) + cos(c * x2))));
    result += a + exp(1);
    return result;
}

__device__ double function_3 (const double x1, const double x2) {
    const int m = 5;
    const int a1[] = {1, 2, 1, 1, 5};
    const int a2[] = {4, 5, 1, 2, 4};
    const int c[] = {2, 1, 4, 7, 2};
    double res = 0.0;
    for (size_t i = 0; i < m; i++) {
        res += c[i] * exp(-1 * M_1_PI*(powf(x1 - a1[i], 2) + powf(x2 - a2[i], 2))) *
        cos(M_PI*(powf(x1 - a1[i], 2) + powf(x2 - a2[i], 2)));
    }

    return -1 * res;
}

__global__ void integrate_func1(const double x1_start, const double dx1, const double x2_start, const double dx2, const int x1_steps, const int x2_steps, double* d_result) {
    __shared__ double local_sum[256];
    int count = threadIdx.x + threadIdx.y * blockDim.x;
    local_sum[count] = 0.0;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < x1_steps && j < x2_steps) {
        double x1 = x1_start + (i + 0.5) * dx1;
        double x2 = x2_start + (j + 0.5) * dx2;
        local_sum[count] = function_1(x1, x2) * dx1 * dx2;
    }
    __syncthreads();

    if (count == 0) {
        double blockSum = 0.0;
        for (int k = 0; k < blockDim.x * blockDim.y; ++k) {
            blockSum += local_sum[k];
        }
        atomicAddD(d_result, blockSum);
    }
}

__global__ void integrate_func2(const double x1_start, const double dx1, const double x2_start, const double dx2, const int x1_steps, const int x2_steps, double* d_result) {
    __shared__ double local_sum[256];
    int count = threadIdx.x + threadIdx.y * blockDim.x;
    local_sum[count] = 0.0;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < x1_steps && j < x2_steps) {
        double x1 = x1_start + (i + 0.5) * dx1;
        double x2 = x2_start + (j + 0.5) * dx2;
        local_sum[count] = function_2(x1, x2) * dx1 * dx2;
    }
    __syncthreads();

    if (count == 0) {
        double blockSum = 0.0;
        for (int k = 0; k < blockDim.x * blockDim.y; ++k) {
            blockSum += local_sum[k];
        }
        atomicAddD(d_result, blockSum);
    }
}

__global__ void integrate_func3(const double x1_start, const double dx1, const double x2_start, const double dx2, const int x1_steps, const int x2_steps, double* d_result) {
    __shared__ double local_sum[256];
    int count = threadIdx.x + threadIdx.y * blockDim.x;
    local_sum[count] = 0.0;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < x1_steps && j < x2_steps) {
        double x1 = x1_start + (i + 0.5) * dx1;
        double x2 = x2_start + (j + 0.5) * dx2;
        local_sum[count] = function_3(x1, x2) * dx1 * dx2;
    }
    __syncthreads();

    if (count == 0) {
        double blockSum = 0.0;
        for (int k = 0; k < blockDim.x * blockDim.y; ++k) {
            blockSum += local_sum[k];
        }
        atomicAddD(d_result, blockSum);
    }
}