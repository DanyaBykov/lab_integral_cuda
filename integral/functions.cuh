#ifndef FUNCTIONS_CUH
#define FUNCTIONS_CUH

#include <cuda_runtime.h>

__device__ double atomicAddD(double* address, double value);

__device__ double function_1(const double x1, const double x2);
__device__ double function_2(const double x1, const double x2);
__device__ double function_3(const double x1, const double x2);

__global__ void integrate(int function_id, const double x1_start, const double dx1, const double x2_start, const double dx2, const int x1_steps, const int x2_steps, double* d_result);

#endif 