#include "exponentialIntegral.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

/**
 * @brief Device function for float exponential integral approximation.
 * 
 * @param n Order of the integral.
 * @param x Evaluation point.
 * @return float Approximated exponential integral.
 */
__device__ float expIntFloat(int n, float x) {
    return expf(-x) / x;  // Simplified for testing; replace with full algorithm later.
}

/**
 * @brief Device function for double exponential integral approximation.
 * 
 * @param n Order of the integral.
 * @param x Evaluation point.
 * @return double Approximated exponential integral.
 */
__device__ double expIntDouble(int n, double x) {
    return exp(-x) / x;
}

/**
 * @brief CUDA kernel for computing float exponential integrals.
 */
__global__ void kernelFloat(int n, int m, float a, float b, float* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * m) {
        int order = i / m + 1;
        int sample = i % m + 1;
        float dx = (b - a) / m;
        float x = a + sample * dx;
        output[i] = expIntFloat(order, x);
    }
}

/**
 * @brief CUDA kernel for computing double exponential integrals.
 */
__global__ void kernelDouble(int n, int m, double a, double b, double* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * m) {
        int order = i / m + 1;
        int sample = i % m + 1;
        double dx = (b - a) / m;
        double x = a + sample * dx;
        output[i] = expIntDouble(order, x);
    }
}

/**
 * @brief Host launcher for float exponential integral CUDA kernel.
 */
void launchKernelFloat(int n, int m, float a, float b, float* output) {
    int total = n * m;
    float* d_output = nullptr;

    cudaMalloc(&d_output, total * sizeof(float));
    
    int threadsPerBlock = 256;
    int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;

    kernelFloat<<<blocks, threadsPerBlock>>>(n, m, a, b, d_output);

    cudaMemcpy(output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_output);
}

/**
 * @brief Host launcher for double exponential integral CUDA kernel.
 */
void launchKernelDouble(int n, int m, double a, double b, double* output) {
    int total = n * m;
    double* d_output = nullptr;

    cudaMalloc(&d_output, total * sizeof(double));

    int threadsPerBlock = 256;
    int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;

    kernelDouble<<<blocks, threadsPerBlock>>>(n, m, a, b, d_output);

    cudaMemcpy(output, d_output, total * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_output);
}
