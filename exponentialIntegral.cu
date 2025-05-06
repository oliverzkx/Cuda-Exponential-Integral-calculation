#include "exponentialIntegral.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

/**
 * @brief Device function to compute exponential integral approximation using float.
 * @param n Order of the exponential integral.
 * @param x Input value.
 * @return Approximate value of E_n(x).
 */
__device__ float expIntFloat(int n, float x) {
    return expf(-x) / x;
}

/**
 * @brief Device function to compute exponential integral approximation using double.
 * @param n Order of the exponential integral.
 * @param x Input value.
 * @return Approximate value of E_n(x).
 */
__device__ double expIntDouble(int n, double x) {
    return exp(-x) / x;
}

/**
 * @brief CUDA kernel for float exponential integral evaluation.
 * @param n Maximum order.
 * @param m Number of samples per order.
 * @param a Interval start.
 * @param b Interval end.
 * @param output Flattened output array of size n × m.
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
 * @brief CUDA kernel for double exponential integral evaluation.
 * @param n Maximum order.
 * @param m Number of samples per order.
 * @param a Interval start.
 * @param b Interval end.
 * @param output Flattened output array of size n × m.
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
 * @brief Host launcher for float precision exponential integral kernel with timing.
 * @param n Number of orders.
 * @param m Number of samples.
 * @param a Start of interval.
 * @param b End of interval.
 * @param output Host-side output array (flattened).
 * @param timing Whether to print timing info.
 */
void launchKernelFloat(int n, int m, float a, float b, float* output, bool timing) {
    int total = n * m;
    float* d_output = nullptr;
    cudaMalloc(&d_output, total * sizeof(float));

    int threadsPerBlock = 256;
    int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernelFloat<<<blocks, threadsPerBlock>>>(n, m, a, b, d_output);
    cudaMemcpy(output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (timing) {
        std::cout << "[Timing] GPU (float): " << milliseconds << " ms" << std::endl;
    }

    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/**
 * @brief Host launcher for double precision exponential integral kernel with timing.
 * @param n Number of orders.
 * @param m Number of samples.
 * @param a Start of interval.
 * @param b End of interval.
 * @param output Host-side output array (flattened).
 * @param timing Whether to print timing info.
 */
void launchKernelDouble(int n, int m, double a, double b, double* output, bool timing) {
    int total = n * m;
    double* d_output = nullptr;
    cudaMalloc(&d_output, total * sizeof(double));

    int threadsPerBlock = 256;
    int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernelDouble<<<blocks, threadsPerBlock>>>(n, m, a, b, d_output);
    cudaMemcpy(output, d_output, total * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (timing) {
        std::cout << "[Timing] GPU (double): " << milliseconds << " ms" << std::endl;
    }

    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
