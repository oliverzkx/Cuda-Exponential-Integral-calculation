#pragma once

/**
 * @brief Launches CUDA kernel to compute exponential integral using float precision.
 * 
 * @param n Order up to which the exponential integrals are computed.
 * @param m Number of sample points in the interval [a, b].
 * @param a Left endpoint of the interval.
 * @param b Right endpoint of the interval.
 * @param output Pointer to host memory to store results (flattened n×m array).
 */
void launchKernelFloat(int n, int m, float a, float b, float* output, bool timing);



/**
 * @brief Launches CUDA kernel to compute exponential integral using double precision.
 * 
 * @param n Order up to which the exponential integrals are computed.
 * @param m Number of sample points in the interval [a, b].
 * @param a Left endpoint of the interval.
 * @param b Right endpoint of the interval.
 * @param output Pointer to host memory to store results (flattened n×m array).
 */
void launchKernelDouble(int n, int m, double a, double b, double* output, bool timing);
