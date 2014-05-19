#ifndef CUDAADAPTER_H
#define CUDAADAPTER_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>

#include <CudaException.h>
#include <CublasException.h>
#include <DimensionMismatch.h>

namespace CuEira {
namespace CUDA {

/**
 * This is some functions to wrap Cuda and Cublas
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */

/**
 * Throws CudaException if error is not cudaSuccess with the message as the string for the exception and the string for the Cuda error.
 */
void handleCudaStatus(cudaError_t error, std::string message);

/**
 * Allocate memory for the PRECISION pointer with size number * sizeof(PRECISION) on the GPU
 */
void allocateDeviceMemory(void** pointerDevice, int number);

/**
 * Allocate pinned memory for the PRECISION pointer with size number * sizeof(PRECISION)
 */
void allocateHostPinnedMemory(void** pointerDevice, int number);

/**
 * Free memory on the GPU
 */
void freeDeviceMemory(void* pointerDevice);

/**
 * Free pinned memory
 */
void freePinnedMemory(void* pointerDevice);

} /* namespace CUDA */
} /* namespace CuEira */

#endif // CUDAADAPTER_H
