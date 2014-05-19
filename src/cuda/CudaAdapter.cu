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
 * These are some functions to wrap Cuda
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */

/**
 * Throws CudaException if error is not cudaSuccess with the message as the string for the exception and the string for the Cuda error.
 */
inline void handleCudaStatus(cudaError_t error, std::string message) {
  if(error != cudaSuccess){
    message.append(cudaGetErrorString(error));
    throw CudaException(message.c_str());
  }
}

/**
 * Allocate memory for the PRECISION pointer with size number * sizeof(PRECISION) on the GPU, throws CudaException if there is an error
 */
inline void allocateDeviceMemory(void** pointerDevice, int number) {
  handleCudaStatus(cudaMalloc(pointerDevice, number * sizeof(PRECISION)), "Device memory allocation failed: ");
}

/**
 * Allocate pinned memory for the PRECISION pointer with size number * sizeof(PRECISION), throws CudaException if there is an error
 */
inline void allocateHostPinnedMemory(void** pointerDevice, int number) {
  handleCudaStatus(cudaHostAlloc(pointerDevice, number * sizeof(PRECISION), cudaHostAllocPortable),
      "Host pinned memory allocation failed: ");
}

/**
 * Free memory on the GPU, throws CudaException if there is an error
 */
inline void freeDeviceMemory(void* pointerDevice) {
  handleCudaStatus(cudaFree(pointerDevice), "Freeing device memory failed: ");
}

/**
 * Free pinned memory, throws CudaException if there is an error
 */
inline void freePinnedMemory(void* pointerDevice) {
  handleCudaStatus(cudaFreeHost(pointerDevice), "Freeing host memory failed: ");
}

} /* namespace CUDA */
} /* namespace CuEira */

#endif // CUDAADAPTER_H
