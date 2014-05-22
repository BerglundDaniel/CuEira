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
 * Convert Cublas error enum to string
 */
inline static const std::string cublasGetErrorString(cublasStatus_t error) {
  switch(error){
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";

  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";

  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";

  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";

  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";

  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";

  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";

  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}

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
 * Throws CudaException if status is not CUBLAS_STATUS_SUCCESS with the message as the string for the exception and the string for the Cuda error.
 */
inline void handleCublasStatus(cublasStatus_t status, std::string message) {
  if(status != CUBLAS_STATUS_SUCCESS){
    message.append(cublasGetErrorString(status)); //FIXME
    throw CublasException(message.c_str());
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
  handleCudaStatus(cudaHostAlloc(pointerDevice, number * sizeof(PRECISION), cudaHostAllocPortable),"Pinned device memory allocation failed: ");
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
