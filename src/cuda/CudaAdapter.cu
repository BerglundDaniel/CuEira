#include "CudaAdapter.h"

namespace CuEira {
namespace CUDA {

void handleCudaStatus(cudaError_t error, std::string message) {
  if(error != cudaSuccess){
    message.append(cudaGetErrorString(error));
    throw CudaException(message.c_str());
  }
}

void allocateDeviceMemory(void** pointerDevice, int number) {
  handleCudaStatus(cudaMalloc(pointerDevice, number * sizeof(PRECISION)), "Device memory allocation failed: ");
}

void allocateHostPinnedMemory(void** pointerDevice, int number) {
  handleCudaStatus(cudaHostAlloc(pointerDevice, number * sizeof(PRECISION), cudaHostAllocPortable),
      "Host pinned memory allocation failed: ");
}

void freeDeviceMemory(void* pointerDevice) {
  handleCudaStatus(cudaFree(pointerDevice), "Freeing device memory failed: ");
}

void freePinnedMemory(void* pointerDevice) {
  handleCudaStatus(cudaFreeHost(pointerDevice), "Freeing host memory failed: ");
}

} /* namespace CUDA */
} /* namespace CuEira */
