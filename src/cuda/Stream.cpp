#include "Stream.h"

namespace CuEira {
namespace CUDA {

Stream::Stream(const Device& device, cudaStream_t* cudaStream, cublasHandle_t* cublasHandle) :
    device(device), cudaStream(cudaStream), cublasHandle(cublasHandle) {

}

Stream::~Stream() {
  CUDA::handleCudaStatus(cudaStreamDestroy(*cudaStream), "Failed to destroy cuda stream:");
  CUDA::handleCublasStatus(cublasDestroy(*cublasHandle), "Failed to destroy cublas handle:");

  delete cudaStream;
  delete cublasHandle;
}

const cudaStream_t& Stream::getCudaStream() const {
  return *cudaStream;
}

const cublasHandle_t& Stream::getCublasHandle() const {
  return *cublasHandle;
}

const Device& Stream::getAssociatedDevice() const {
  return device;
}

} /* namespace CUDA */
} /* namespace CuEira */
