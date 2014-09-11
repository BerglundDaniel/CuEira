#include "StreamFactory.h"

namespace CuEira {
namespace CUDA {

StreamFactory::StreamFactory() {

}

StreamFactory::~StreamFactory() {

}

Stream* StreamFactory::constructStream(const Device& device) const {
  if(!device.isActive()){
    throw new CudaException("Device provided to StreamFactory is not currently active.");
  }
  cudaStream_t* cudaStream = new cudaStream_t();
  cublasHandle_t* cublasHandle = new cublasHandle_t();

  handleCublasStatus(cublasCreate(cublasHandle), "Failed to create new cublas handle:");
  handleCudaStatus(cudaStreamCreate(cudaStream), "Failed to create new cuda stream:");
  handleCublasStatus(cublasSetStream(*cublasHandle, *cudaStream), "Failed to set cuda stream for cublas handle:");

  return new Stream(device, cudaStream, cublasHandle);
}

} /* namespace CUDA */
} /* namespace CuEira */
