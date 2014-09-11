#ifndef STREAM_H_
#define STREAM_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <Device.h>
#include <CudaAdapter.cu>
#include <CudaException.h>

namespace CuEira {
namespace CUDA {

/**
 * This is a
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class Stream {
public:
  Stream(const Device& device, cudaStream_t* cudaStream, cublasHandle_t* cublasHandle);
  virtual ~Stream();

  virtual const cudaStream_t& getCudaStream() const;
  virtual const cublasHandle_t& getCublasHandle() const;
  virtual const Device& getAssociatedDevice() const;

  /**
   * Sync the stream
   */
  inline void syncStream() const {
    cudaStreamSynchronize(*cudaStream);
  }

private:
  const Device& device;
  cudaStream_t* cudaStream;
  cublasHandle_t* cublasHandle;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* STREAM_H_ */
