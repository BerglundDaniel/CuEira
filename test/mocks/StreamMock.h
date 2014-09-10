#ifndef STREAMMOCK_H_
#define STREAMMOCK_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <gmock/gmock.h>

#include <Device.h>
#include <Stream.h>

namespace CuEira {
namespace CUDA {

class StreamMock: public Stream {
public:
  StreamMock(const Device& device, cudaStream_t* cudaStream, cublasHandle_t* cublasHandle) :
      Stream(device, cudaStream, cublasHandle) {

  }

  virtual ~StreamMock() {

  }

  MOCK_CONST_METHOD0(getCudaStream, const cudaStream_t&());
  MOCK_CONST_METHOD0(getCublasHandle, const cublasHandle_t&());
  MOCK_CONST_METHOD0(getAssociatedDevice, const Device&());

};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* STREAMMOCK_H_ */
