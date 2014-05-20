#ifndef DEVICETOHOST_H_
#define DEVICETOHOST_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <CudaAdapter.cu>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <HostVector.h>
#include <HostMatrix.h>
#include <PinnedHostVector.h>
#include <PinnedHostMatrix.h>

namespace CuEira {
namespace CUDA {

using namespace CuEira::Container;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DeviceToHost {
public:
  /**
   * Constructor for the class. Takes the handle to the Cublas context and the CudaStream the transfers should be on.
   */
  DeviceToHost(cublasHandle_t& cublasHandle);
  virtual ~DeviceToHost();

  /**
   * Creates a matrix on the host and copies the data from the device matrix without releasing it, the call is made on the stream that was provided in the constructor.
   */
  HostMatrix* transferMatrix(const DeviceMatrix* matrixDevice) const;

  /**
   * Creates a vector on the host and copies the data from the device vector without releasing it, the call is made on the stream that was provided in the constructor.
   */
  HostVector* transferVector(const DeviceVector* vectorDevice) const;

private:
  cublasHandle_t& cublasHandle;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* DEVICETOHOST_H_ */
