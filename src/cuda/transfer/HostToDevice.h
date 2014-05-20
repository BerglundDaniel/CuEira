#ifndef HOSTTODEVICE_H_
#define HOSTTODEVICE_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <CudaAdapter.cu>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <HostVector.h>
#include <HostMatrix.h>

namespace CuEira {
namespace CUDA {

using namespace CuEira::Container;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class HostToDevice {
public:
  /**
   * Constructor for the class. Takes the handle to the Cublas context and the CudaStream the transfers should be on.
   */
  HostToDevice(cublasHandle_t& cublasHandle);
  virtual ~HostToDevice();

  /**
   * Creates a matrix on the devices and copies the data from the matrix on the host, the call is made on the stream that was provided in the constructor.
   * Does not release the memory on the device.
   */
  DeviceMatrix* transferMatrix(const HostMatrix* matrixHost) const;

  /**
   * Creates a vector on the devices and copies the data from the vector on the host, the call is made on the stream that was provided in the constructor.
   * Does not release the memory on the device.
   */
  DeviceVector* transferVector(const HostVector* vectorHost) const;

private:
  cublasHandle_t& cublasHandle;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* HOSTTODEVICE_H_ */
