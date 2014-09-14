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
#include <Stream.h>

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
   * Constructor for the class. Takes the stream the transfers should be executed on. A cublas handle as to be initiated before any calls to the functions.
   */
  DeviceToHost(const Stream& stream);
  virtual ~DeviceToHost();

  /**
   * Creates a matrix on the host and copies the data from the device matrix without releasing it, the call is made on the stream that was provided in the constructor.
   */
  HostMatrix* transferMatrix(const DeviceMatrix* matrixDevice) const;

  /**
   * Creates a vector on the host and copies the data from the device vector without releasing it, the call is made on the stream that was provided in the constructor.
   */
  HostVector* transferVector(const DeviceVector* vectorDevice) const;

  /**
   * Copies the data from the device matrix to the specified point in host memory without releasing the device memory, the call is made on the stream that
   * was provided in the constructor. The host memory has to be previously allocated with as cuda pinned host memory.
   */
  void transferMatrix(const DeviceMatrix* matrixDevice, PRECISION* hostMemoryPosition) const;

  /**
   * Copies the data from the device vector to the specified point in host memory without releasing the device memory, the call is made on the stream that
   * was provided in the constructor. The host memory has to be previously allocated with as cuda pinned host memory.
   */
  void transferVector(const DeviceVector* vectorDevice, PRECISION* hostMemoryPosition) const;

private:
  const cudaStream_t& cudaStream;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* DEVICETOHOST_H_ */
