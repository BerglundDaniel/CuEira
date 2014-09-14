#ifndef HOSTTODEVICE_H_
#define HOSTTODEVICE_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <CudaAdapter.cu>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <HostVector.h>
#include <HostMatrix.h>
#include <Stream.h>

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
   * Constructor for the class. Takes the stream the transfers should be executed on. A cublas handle as to be initiated before any calls to the functions.
   */
  HostToDevice(const Stream& stream);
  virtual ~HostToDevice();

  /**
   * Creates a matrix on the devices and copies the data from the matrix on the host, the call is made on the stream that was provided in the constructor.
   * Does not release the memory on the device.
   */
  virtual DeviceMatrix* transferMatrix(const HostMatrix* matrixHost) const;

  /**
   * Creates a vector on the devices and copies the data from the vector on the host, the call is made on the stream that was provided in the constructor.
   * Does not release the memory on the device.
   */
  virtual DeviceVector* transferVector(const HostVector* vectorHost) const;

  /**
   * Transfers the matrix to the area in device memory given by deviceMemoryPosition. It has to point to an area with already allocated memory and with
   * enough space to fit the matrix after the point specified.
   */
  virtual void transferMatrix(const HostMatrix* matrixHost, PRECISION* deviceMemoryPosition) const;

  /**
   * Transfers the vector to the area in device memory given by deviceMemoryPosition. It has to point to an area with already allocated memory and with
   * enough space to fit the vector after the point specified.
   */
  virtual void transferVector(const HostVector* vectorHost, PRECISION* deviceMemoryPosition) const;

protected:
  HostToDevice(); //For the mock object

private:
  const cudaStream_t* cudaStream;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* HOSTTODEVICE_H_ */
