#ifndef DEVICE_H_
#define DEVICE_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <CudaAdapter.cu>
#include <CudaException.h>
#include <DeviceVector.h>
#include <InvalidState.h>

namespace CuEira {
namespace CUDA {

/**
 * This is a
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class Device {
public:
  Device(int deviceNumber);
  virtual ~Device();

  virtual bool isActive() const;
  virtual bool setActiveDevice() const;

#ifndef __CUDACC__
  Device(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(const Device&) = delete;
  Device& operator=(Device&&) = delete;
#endif

private:
  const int deviceNumber;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* DEVICE_H_ */
