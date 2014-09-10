#ifndef DEVICE_H_
#define DEVICE_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <Stream.h>
#include <CudaAdapter.cu>
#include <CudaException.h>
#include <DeviceVector.h>

namespace CuEira {
namespace CUDA {

/**
 * This is a
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class Device {
public:
  Device(int deviceNumber, const Container::DeviceVector* outcomes);
  virtual ~Device();

  virtual bool isActive() const;
  virtual bool setActiveDevice() const;

  virtual const Container::DeviceVector& getOutcomes() const;

private:
  const Container::DeviceVector* outcomes;
  const int deviceNumber;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* DEVICE_H_ */
