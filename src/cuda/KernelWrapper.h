#ifndef KERNELWRAPPER_H_
#define KERNELWRAPPER_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <CudaAdapter.cu>
#include <DeviceVector.h>
#include <DeviceMatrix.h>

namespace CuEira {
namespace CUDA {

using namespace CuEira::Container;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class KernelWrapper {
public:
  /**
   * Constructor for the class. Takes the stream the transfers should be executed on. Some functions requires that a cublas context has been created.
   * All of them assumes a cuda context exists for the stream.
   */
  KernelWrapper(const cudaStream_t& cudaStream);
  virtual ~KernelWrapper();

  void logisticTransform(const DeviceVector& logitVector, DeviceVector& probabilites) const;

private:
  const cudaStream_t& cudaStream;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* KERNELWRAPPER_H_ */
