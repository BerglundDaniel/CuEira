#ifndef KERNELWRAPPERFACTORY_H_
#define KERNELWRAPPERFACTORY_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <Stream.h>
#include <KernelWrapper.h>

namespace CuEira {
namespace CUDA {

/**
 * This is a
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class KernelWrapperFactory {
public:
  KernelWrapperFactory();
  virtual ~KernelWrapperFactory();

  virtual KernelWrapper* constructKernelWrapper(Stream stream) const;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* KERNELWRAPPERFACTORY_H_ */
