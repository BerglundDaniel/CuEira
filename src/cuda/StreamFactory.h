#ifndef STREAMFACTORY_H_
#define STREAMFACTORY_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <Stream.h>
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
class StreamFactory {
public:
  StreamFactory();
  virtual ~StreamFactory();

  Stream* constructStream(const Device& device) const;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* STREAMFACTORY_H_ */
