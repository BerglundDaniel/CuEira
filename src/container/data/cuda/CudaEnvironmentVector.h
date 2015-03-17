#ifndef CUDAENVIRONMENTVECTOR_H_
#define CUDAENVIRONMENTVECTOR_H_

#include <EnvironmentVector.h>
#include <DeviceVector.h>

namespace CuEira {
namespace Container {
namespace CUDA {

using namespace CuEira::CUDA;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaEnvironmentVector: public EnvironmentVector {
public:
  CudaEnvironmentVector();
  virtual ~CudaEnvironmentVector();
};

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */

#endif /* CUDAENVIRONMENTVECTOR_H_ */
