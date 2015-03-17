#ifndef CUDAINTERACTIONVECTOR_H_
#define CUDAINTERACTIONVECTOR_H_

#include <InteractionVector.h>
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
class CudaInteractionVector: public InteractionVector {
public:
  CudaInteractionVector();
  virtual ~CudaInteractionVector();
};

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */

#endif /* CUDAINTERACTIONVECTOR_H_ */
