#ifndef CUDASNPVECTOR_H_
#define CUDASNPVECTOR_H_

#include <SNPVector.h>
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
class CudaSNPVector: public SNPVector {
public:
  CudaSNPVector();
  virtual ~CudaSNPVector();
};

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */

#endif /* CUDASNPVECTOR_H_ */
