#ifndef CPUINTERACTIONVECTOR_H_
#define CPUINTERACTIONVECTOR_H_

#include <InteractionVector.h>
#include <RegularHostVector.h>

namespace CuEira {
namespace Container {
namespace CPU {

using namespace CuEira::CPU;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuInteractionVector: public InteractionVector {
public:
  CpuInteractionVector();
  virtual ~CpuInteractionVector();
};

} /* namespace CPU */
} /* namespace Container */
} /* namespace CuEira */

#endif /* CPUINTERACTIONVECTOR_H_ */
