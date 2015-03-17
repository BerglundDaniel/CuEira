#ifndef CPUSNPVECTOR_H_
#define CPUSNPVECTOR_H_

#include <SNPVector.h>
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
class CpuSNPVector: public SNPVector {
public:
  CpuSNPVector();
  virtual ~CpuSNPVector();
};

} /* namespace CPU */
} /* namespace Container */
} /* namespace CuEira */

#endif /* CPUSNPVECTOR_H_ */
