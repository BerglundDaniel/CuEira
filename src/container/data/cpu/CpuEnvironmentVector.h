#ifndef CPUENVIRONMENTVECTOR_H_
#define CPUENVIRONMENTVECTOR_H_

#include <EnvironmentVector.h>
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
class CpuEnvironmentVector: public EnvironmentVector {
public:
  CpuEnvironmentVector();
  virtual ~CpuEnvironmentVector();
};

} /* namespace CPU */
} /* namespace Container */
} /* namespace CuEira */

#endif /* CPUENVIRONMENTVECTOR_H_ */
