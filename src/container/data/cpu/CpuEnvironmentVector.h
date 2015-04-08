#ifndef CPUENVIRONMENTVECTOR_H_
#define CPUENVIRONMENTVECTOR_H_

#include <EnvironmentFactorHandler.h>
#include <EnvironmentVector.h>
#include <RegularHostVector.h>
#include <HostVector.h>
#include <Recode.h>
#include <EnvironmentFactor.h>
#include <VariableType.h>
#include <InvalidState.h>
#include <MKLWrapper.h>

namespace CuEira {
namespace Container {
namespace CPU {

using namespace CuEira::CPU;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuEnvironmentVector: public EnvironmentVector<RegularHostVector> {
public:
  CpuEnvironmentVector(const EnvironmentFactorHandler<RegularHostVector>& environmentFactorHandler,
      const MKLWrapper& mklWrapper);
  virtual ~CpuEnvironmentVector();

protected:
  virtual void recodeProtective();
  virtual void recodeAllRisk();

  const MKLWrapper& mklWrapper;
};

} /* namespace CPU */
} /* namespace Container */
} /* namespace CuEira */

#endif /* CPUENVIRONMENTVECTOR_H_ */