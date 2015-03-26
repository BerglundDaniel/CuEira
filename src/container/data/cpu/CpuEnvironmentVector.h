#ifndef CPUENVIRONMENTVECTOR_H_
#define CPUENVIRONMENTVECTOR_H_

#include <CpuEnvironmentFactorHandler.h>
#include <EnvironmentVector.h>
#include <RegularHostVector.h>
#include <HostVector.h>
#include <CpuMissingDataHandler.h>
#include <StatisticModel.h>
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
class CpuEnvironmentVector: public EnvironmentVector {
public:
  CpuEnvironmentVector(const CpuEnvironmentFactorHandler& cpuEnvironmentFactorHandler, const MKLWrapper& mklWrapper);
  virtual ~CpuEnvironmentVector();

  virtual const Container::HostVector& getEnvironmentData() const;
  virtual void recode(Recode recode);
  virtual void recode(Recode recode, const CpuMissingDataHandler& missingDataHandler);

private:
  void recodeProtective();

  const MKLWrapper& mklWrapper;
  const HostVector& originalData;
  HostVector* recodedData;
};

} /* namespace CPU */
} /* namespace Container */
} /* namespace CuEira */

#endif /* CPUENVIRONMENTVECTOR_H_ */
