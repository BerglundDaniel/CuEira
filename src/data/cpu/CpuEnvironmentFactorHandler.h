#ifndef CPUENVIRONMENTFACTORHANDLER_H_
#define CPUENVIRONMENTFACTORHANDLER_H_

#include <vector>
#include <set>

#include <EnvironmentFactorHandler.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <EnvironmentFactor.h>
#include <CpuEnvironmentVector.h>
#include <EnvironmentFactorHandlerException.h>

namespace CuEira {
namespace CPU {

/**
 * This class ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuEnvironmentFactorHandler: public EnvironmentFactorHandler {
public:
  explicit CpuEnvironmentFactorHandler(const Container::HostMatrix* dataMatrix,
      const std::vector<const EnvironmentFactor*>* environmentFactors, const std::vector<std::set<int>>* personsToSkip);
  virtual ~CpuEnvironmentFactorHandler();

  virtual Container::CPU::CpuEnvironmentVector* getEnvironmentVector(const EnvironmentFactor& environmentFactor) const;

private:
  const Container::HostMatrix* dataMatrix;
};

} /* namespace CPU */
} /* namespace CuEira */

#endif /* CPUENVIRONMENTFACTORHANDLER_H_ */
