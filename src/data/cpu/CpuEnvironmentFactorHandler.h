#ifndef CPUENVIRONMENTFACTORHANDLER_H_
#define CPUENVIRONMENTFACTORHANDLER_H_

#include <EnvironmentFactorHandler.h>
#include <HostVector.h>
#include <EnvironmentFactor.h>
#include <CpuEnvironmentVector.h>

namespace CuEira {
namespace CPU {

/**
 * This class ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuEnvironmentFactorHandler: public EnvironmentFactorHandler {
public:
  explicit CpuEnvironmentFactorHandler(const Container::HostVector* envData,
      const EnvironmentFactor* environmentFactor);
  virtual ~CpuEnvironmentFactorHandler();

  virtual const Container::HostVector& getEnvironmentData() const;

private:
  const Container::HostVector* envData;
};

} /* namespace CPU */
} /* namespace CuEira */

#endif /* CPUENVIRONMENTFACTORHANDLER_H_ */
