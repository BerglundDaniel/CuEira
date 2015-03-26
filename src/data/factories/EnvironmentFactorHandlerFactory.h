#ifndef ENVIRONMENTFACTORHANDLERFACTORY_H_
#define ENVIRONMENTFACTORHANDLERFACTORY_H_

#include <vector>

#include <EnvironmentFactorHandler.h>
#include <EnvironmentFactor.h>
#include <HostVector.h>

#ifdef CPU
#include <CpuEnvironmentFactorHandler.h>
#else
#include <CudaEnvironmentFactorHandler.h>
#include <DeviceVector.h>
#include <HostToDevice.h>
#endif

namespace CuEira {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class EnvironmentFactorHandlerFactory {
public:
  explicit EnvironmentFactorHandlerFactory();
  virtual ~EnvironmentFactorHandlerFactory();

  EnvironmentFactorHandler* constructEnvironmentFactorHandler(const Container::HostVector* envData,
      EnvironmentFactor* environmentFactor) const;
};

} /* namespace CuEira */

#endif /* ENVIRONMENTFACTORHANDLERFACTORY_H_ */
