#ifndef ENVIRONMENTFACTORHANDLERFACTORY_H_
#define ENVIRONMENTFACTORHANDLERFACTORY_H_

#include <vector>

#include <EnvironmentFactorHandler.h>
#include <EnvironmentFactor.h>
#include <HostMatrix.h>

#ifdef CPU
#include <CpuEnvironmentFactorHandler.h>
#else
#include <CudaEnvironmentFactorHandler.h>
#include <DeviceMatrix.h>
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

  EnvironmentFactorHandler* constructEnvironmentFactorHandler(const Container::HostMatrix* dataMatrix,
      const std::vector<EnvironmentFactor*>* environmentFactors) const;
};

} /* namespace CuEira */

#endif /* ENVIRONMENTFACTORHANDLERFACTORY_H_ */
