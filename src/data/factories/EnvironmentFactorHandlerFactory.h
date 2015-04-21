#ifndef ENVIRONMENTFACTORHANDLERFACTORY_H_
#define ENVIRONMENTFACTORHANDLERFACTORY_H_

#include <memory>
#include <vector>
#include <string>

#include <EnvironmentFactorHandler.h>
#include <EnvironmentFactor.h>
#include <HostVector.h>
#include <HostMatrix.h>
#include <Configuration.h>
#include <Id.h>

namespace CuEira {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class EnvironmentFactorHandlerFactory {
public:
  explicit EnvironmentFactorHandlerFactory(const Configuration& configuration,
      const std::vector<std::string>& columnNames, const Container::HostMatrix& matrix);
  virtual ~EnvironmentFactorHandlerFactory();

protected:
  const Container::HostVector* envData;
  std::shared_ptr<const EnvironmentFactor> environmentFactor;
};

} /* namespace CuEira */

#endif /* ENVIRONMENTFACTORHANDLERFACTORY_H_ */
