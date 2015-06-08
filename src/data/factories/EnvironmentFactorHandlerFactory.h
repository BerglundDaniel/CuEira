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
template<typename Matrix, typename Vector>
class EnvironmentFactorHandlerFactory {
public:
  explicit EnvironmentFactorHandlerFactory(const Configuration& configuration,
      const std::vector<std::string>& columnNames, const Matrix& matrix);
  virtual ~EnvironmentFactorHandlerFactory();

protected:
  const Vector* envData; //Not the actual data since it's owned by the input matrix
  std::shared_ptr<const EnvironmentFactor> environmentFactor;
};

} /* namespace CuEira */

#endif /* ENVIRONMENTFACTORHANDLERFACTORY_H_ */
