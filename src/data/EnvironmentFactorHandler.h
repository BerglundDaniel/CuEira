#ifndef ENVIRONMENTFACTORHANDLER_H_
#define ENVIRONMENTFACTORHANDLER_H_

#include <vector>

#include <Id.h>
#include <EnvironmentFactor.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <EnvironmentFactorHandlerException.h>

namespace CuEira {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class EnvironmentFactorHandler {
public:
  EnvironmentFactorHandler(Container::HostMatrix* dataMatrix, std::vector<EnvironmentFactor*>* environmentFactors);
  virtual ~EnvironmentFactorHandler();

  virtual const std::vector<EnvironmentFactor*>& getHeaders() const;
  virtual const Container::HostVector& getData(const EnvironmentFactor& environmentFactor) const;

private:
  Container::HostMatrix* dataMatrix;
  std::vector<EnvironmentFactor*>* environmentFactors;
  const int numberOfColumns;
  const int numberOfIndividualsToInclude;
};

} /* namespace CuEira */

#endif /* ENVIRONMENTFACTORHANDLER_H_ */
