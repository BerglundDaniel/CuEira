#ifndef ENVIRONMENTFACTORHANDLER_H_
#define ENVIRONMENTFACTORHANDLER_H_

#include <vector>
#include <set>

#include <Id.h>
#include <EnvironmentFactor.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <EnvironmentFactorHandlerException.h>
#include <EnvironmentVector.h>

namespace CuEira {

/*
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class EnvironmentFactorHandler {
public:
  explicit EnvironmentFactorHandler(const EnvironmentFactor* environmentFactors, int numberOfIndividualsTotal);
  virtual ~EnvironmentFactorHandler();

  virtual int getNumberOfIndividualsTotal() const;
  virtual const EnvironmentFactor& getEnvironmentFactor() const;

  virtual Container::Vector& getEnvironmentData() const=0;

  EnvironmentFactorHandler(const EnvironmentFactorHandler&) = delete;
  EnvironmentFactorHandler(EnvironmentFactorHandler&&) = delete;
  EnvironmentFactorHandler& operator=(const EnvironmentFactorHandler&) = delete;
  EnvironmentFactorHandler& operator=(EnvironmentFactorHandler&&) = delete;

protected:
  const EnvironmentFactor* environmentFactor;
  const int numberOfIndividualsTotal;
};

} /* namespace CuEira */

#endif /* ENVIRONMENTFACTORHANDLER_H_ */
