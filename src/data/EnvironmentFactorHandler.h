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
  EnvironmentFactorHandler(const std::vector<const EnvironmentFactor*>* environmentFactors,
      const std::vector<std::set<int>>* personsToSkip, int numberOfIndividualsTotal);
  virtual ~EnvironmentFactorHandler();

  virtual int getNumberOfIndividualsTotal() const;
  virtual int getNumberOfEnvironmentFactors() const;
  virtual const std::vector<const EnvironmentFactor*>& getHeaders() const;

  virtual Container::EnvironmentVector* getEnvironmentVector(const EnvironmentFactor& environmentFactor) const=0;

  EnvironmentFactorHandler(const EnvironmentFactorHandler&) = delete;
  EnvironmentFactorHandler(EnvironmentFactorHandler&&) = delete;
  EnvironmentFactorHandler& operator=(const EnvironmentFactorHandler&) = delete;
  EnvironmentFactorHandler& operator=(EnvironmentFactorHandler&&) = delete;

protected:
  const std::vector<std::set<int>>* personsToSkip;
  const std::vector<const EnvironmentFactor*>* environmentFactors;
  const int numberOfEnvironmentFactors;
  const int numberOfIndividualsTotal;
};

} /* namespace CuEira */

#endif /* ENVIRONMENTFACTORHANDLER_H_ */
