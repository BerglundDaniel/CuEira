#ifndef ENVIRONMENTFACTORHANDLER_H_
#define ENVIRONMENTFACTORHANDLER_H_

#include <memory>

#include <Id.h>
#include <EnvironmentFactor.h>

namespace CuEira {

/*
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */

template<typename Vector>
class EnvironmentFactorHandler {
public:
  explicit EnvironmentFactorHandler(std::shared_ptr<const EnvironmentFactor> environmentFactor, const Vector* vector);
  virtual ~EnvironmentFactorHandler();

  virtual int getNumberOfIndividualsTotal() const;
  virtual const EnvironmentFactor& getEnvironmentFactor() const;
  virtual const Vector& getEnvironmentData() const;

  EnvironmentFactorHandler(const EnvironmentFactorHandler&) = delete;
  EnvironmentFactorHandler(EnvironmentFactorHandler&&) = delete;
  EnvironmentFactorHandler& operator=(const EnvironmentFactorHandler&) = delete;
  EnvironmentFactorHandler& operator=(EnvironmentFactorHandler&&) = delete;

protected:
  const Vector* vector;
  std::shared_ptr<const EnvironmentFactor> environmentFactor;
  const int numberOfIndividualsTotal;
};

} /* namespace CuEira */

#endif /* ENVIRONMENTFACTORHANDLER_H_ */
