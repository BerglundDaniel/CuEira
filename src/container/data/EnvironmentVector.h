#ifndef ENVIRONMENTVECTOR_H_
#define ENVIRONMENTVECTOR_H_

#include <HostVector.h>
#include <Recode.h>
#include <StatisticModel.h>
#include <EnvironmentFactor.h>
#include <EnvironmentFactorHandler.h>

#ifdef CPU
#include <lapackpp/lavd.h>
#include <LapackppHostVector.h>
#else
#include <PinnedHostVector.h>
#endif

namespace CuEira {
namespace Container {

/**
 * This class ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class EnvironmentVector {
public:
  EnvironmentVector(const EnvironmentFactorHandler& environmentHandler, EnvironmentFactor& environmentFactor);
  virtual ~EnvironmentVector();

  int getNumberOfIndividualsToInclude() const;
  const Container::HostVector& getRecodedData() const;
  void recode(Recode recode);
  void applyStatisticModel(StatisticModel statisticModel, const HostVector& interactionVector);

private:
  PRECISION invertEnvironmentFactor(PRECISION envData) const;

  const EnvironmentFactorHandler& environmentHandler;
  int numberOfIndividualsToInclude;
  const HostVector& originalData;
  HostVector* recodedData;
  Recode currentRecode;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* ENVIRONMENTVECTOR_H_ */
