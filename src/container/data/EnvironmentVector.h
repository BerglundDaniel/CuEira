#ifndef ENVIRONMENTVECTOR_H_
#define ENVIRONMENTVECTOR_H_

#include <HostVector.h>
#include <Recode.h>
#include <StatisticModel.h>

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
  EnvironmentVector();
  virtual ~EnvironmentVector();

  int getNumberOfIndividualsToInclude() const;
  const Container::HostVector& getRecodedData() const;
  void recode(Recode recode);
  void applyStatisticModel(StatisticModel statisticModel, const HostVector& interactionVector);

private:
  int numberOfIndividualsToInclude;
  const std::vector<int>* originalData;
  HostVector* recodedData;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* ENVIRONMENTVECTOR_H_ */
