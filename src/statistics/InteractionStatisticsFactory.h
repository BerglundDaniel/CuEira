#ifndef INTERACTIONSTATISTICSFACTORY_H_
#define INTERACTIONSTATISTICSFACTORY_H_

#include <InteractionStatistics.h>
#include <LogisticRegressionResult.h>

namespace CuEira {

/**
 * This is class is responsible for creating instances of Statistics
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class InteractionStatisticsFactory {
public:
  InteractionStatisticsFactory();
  virtual ~InteractionStatisticsFactory();

  virtual InteractionStatistics* constructInteractionStatistics(const Model::LogisticRegression::LogisticRegressionResult* logisticRegressionResult) const;
};

} /* namespace CuEira */

#endif /* INTERACTIONSTATISTICSFACTORY_H_ */
