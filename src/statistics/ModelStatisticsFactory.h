#ifndef MODELSTATISTICSFACTORY_H_
#define MODELSTATISTICSFACTORY_H_

#include <ModelStatistics.h>
#include <LogisticRegressionResult.h>
#include <StatisticModel.h>
#include <OddsRatioStatistics.h>
#include <InteractionStatistics.h>

namespace CuEira {

/**
 * This is class is responsible for creating instances of ModelStatistics of various subtypes
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ModelStatisticsFactory {
public:
  ModelStatisticsFactory();
  virtual ~ModelStatisticsFactory();

  virtual ModelStatistics* constructModelStatistics(
      const Model::LogisticRegression::LogisticRegressionResult* logisticRegressionResult,
      StatisticModel statisticModel) const;
};

} /* namespace CuEira */

#endif /* MODELSTATISTICSFACTORY_H_ */
