#ifndef COMBINEDRESULTSFACTORY_H_
#define COMBINEDRESULTSFACTORY_H_

#include <CombinedResults.h>
#include <CombinedResultsLogisticRegression.h>
#include <InteractionStatistics.h>
#include <ModelStatistics.h>
#include <OddsRatioStatistics.h>
#include <ModelStatisticsFactory.h>
#include <LogisticRegressionResult.h>
#include <Recode.h>
#include <StatisticModel.h>

namespace CuEira {
namespace Model {

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CombinedResultsFactory {
public:
  CombinedResultsFactory(const ModelStatisticsFactory& modelStatisticsFactory);
  virtual ~CombinedResultsFactory();

  virtual CombinedResults* constructCombinedResults(
      LogisticRegression::LogisticRegressionResult* additiveLogisticRegressionResult,
      LogisticRegression::LogisticRegressionResult* multiplicativeLogisticRegressionResult, Recode recode) const;

protected:
  CombinedResultsFactory(); //For the mock

private:
  const ModelStatisticsFactory* modelStatisticsFactory;
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* COMBINEDRESULTSFACTORY_H_ */
