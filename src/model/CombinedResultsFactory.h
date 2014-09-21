#ifndef COMBINEDRESULTSFACTORY_H_
#define COMBINEDRESULTSFACTORY_H_

#include <CombinedResults.h>
#include <InteractionStatistics.h>
#include <InteractionStatisticsFactory.h>
#include <LogisticRegressionResult.h>
#include <Recode.h>

namespace CuEira {
namespace Model {

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CombinedResultsFactory {
public:
  CombinedResultsFactory(const InteractionStatisticsFactory& interactionStatisticsFactory);
  virtual ~CombinedResultsFactory();

  virtual CombinedResults* constructCombinedResults(LogisticRegression::LogisticRegressionResult* logisticRegressionResult, Recode recode) const;

protected:
  CombinedResultsFactory(); //For the mock

private:
  const InteractionStatisticsFactory* interactionStatisticsFactory;
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* COMBINEDRESULTSFACTORY_H_ */
