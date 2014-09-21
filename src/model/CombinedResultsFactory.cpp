#include "CombinedResultsFactory.h"

namespace CuEira {
namespace Model {

CombinedResultsFactory::CombinedResultsFactory(const InteractionStatisticsFactory& interactionStatisticsFactory) :
    interactionStatisticsFactory(&interactionStatisticsFactory) {

}

CombinedResultsFactory::~CombinedResultsFactory() {

}

CombinedResultsFactory::CombinedResultsFactory() :
    interactionStatisticsFactory(nullptr) {

}

CombinedResults* CombinedResultsFactory::constructCombinedResults(
    LogisticRegression::LogisticRegressionResult* logisticRegressionResult, Recode recode) const {
  InteractionStatistics* interactionStatistics = interactionStatisticsFactory->constructInteractionStatistics(
      logisticRegressionResult);

  return new CombinedResults(interactionStatistics, recode);
}

} /* namespace Model */
} /* namespace CuEira */
