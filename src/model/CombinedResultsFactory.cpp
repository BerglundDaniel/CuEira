#include "CombinedResultsFactory.h"

namespace CuEira {
namespace Model {

CombinedResultsFactory::CombinedResultsFactory(const ModelStatisticsFactory& modelStatisticsFactory) :
    modelStatisticsFactory(&modelStatisticsFactory) {

}

CombinedResultsFactory::~CombinedResultsFactory() {

}

CombinedResultsFactory::CombinedResultsFactory() :
    modelStatisticsFactory(nullptr) {

}

CombinedResults* CombinedResultsFactory::constructCombinedResults(
    LogisticRegression::LogisticRegressionResult* additiveLogisticRegressionResult,
    LogisticRegression::LogisticRegressionResult* multiplicativeLogisticRegressionResult, Recode recode) const {

  ModelStatistics* additiveInteractionStatistics = modelStatisticsFactory->constructModelStatistics(
      additiveLogisticRegressionResult, ADDITIVE);
  ModelStatistics* multiplicativeInteractionStatistics = modelStatisticsFactory->constructModelStatistics(
      multiplicativeLogisticRegressionResult, MULTIPLICATIVE);

  return new LogisticRegression::LogisticRegressionCombinedResults((InteractionStatistics*)additiveInteractionStatistics, (OddsRatioStatistics*)multiplicativeInteractionStatistics,
      recode);
}

} /* namespace Model */
} /* namespace CuEira */
