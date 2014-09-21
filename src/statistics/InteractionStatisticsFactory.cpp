#include "InteractionStatisticsFactory.h"

namespace CuEira {

InteractionStatisticsFactory::InteractionStatisticsFactory() {

}

InteractionStatisticsFactory::~InteractionStatisticsFactory() {

}

InteractionStatistics* InteractionStatisticsFactory::constructInteractionStatistics(
    const Model::LogisticRegression::LogisticRegressionResult* logisticRegressionResult) const {
  return new InteractionStatistics(logisticRegressionResult);
}

} /* namespace CuEira */
