#include "StatisticsFactory.h"

namespace CuEira {

StatisticsFactory::StatisticsFactory() {

}

StatisticsFactory::~StatisticsFactory() {

}

Statistics* StatisticsFactory::constructStatistics(
    const Model::LogisticRegression::LogisticRegressionResult* logisticRegressionResult) const {
  return new Statistics(logisticRegressionResult);
}

} /* namespace CuEira */
