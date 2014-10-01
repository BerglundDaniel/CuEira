#include "ModelStatisticsFactory.h"

namespace CuEira {

ModelStatisticsFactory::ModelStatisticsFactory() {

}

ModelStatisticsFactory::~ModelStatisticsFactory() {

}

ModelStatistics* ModelStatisticsFactory::constructModelStatistics(
    const Model::LogisticRegression::LogisticRegressionResult* logisticRegressionResult,
    StatisticModel statisticModel) const {
  if(statisticModel == ADDITIVE){
    return new InteractionStatistics(logisticRegressionResult);
  }else{
    return new OddsRatioStatistics(logisticRegressionResult);
  }

}

} /* namespace CuEira */
