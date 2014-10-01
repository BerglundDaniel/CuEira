#ifndef MODELSTATISTICSFACTORYMOCK_H_
#define MODELSTATISTICSFACTORYMOCK_H_

#include <gmock/gmock.h>

#include <ModelStatistics.h>
#include <LogisticRegressionResult.h>
#include <ModelStatisticsFactory.h>
#include <StatisticModel.h>

namespace CuEira {

class ModelStatisticsFactoryMock: public ModelStatisticsFactory {
public:
  ModelStatisticsFactoryMock() :
      ModelStatisticsFactory() {

  }

  virtual ~ModelStatisticsFactoryMock() {

  }

  MOCK_CONST_METHOD2(constructModelStatistics, ModelStatistics*(const CuEira::Model::LogisticRegression::LogisticRegressionResult*, StatisticModel));
};

} /* namespace CuEira */

#endif /* MODELSTATISTICSFACTORYMOCK_H_ */
