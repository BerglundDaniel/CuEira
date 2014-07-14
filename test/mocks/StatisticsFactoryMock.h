#ifndef STATISTICSFACTORYMOCK_H_
#define STATISTICSFACTORYMOCK_H_

#include <gmock/gmock.h>

#include <Statistics.h>
#include <LogisticRegressionResult.h>
#include <StatisticsFactory.h>

namespace CuEira {

class StatisticsFactoryMock: public StatisticsFactory {
public:
  StatisticsFactoryMock() :
      StatisticsFactory() {

  }

  virtual ~StatisticsFactoryMock() {

  }

  MOCK_CONST_METHOD1(constructStatistics, Statistics*(const CuEira::Model::LogisticRegression::LogisticRegressionResult*));
};

} /* namespace CuEira */

#endif /* STATISTICSFACTORYMOCK_H_ */
