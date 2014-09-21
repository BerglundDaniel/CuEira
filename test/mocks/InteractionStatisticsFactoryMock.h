#ifndef INTERACTIONSTATISTICSFACTORYMOCK_H_
#define INTERACTIONSTATISTICSFACTORYMOCK_H_

#include <gmock/gmock.h>

#include <InteractionStatistics.h>
#include <LogisticRegressionResult.h>
#include <InteractionStatisticsFactory.h>

namespace CuEira {

class InteractionStatisticsFactoryMock: public InteractionStatisticsFactory {
public:
  InteractionStatisticsFactoryMock() :
      InteractionStatisticsFactory() {

  }

  virtual ~InteractionStatisticsFactoryMock() {

  }

  MOCK_CONST_METHOD1(constructInteractionStatistics, InteractionStatistics*(const CuEira::Model::LogisticRegression::LogisticRegressionResult*));
};

} /* namespace CuEira */

#endif /* INTERACTIONSTATISTICSFACTORYMOCK_H_ */
