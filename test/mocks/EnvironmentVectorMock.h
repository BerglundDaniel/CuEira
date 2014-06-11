#ifndef ENVIRONMENTMOCK_H_
#define ENVIRONMENTMOCK_H_

#include <gmock/gmock.h>

#include <EnvironmentVector.h>
#include <Recode.h>
#include <StatisticModel.h>
#include <EnvironmentFactor.h>
#include <EnvironmentFactorHandler.h>
#include <HostVector.h>

namespace CuEira {
namespace Container {

class EnvironmentVectorMock: public EnvironmentVector {
public:
  EnvironmentVectorMock(const EnvironmentFactorHandler& environmentHandler, EnvironmentFactor& environmentFactor) :
      EnvironmentVector(environmentHandler, environmentFactor) {

  }

  virtual ~EnvironmentVectorMock() {

  }

  MOCK_METHOD1(switchEnvironmentFactor, void(EnvironmentFactor&));
  MOCK_METHOD1(recode, void(Recode));

  MOCK_CONST_METHOD0(getNumberOfIndividualsToInclude, int());
  MOCK_CONST_METHOD0(getRecodedData, const Container::HostVector& ());
  MOCK_CONST_METHOD0(getCurrentEnvironmentFactor, const EnvironmentFactor&());
  MOCK_CONST_METHOD2(applyStatisticModel, void(StatisticModel, const Container::HostVector&));
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* ENVIRONMENTMOCK_H_ */
