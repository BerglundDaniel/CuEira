#ifndef INTERACTIONVECTORMOCK_H_
#define INTERACTIONVECTORMOCK_H_

#include <gmock/gmock.h>

#include <EnvironmentVector.h>
#include <Recode.h>
#include <StatisticModel.h>
#include <EnvironmentFactor.h>
#include <EnvironmentFactorHandler.h>
#include <HostVector.h>

namespace CuEira {
namespace Container {

class InteractionVectorMock: public InteractionVector {
public:
  InteractionVectorMock() :
      InteractionVector() {

  }

  virtual ~InteractionVectorMock() {

  }

  MOCK_METHOD1(recode, void(const SNPVector&));

  MOCK_CONST_METHOD0(getNumberOfIndividualsToInclude, int());
  MOCK_CONST_METHOD0(getRecodedData, const Container::HostVector&());
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* INTERACTIONVECTORMOCK_H_ */
