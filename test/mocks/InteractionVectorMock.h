#ifndef INTERACTIONVECTORMOCK_H_
#define INTERACTIONVECTORMOCK_H_

#include <gmock/gmock.h>

#include <EnvironmentVector.h>
#include <Recode.h>
#include <EnvironmentFactor.h>
#include <EnvironmentFactorHandler.h>
#include <HostVector.h>

namespace CuEira {
namespace Container {

template<typename Vector>
class InteractionVectorMock: public InteractionVector<Vector> {
public:
  InteractionVectorMock() :
      InteractionVector() {

  }

  virtual ~InteractionVectorMock() {

  }

  MOCK_CONST_METHOD0(getNumberOfIndividualsToInclude, int());
  MOCK_CONST_METHOD0(getInteractionData, Vector&());

  MOCK_METHOD0(getInteractionData, Vector&());
  MOCK_METHOD1(updateSize, void(int));

};

} /* namespace Container */
} /* namespace CuEira */

#endif /* INTERACTIONVECTORMOCK_H_ */
