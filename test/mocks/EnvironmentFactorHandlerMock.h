#ifndef ENVIRONMENTFACTORHANDLERMOCK_H_
#define ENVIRONMENTFACTORHANDLERMOCK_H_

#include <gmock/gmock.h>
#include <memory>

#include <Id.h>
#include <EnvironmentFactor.h>
#include <EnvironmentFactorHandler.h>

namespace CuEira {

template<typename Vector>
class EnvironmentFactorHandlerMock: public EnvironmentFactorHandler<Vector> {
public:
  EnvironmentFactorHandlerMock() :
      EnvironmentFactorHandler(make_shared(new EnvironmentFactor(Id("env1"))), new Vector(1)) {

  }

  virtual ~EnvironmentFactorHandlerMock() {

  }

  MOCK_CONST_METHOD0(getNumberOfIndividualsTotal, int());
  MOCK_CONST_METHOD0(getEnvironmentFactor, const EnvironmentFactor&());
  MOCK_CONST_METHOD0(getEnvironmentData, const Vector& ());

};

} /* namespace CuEira */

#endif /* ENVIRONMENTFACTORHANDLERMOCK_H_ */
