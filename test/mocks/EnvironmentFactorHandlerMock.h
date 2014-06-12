#ifndef ENVIRONMENTFACTORHANDLERMOCK_H_
#define ENVIRONMENTFACTORHANDLERMOCK_H_

#include <gmock/gmock.h>

#include <Id.h>
#include <EnvironmentFactor.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <EnvironmentFactorHandler.h>

namespace CuEira {

class EnvironmentFactorHandlerMock: public EnvironmentFactorHandler {
public:
  EnvironmentFactorHandlerMock(Container::HostMatrix* dataMatrix, std::vector<EnvironmentFactor*>* environmentFactors) :
      EnvironmentFactorHandler(dataMatrix, environmentFactors) {

  }

  virtual ~EnvironmentFactorHandlerMock() {

  }

  MOCK_CONST_METHOD0(getHeaders, const std::vector<EnvironmentFactor*>&());
  MOCK_CONST_METHOD1(getData, const Container::HostVector& (const EnvironmentFactor&));
};

} /* namespace CuEira */

#endif /* ENVIRONMENTFACTORHANDLERMOCK_H_ */
