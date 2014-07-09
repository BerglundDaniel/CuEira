#ifndef DATAHANDLERMOCK_H_
#define DATAHANDLERMOCK_H_

#include <gmock/gmock.h>

#include <DataHandler.h>
#include <SNP.h>
#include <HostVector.h>
#include <EnvironmentFactor.h>
#include <Recode.h>

namespace CuEira {

class DataHandlerMock: public DataHandler {
public:
  DataHandlerMock() :
      DataHandler() { //FIXME

  }

  virtual ~DataHandlerMock() {

  }

  MOCK_CONST_METHOD0(getCurrentSNP, const SNP&());
  MOCK_CONST_METHOD0(getCurrentEnvironmentFactor, const EnvironmentFactor&());
  MOCK_CONST_METHOD0(getRecode, Recode());
  MOCK_CONST_METHOD0(getSNP, const Container::HostVector&());
  MOCK_CONST_METHOD0(getInteraction, const Container::HostVector&());
  MOCK_CONST_METHOD0(getEnvironment, const Container::HostVector&());

  MOCK_METHOD0(next, bool());
  MOCK_METHOD1(recode, void(Recode));
};

} /* namespace CuEira */

#endif /* DATAHANDLERMOCK_H_ */
