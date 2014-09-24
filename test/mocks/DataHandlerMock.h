#ifndef DATAHANDLERMOCK_H_
#define DATAHANDLERMOCK_H_

#include <gmock/gmock.h>

#include <DataHandler.h>
#include <SNP.h>
#include <HostVector.h>
#include <EnvironmentFactor.h>
#include <Recode.h>
#include <SNPVector.h>
#include <InteractionVector.h>
#include <EnvironmentVector.h>
#include <ModelInformation.h>
#include <DataHandlerState.h>

namespace CuEira {

class DataHandlerMock: public DataHandler {
public:
  DataHandlerMock() :
      DataHandler() { //FIXME WTF?

  }

  virtual ~DataHandlerMock() {

  }

  MOCK_CONST_METHOD0(getCurrentModelInformation, const Model::ModelInformation&());
  MOCK_CONST_METHOD0(getCurrentSNP, const SNP&());
  MOCK_CONST_METHOD0(getCurrentEnvironmentFactor, const EnvironmentFactor&());
  MOCK_CONST_METHOD0(getRecode, Recode());
  MOCK_CONST_METHOD0(getSNPVector, const Container::SNPVector&());
  MOCK_CONST_METHOD0(getInteractionVector, const Container::InteractionVector&());
  MOCK_CONST_METHOD0(getEnvironmentVector, const Container::EnvironmentVector&());

  MOCK_METHOD0(next, DataHandlerState());
  MOCK_METHOD1(recode, void(Recode));
};

} /* namespace CuEira */

#endif /* DATAHANDLERMOCK_H_ */
