#ifndef MODELINFORMATIOFACTORYNMOCK_H_
#define MODELINFORMATIOFACTORYNMOCK_H_

#include <gmock/gmock.h>

#include <ModelInformation.h>
#include <ModelInformationFactory.h>
#include <SNP.h>
#include <EnvironmentFactor.h>
#include <AlleleStatistics.h>
#include <ContingencyTable.h>

namespace CuEira {
namespace Model {

class ModelInformationFactoryMock: public ModelInformationFactory {
public:
  ModelInformationFactoryMock() :
      ModelInformationFactory() {

  }

  virtual ~ModelInformationFactoryMock() {

  }

  MOCK_CONST_METHOD3(constructModelInformation, ModelInformation*(const SNP&, const EnvironmentFactor&, const AlleleStatistics&));
  MOCK_CONST_METHOD4(constructModelInformation, ModelInformation*(const SNP&, const EnvironmentFactor&, const AlleleStatistics&, const ContingencyTable&));
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* MODELINFORMATIOFACTORYNMOCK_H_ */
