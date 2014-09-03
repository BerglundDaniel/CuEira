#ifndef CONTINGENCYTABLEFACTORYMOCK_H_
#define CONTINGENCYTABLEFACTORYMOCK_H_

#include <vector>
#include <gmock/gmock.h>

#include <ContingencyTable.h>
#include <ContingencyTableFactory.h>
#include <HostVector.h>

namespace CuEira {

class ContingencyTableFactoryMock: public ContingencyTableFactory {
public:
  ContingencyTableFactoryMock(const Container::HostVector& outcomes) :
      ContingencyTableFactory(outcomes) {

  }

  virtual ~ContingencyTableFactoryMock() {

  }

  MOCK_CONST_METHOD2(constructContingencyTable, ContingencyTable*(const Container::SNPVector& snpVector,
          const Container::EnvironmentVector& environmentVector));
};

} /* namespace CuEira */

#endif /* CONTINGENCYTABLEFACTORYMOCK_H_ */
