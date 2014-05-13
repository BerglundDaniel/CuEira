#ifndef PERSONHANDLERMOCK_H_
#define PERSONHANDLERMOCK_H_

#include <gmock/gmock.h>
#include <string>
#include <iostream>

#include <PersonHandler.h>
#include <Id.h>
#include <Person.h>

namespace CuEira {

class PersonHandlerMock : public PersonHandler{
public:
  PersonHandlerMock():PersonHandler(){

  }

  virtual ~PersonHandlerMock(){

  }

  MOCK_METHOD4(createPerson, const Person&(Id id, Sex sex, Phenotype phenotype, int));

  MOCK_CONST_METHOD0(getNumberOfIndividualsTotal, int());
  MOCK_CONST_METHOD0(getNumberOfIndividualsToInclude, int());

  MOCK_METHOD0(createOutcomes, void());
  MOCK_CONST_METHOD0(getOutcomes, const Container::HostVector&());

  MOCK_CONST_METHOD1(getPersonFromId, const Person&(Id));
  MOCK_CONST_METHOD1(getPersonFromRowAll, const Person&(int));
  MOCK_CONST_METHOD1(getPersonFromRowInclude, const Person&(int));
  MOCK_CONST_METHOD1(getRowIncludeFromPerson, int(const Person&));

};

} /* namespace CuEira */

#endif /* PERSONHANDLERMOCK_H_ */
