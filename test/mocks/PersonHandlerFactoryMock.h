#ifndef PERSONHANDLERFACTORYMOCK_H_
#define PERSONHANDLERFACTORYMOCK_H_

#include <gmock/gmock.h>
#include <vector>

#include <Person.h>
#include <PersonHandler.h>
#include <PersonHandlerFactory.h>

namespace CuEira {

class PersonHandlerFactoryMock: public PersonHandlerFactory {
public:
  PersonHandlerFactoryMock() :
      PersonHandlerFactory() {

  }

  virtual ~PersonHandlerFactoryMock() {

  }

  MOCK_CONST_METHOD1(constructPersonHandler, PersonHandler*(std::vector<Person*>*));

};

} /* namespace CuEira */

#endif /* PERSONHANDLERFACTORYMOCK_H_ */
