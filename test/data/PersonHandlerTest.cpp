#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>

#include <Person.h>
#include <Sex.h>
#include <Id.h>
#include <Phenotype.h>
#include <PersonHandler.h>
#include <PersonHandlerException.h>
#include <ConstructorHelpers.h>

namespace CuEira {
namespace CuEira_Test {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PersonHandlerTest: public ::testing::Test {
protected:
  PersonHandlerTest();
  virtual ~PersonHandlerTest();
  virtual void SetUp();
  virtual void TearDown();

  PersonHandler personHandler;
  ConstructorHelpers constructorHelpers;
};

PersonHandlerTest::PersonHandlerTest() {

}

PersonHandlerTest::~PersonHandlerTest() {

}

void PersonHandlerTest::SetUp() {

}

void PersonHandlerTest::TearDown() {

}

TEST_F(PersonHandlerTest, ExceptionSamePersonSameRow){
  Person person=constructorHelpers.constructPersonInclude(1);

  personHandler.addPerson(person,0);
  ASSERT_THROW(personHandler.addPerson(person,0), PersonHandlerException);
}

TEST_F(PersonHandlerTest, ExceptionSamePersonDifferentRow){
  Person person=constructorHelpers.constructPersonInclude(1);

  personHandler.addPerson(person,0);
  ASSERT_THROW(personHandler.addPerson(person,1), PersonHandlerException);
}

TEST_F(PersonHandlerTest, ExceptionDifferentPersonSameRow){
  Person person1=constructorHelpers.constructPersonInclude(1);
  Person person2=constructorHelpers.constructPersonInclude(2);

  personHandler.addPerson(person1,0);
  ASSERT_THROW(personHandler.addPerson(person2,0), PersonHandlerException);
}

TEST_F(PersonHandlerTest, NumberOfIndividuals){
  int numberOfIndividuals=10;
  int numberOfIndividualsNotInclude=4;
  int notInclude[4]={1,2,5,7};
  int j=0;

  for(int i=0;i<numberOfIndividuals;++i){
    if(i==notInclude[j]){
      ++j;
      personHandler.addPerson(constructorHelpers.constructPersonNotInclude(i), i);
    } else{
      personHandler.addPerson(constructorHelpers.constructPersonInclude(i), i);
    }
  }

  ASSERT_EQ(numberOfIndividuals, personHandler.getNumberOfIndividualsTotal());
  ASSERT_EQ(numberOfIndividuals-numberOfIndividualsNotInclude, personHandler.getNumberOfIndividualsToInclude());
}

} /* namespace CuEira_Test */
} /* namespace CuEira */

