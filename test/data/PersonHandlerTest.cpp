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

TEST_F(PersonHandlerTest, ExceptionSamePersonSameRow) {
  Person* person = constructorHelpers.constructPersonInclude(1);

  personHandler.addPerson(*person, 0);
  ASSERT_THROW(personHandler.addPerson(*person, 0), PersonHandlerException);

  delete person;
}

TEST_F(PersonHandlerTest, ExceptionSamePersonDifferentRow) {
  Person* person = constructorHelpers.constructPersonInclude(1);

  personHandler.addPerson(*person, 0);
  ASSERT_THROW(personHandler.addPerson(*person, 1), PersonHandlerException);

  delete person;
}

TEST_F(PersonHandlerTest, ExceptionDifferentPersonSameRow) {
  Person* person1 = constructorHelpers.constructPersonInclude(1);
  Person* person2 = constructorHelpers.constructPersonInclude(2);

  personHandler.addPerson(*person1, 0);
  ASSERT_THROW(personHandler.addPerson(*person2, 0), PersonHandlerException);

  delete person1;
  delete person2;
}

TEST_F(PersonHandlerTest, NumberOfIndividuals) {
  int numberOfIndividuals = 10;
  int numberOfIndividualsNotInclude = 4;
  int notInclude[4] = {1, 2, 5, 7};
  int j = 0;
  std::vector<Person*> personVector(numberOfIndividuals);

  for(int i = 0; i < numberOfIndividuals; ++i){
    Person* person;
    if(i == notInclude[j]){
      ++j;
      person = constructorHelpers.constructPersonNotInclude(i);
    }else{
      person = constructorHelpers.constructPersonInclude(i);
    }
    personHandler.addPerson(*person, i);
    personVector[i] = person;
  }

  ASSERT_EQ(numberOfIndividuals, personHandler.getNumberOfIndividualsTotal());
  ASSERT_EQ(numberOfIndividuals - numberOfIndividualsNotInclude, personHandler.getNumberOfIndividualsToInclude());

  for(int i = 0; i < numberOfIndividuals; ++i){
    delete personVector[i];
  }
}

TEST_F(PersonHandlerTest, Getters) {
  int numberOfIndividuals = 10;
  int numberOfIndividualsNotInclude = 4;
  int notInclude[4] = {0, 3, 5, 8};
  int j = 0;
  std::vector<Person*> personVector(numberOfIndividuals);

  for(int i = 0; i < numberOfIndividuals; ++i){
    Person* person;
    if(i == notInclude[j]){
      ++j;
      person = constructorHelpers.constructPersonNotInclude(i);
    }else{
      person = constructorHelpers.constructPersonInclude(i);
    }
    personHandler.addPerson(*person, i);
    personVector[i] = person;
  }

  int rowInclude = 0;
  j = 0;
  for(int i = 0; i < numberOfIndividuals; ++i){
    Person* person = personVector[i];

    //Id to person
    ASSERT_EQ(*person, personHandler.getPersonFromId(person->getId()));

    //Plink row to person
    ASSERT_EQ(*person, personHandler.getPersonFromRowAll(i));

    if(i == notInclude[j]){
      j++;
    }else{
      //Row include to person
      ASSERT_EQ(*person, personHandler.getPersonFromRowInclude(rowInclude));

      //Person to row include
      ASSERT_EQ(rowInclude, personHandler.getRowIncludeFromPerson(*person));

      rowInclude++;
    }

    delete person;
  }
}

TEST_F(PersonHandlerTest, GettersException) {
  int numberOfIndividuals = 10;
  int numberOfIndividualsNotInclude = 4;
  int notInclude[4] = {0, 3, 5, 8};
  int j = 0;
  std::vector<Person*> personVector(numberOfIndividuals);

  for(int i = 0; i < numberOfIndividuals; ++i){
    Person* person;
    if(i == notInclude[j]){
      ++j;
      person = constructorHelpers.constructPersonNotInclude(i);
    }else{
      person = constructorHelpers.constructPersonInclude(i);
    }
    personHandler.addPerson(*person, i);
    personVector[i] = person;
  }

  int personNumber = 22;
  Person* personNotInHandler = constructorHelpers.constructPersonInclude(personNumber);

  //Id to person
  ASSERT_THROW(personHandler.getPersonFromId(personNotInHandler->getId()), PersonHandlerException);

  //Plink row to person
  ASSERT_THROW(personHandler.getPersonFromRowAll(personNumber), PersonHandlerException);

  //Row include to person
  ASSERT_THROW(personHandler.getPersonFromRowInclude(personNumber), PersonHandlerException);

  //Person to row include
  ASSERT_THROW(personHandler.getRowIncludeFromPerson(*personNotInHandler), PersonHandlerException);

  for(int i = 0; i < numberOfIndividuals; ++i){
    Person* person = personVector[i];
    delete person;
  }
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */

