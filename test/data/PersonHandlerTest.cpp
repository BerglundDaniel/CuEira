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

  personHandler.createPerson(person->getId(), person->getSex(), person->getPhenotype(), 0);
  ASSERT_THROW(personHandler.createPerson(person->getId(), person->getSex(), person->getPhenotype(), 0),
      PersonHandlerException);

  delete person;
}

TEST_F(PersonHandlerTest, ExceptionSamePersonDifferentRow) {
  Person* person = constructorHelpers.constructPersonInclude(1);

  personHandler.createPerson(person->getId(), person->getSex(), person->getPhenotype(), 0);
  ASSERT_THROW(personHandler.createPerson(person->getId(), person->getSex(), person->getPhenotype(), 1),
      PersonHandlerException);

  delete person;
}

TEST_F(PersonHandlerTest, ExceptionDifferentPersonSameRow) {
  Person* person1 = constructorHelpers.constructPersonInclude(1);
  Person* person2 = constructorHelpers.constructPersonInclude(2);

  personHandler.createPerson(person1->getId(), person1->getSex(), person1->getPhenotype(), 0);
  ASSERT_THROW(personHandler.createPerson(person2->getId(), person2->getSex(), person2->getPhenotype(), 0),
      PersonHandlerException);

  delete person1;
  delete person2;
}

TEST_F(PersonHandlerTest, NumberOfIndividuals) {
  int numberOfIndividuals = 10;
  int numberOfIndividualsNotInclude = 4;
  int notInclude[4] = {1, 2, 5, 7};
  int j = 0;
  std::vector<const Person*> personVector(numberOfIndividuals);

  for(int i = 0; i < numberOfIndividuals; ++i){
    Person* person;
    if(i == notInclude[j]){
      ++j;
      person = constructorHelpers.constructPersonNotInclude(i);
    }else{
      person = constructorHelpers.constructPersonInclude(i);
    }
    const Person& person2 = personHandler.createPerson(person->getId(), person->getSex(), person->getPhenotype(), i);
    delete person;
    personVector[i] = &person2;
  }

  ASSERT_EQ(numberOfIndividuals, personHandler.getNumberOfIndividualsTotal());
  ASSERT_EQ(numberOfIndividuals - numberOfIndividualsNotInclude, personHandler.getNumberOfIndividualsToInclude());
}

TEST_F(PersonHandlerTest, Getters) {
  int numberOfIndividuals = 10;
  int numberOfIndividualsNotInclude = 4;
  int notInclude[4] = {0, 3, 5, 8};
  int j = 0;
  std::vector<const Person*> personVector(numberOfIndividuals);

  for(int i = 0; i < numberOfIndividuals; ++i){
    Person* person;
    if(i == notInclude[j]){
      ++j;
      person = constructorHelpers.constructPersonNotInclude(i);
    }else{
      person = constructorHelpers.constructPersonInclude(i);
    }
    const Person& person2 = personHandler.createPerson(person->getId(), person->getSex(), person->getPhenotype(), i);
    delete person;
    personVector[i] = &person2;
  }

  int rowInclude = 0;
  j = 0;
  for(int i = 0; i < numberOfIndividuals; ++i){
    const Person& person = personVector[i];

    //Id to person
    ASSERT_EQ(person, personHandler.getPersonFromId(person.getId()));

    //Plink row to person
    ASSERT_EQ(person, personHandler.getPersonFromRowAll(i));

    if(i == notInclude[j]){
      j++;
    }else{
      //Row include to person
      ASSERT_EQ(person, personHandler.getPersonFromRowInclude(rowInclude));

      //Person to row include
      ASSERT_EQ(rowInclude, personHandler.getRowIncludeFromPerson(person));

      rowInclude++;
    }
  }
}

TEST_F(PersonHandlerTest, GettersException) {
  int numberOfIndividuals = 10;
  int numberOfIndividualsNotInclude = 4;
  int notInclude[4] = {0, 3, 5, 8};
  int j = 0;
  std::vector<const Person*> personVector(numberOfIndividuals);

  for(int i = 0; i < numberOfIndividuals; ++i){
    Person* person;
    if(i == notInclude[j]){
      ++j;
      person = constructorHelpers.constructPersonNotInclude(i);
    }else{
      person = constructorHelpers.constructPersonInclude(i);
    }
    const Person& person2 = personHandler.createPerson(person->getId(), person->getSex(), person->getPhenotype(), i);
    delete person;
    personVector[i] = &person2;
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
}

TEST_F(PersonHandlerTest, GetOutcomesException) {
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
    const Person& person2 = personHandler.createPerson(person->getId(), person->getSex(), person->getPhenotype(), i);
    delete person;
    personVector[i] = &person2;
  }

  ASSERT_THROW(personHandler.getOutcomes(), InvalidState);
}

TEST_F(PersonHandlerTest, CreateAndGetOutcomes) {
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
      if(i < 4 || i == 8){
        person = constructorHelpers.constructPersonInclude(i, AFFECTED);
      }else{
        person = constructorHelpers.constructPersonInclude(i, UNAFFECTED);
      }
    }
    const Person& person2 = personHandler.createPerson(person->getId(), person->getSex(), person->getPhenotype(), i);
    delete person;
    personVector[i] = &person2;
  }

  personHandler.createOutcomes();
  const Container::HostVector& outcomes = personHandler.getOutcomes();

  for(int i = 0; i < 6; ++i){
    if(i == 0 || i == 1 || i == 4){
      EXPECT_EQ(1, outcomes(i));
    }else{
      EXPECT_EQ(0, outcomes(i));
    }
  }
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */

