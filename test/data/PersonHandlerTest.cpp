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
  Person* person=constructorHelpers.constructPersonInclude(1);

  personHandler.addPerson(*person,0);
  ASSERT_THROW(personHandler.addPerson(*person,0), PersonHandlerException);

  delete person;
}

TEST_F(PersonHandlerTest, ExceptionSamePersonDifferentRow){
  Person* person=constructorHelpers.constructPersonInclude(1);

  personHandler.addPerson(*person,0);
  ASSERT_THROW(personHandler.addPerson(*person,1), PersonHandlerException);

  delete person;
}

TEST_F(PersonHandlerTest, ExceptionDifferentPersonSameRow){
  Person* person1=constructorHelpers.constructPersonInclude(1);
  Person* person2=constructorHelpers.constructPersonInclude(2);

  personHandler.addPerson(*person1,0);
  ASSERT_THROW(personHandler.addPerson(*person2,0), PersonHandlerException);

  delete person1;
  delete person2;
}

TEST_F(PersonHandlerTest, NumberOfIndividuals){
  int numberOfIndividuals=10;
  int numberOfIndividualsNotInclude=4;
  int notInclude[4]={1,2,5,7};
  int j=0;
  std::vector<Person*> personVector(numberOfIndividuals);

  for(int i=0;i<numberOfIndividuals;++i){
    Person* person;
    if(i==notInclude[j]){
      ++j;
      person=constructorHelpers.constructPersonNotInclude(i);
    } else{
      person=constructorHelpers.constructPersonInclude(i);
    }
    personHandler.addPerson(*person, i);
    personVector[i]=person;
  }

  ASSERT_EQ(numberOfIndividuals, personHandler.getNumberOfIndividualsTotal());
  ASSERT_EQ(numberOfIndividuals-numberOfIndividualsNotInclude, personHandler.getNumberOfIndividualsToInclude());

  for(int i=0;i<numberOfIndividuals;++i){
    delete personVector[i];
  }
}

TEST_F(PersonHandlerTest, getters){
  int numberOfIndividuals=10;
  int numberOfIndividualsNotInclude=4;
  int notInclude[4]={0,3,5,8};
  int j=0;
  std::vector<Person*> personVector(numberOfIndividuals);

  for(int i=0;i<numberOfIndividuals;++i){
    Person* person;
    if(i==notInclude[j]){
      ++j;
      person=constructorHelpers.constructPersonNotInclude(i);
    } else{
      person=constructorHelpers.constructPersonInclude(i);
    }
    personHandler.addPerson(*person, i);
    personVector[i]=person;
  }

  for(int i=0;i<numberOfIndividuals;++i){
    Person* person=personVector[i];
    ASSERT_EQ(*person,personHandler.getPersonFromId(person->getId()));
    delete person;
  }

  /*const Person& getPersonFromId(Id id)

   const Person& getPersonFromRowAll(int row) const;

   const Person& getPersonFromRowInclude(int row) const;

   int getRowIncludeFromPerson(Person& person) const;*/
}

} /* namespace CuEira_Test */
} /* namespace CuEira */

