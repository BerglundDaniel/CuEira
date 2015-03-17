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
};

PersonHandlerTest::PersonHandlerTest() {

}

PersonHandlerTest::~PersonHandlerTest() {

}

void PersonHandlerTest::SetUp() {

}

void PersonHandlerTest::TearDown() {

}

TEST_F(PersonHandlerTest, Getters) {
  int numberOfIndividualsTotal = 10;
  int numberOfIndividualsToInclude = 0;
  std::vector<Person*>* personVector = new std::vector<Person*>(numberOfIndividualsTotal);

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    std::ostringstream os;
    os << "ind" << i;
    Id id(os.str());
    Sex sex;
    Phenotype phenotype;

    if(rand() % 2 == 0){
      sex = MALE;
    }else{
      sex = FEMALE;
    }

    int r = rand() % 3;
    if(r == 0){
      phenotype = AFFECTED;
      numberOfIndividualsToInclude++;
    }else if(r == 1){
      phenotype = UNAFFECTED;
      numberOfIndividualsToInclude++;
    }else{
      phenotype = MISSING;
    }

    (*personVector)[i] = new Person(id, sex, phenotype, phenotype != MISSING);
  }

  PersonHandler personHandler(personVector);

  ASSERT_EQ(numberOfIndividualsTotal, personHandler.getNumberOfIndividualsTotal());
  for(int i = 0; i < numberOfIndividuals; ++i){
    const Person& person = *(*personVector)[i];

    ASSERT_EQ(person, personHandler.getPersonFromId(person.getId()));

    ASSERT_EQ(person, personHandler.getPersonFromRowAll(i));
  }

  //Lock
  personHandler.lockIndividuals();

  ASSERT_EQ(numberOfIndividualsToInclude, personHandler.getNumberOfIndividualsToInclude());

  int includeNumber = 0;
  for(int i = 0; i < numberOfIndividuals; ++i){
    Person* person = (*personVector)[i];

    if(person->getInclude()){
      includeNumber++;
      ASSERT_EQ(includeNumber, personHandler.getRowIncludeFromPerson(*person));
    }
  }
  ASSERT_EQ(includeNumber, personHandler.getNumberOfIndividualsToInclude());
}

TEST_F(PersonHandlerTest, LockException) {
  int numberOfIndividualsTotal = 10;
  int numberOfIndividualsToInclude = 0;
  std::vector<Person*>* personVector = new std::vector<Person*>(numberOfIndividualsTotal);

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    std::ostringstream os;
    os << "ind" << i;
    Id id(os.str());
    Sex sex;
    Phenotype phenotype;

    if(rand() % 2 == 0){
      sex = MALE;
    }else{
      sex = FEMALE;
    }

    int r = rand() % 3;
    if(r == 0){
      phenotype = AFFECTED;
      numberOfIndividualsToInclude++;
    }else if(r == 1){
      phenotype = UNAFFECTED;
      numberOfIndividualsToInclude++;
    }else{
      phenotype = MISSING;
    }

    (*personVector)[i] = new Person(id, sex, phenotype, phenotype != MISSING);
  }

  PersonHandler personHandler(personVector);

  ASSERT_THROW(personHandler.getNumberOfIndividualsToInclude(), PersonHandlerException);
  ASSERT_THROW(personHandler.getRowIncludeFromPerson(*(*personVector)[0]), PersonHandlerException);
}

TEST_F(PersonHandlerTest, GettersException) {
  int numberOfIndividualsTotal = 10;
  int numberOfIndividualsToInclude = 0;
  std::vector<Person*>* personVector = new std::vector<Person*>(numberOfIndividualsTotal);

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    std::ostringstream os;
    os << "ind" << i;
    Id id(os.str());
    Sex sex;
    Phenotype phenotype;

    if(rand() % 2 == 0){
      sex = MALE;
    }else{
      sex = FEMALE;
    }

    int r = rand() % 3;
    if(r == 0){
      phenotype = AFFECTED;
      numberOfIndividualsToInclude++;
    }else if(r == 1){
      phenotype = UNAFFECTED;
      numberOfIndividualsToInclude++;
    }else{
      phenotype = MISSING;
    }

    (*personVector)[i] = new Person(id, sex, phenotype, phenotype != MISSING);
  }

  PersonHandler personHandler(personVector);

  //Lock
  personHandler.lockIndividuals();

  ASSERT_EQ(numberOfIndividualsTotal, personHandler.getNumberOfIndividualsTotal());

  Person personNotInHandler(Id("other"), FEMALE, UNAFFECTED, true);

  ASSERT_THROW(personHandler.getPersonFromId(personNotInHandler.getId()), PersonHandlerException);
  ASSERT_THROW(personHandler.getRowIncludeFromPerson(personNotInHandler), PersonHandlerException);
  ASSERT_THROW(personHandler.getPersonFromRowAll(11), PersonHandlerException);
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */

