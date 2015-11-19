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
#include <PersonHandlerLocked.h>
#include <PersonHandlerException.h>

namespace CuEira {
namespace CuEira_Test {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PersonHandlerLockedTest: public ::testing::Test {
protected:
  PersonHandlerLockedTest();
  virtual ~PersonHandlerLockedTest();
  virtual void SetUp();
  virtual void TearDown();
};

PersonHandlerLockedTest::PersonHandlerLockedTest(){

}

PersonHandlerLockedTest::~PersonHandlerLockedTest(){

}

void PersonHandlerLockedTest::SetUp(){

}

void PersonHandlerLockedTest::TearDown(){

}

TEST_F(PersonHandlerLockedTest, Getters){
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
    } else{
      sex = FEMALE;
    }

    int r = rand() % 3;
    if(r == 0){
      phenotype = AFFECTED;
      numberOfIndividualsToInclude++;
    } else if(r == 1){
      phenotype = UNAFFECTED;
      numberOfIndividualsToInclude++;
    } else{
      phenotype = MISSING;
    }

    (*personVector)[i] = new Person(id, sex, phenotype, phenotype != MISSING);
  }

  PersonHandler personHandler(personVector);
  PersonHandlerLocked personHandlerLocked(personHandler);

  ASSERT_EQ(numberOfIndividualsTotal, personHandlerLocked.getNumberOfIndividualsTotal());
  for(int i = 0; i < numberOfIndividuals; ++i){
    const Person& person = *(*personVector)[i];

    ASSERT_EQ(person, personHandlerLocked.getPersonFromId(person.getId()));

    ASSERT_EQ(person, personHandlerLocked.getPersonFromRowAll(i));
  }

  int includeNumber = 0;
  for(int i = 0; i < numberOfIndividuals; ++i){
    Person* person = (*personVector)[i];

    if(person->getInclude()){
      includeNumber++;
      ASSERT_EQ(includeNumber, personHandlerLocked.getRowIncludeFromPerson(*person));
    }
  }
  ASSERT_EQ(includeNumber, personHandlerLocked.getNumberOfIndividualsToInclude());

  int i=0;
  for(PersonHandlerLocked::iterator iter = personHandlerLocked.begin(); iter=personHandlerLocked.end(); iter++){
    ASSERT_EQ(*(*iter), *(*personVector)[i]);
    ++i;
  }
}

TEST_F(PersonHandlerLockedTest, GettersException){
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
    } else{
      sex = FEMALE;
    }

    int r = rand() % 3;
    if(r == 0){
      phenotype = AFFECTED;
      numberOfIndividualsToInclude++;
    } else if(r == 1){
      phenotype = UNAFFECTED;
      numberOfIndividualsToInclude++;
    } else{
      phenotype = MISSING;
    }

    (*personVector)[i] = new Person(id, sex, phenotype, phenotype != MISSING);
  }

  PersonHandler personHandler(personVector);
  PersonHandlerLocked personHandlerLocked(personHandler);

  ASSERT_EQ(numberOfIndividualsTotal, personHandlerLocked.getNumberOfIndividualsTotal());

  Person personNotInHandler(Id("other"), FEMALE, UNAFFECTED, true);

  ASSERT_THROW(personHandlerLocked.getPersonFromId(personNotInHandler.getId()), PersonHandlerException);
  ASSERT_THROW(personHandlerLocked.getRowIncludeFromPerson(personNotInHandler), PersonHandlerException);
  ASSERT_THROW(personHandlerLocked.getPersonFromRowAll(11), PersonHandlerException);
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */

