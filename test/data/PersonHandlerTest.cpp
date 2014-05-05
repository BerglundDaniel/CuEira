#include <gmock/gmock.h>
#include <gtest/gtest.h>

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

  PersonHandler personHandler;
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
  Id id("person1");
  Person person(id, MALE, AFFECTED);
  personHandler.addPerson(person,0);

  ASSERT_THROW(personHandler.addPerson(person,0), PersonHandlerException);
}

TEST_F(PersonHandlerTest, ExceptionSamePersonDifferentRow){
  Id id("person1");
  Person person(id, MALE, AFFECTED);
  personHandler.addPerson(person,0);

  ASSERT_THROW(personHandler.addPerson(person,1), PersonHandlerException);
}

TEST_F(PersonHandlerTest, ExceptionDifferentPersonSameRow){
  Id id1("person1");
  Person person1(id1, MALE, AFFECTED);

  Id id2("person2");
  Person person2(id2, FEMALE, AFFECTED);

  personHandler.addPerson(person1,0);

  ASSERT_THROW(personHandler.addPerson(person2,0), PersonHandlerException);
}

} /* namespace CuEira_Test */
} /* namespace CuEira */

