#include <gmock/gmock.h>
#include <gtest/gtest.h>

/*#include <Person.h>
#include <Sex.h>
#include <Id.h>
#include <Phenotype.h>
#include <PersonHandler.h>
#include <PersonHandlerException.h>*/

//using ::testing::Eq;
//using ::testing::Test;

//namespace CuEira {
//namespace CuEira_Test {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PersonHandlerTest : public ::testing::Test {
protected:
  PersonHandlerTest();
  virtual ~PersonHandlerTest();
  virtual void SetUp();
  virtual void TearDown();

  //PersonHandler personHandler;
};

PersonHandlerTest::PersonHandlerTest(){

}

PersonHandlerTest::~PersonHandlerTest(){

}

void PersonHandlerTest::SetUp(){

}

void PersonHandlerTest::TearDown(){

}

TEST_F(PersonHandlerTest, Bar){
  int a=5;
  //Person person(id, MALE, AFFECTED);
  //personHandler.addPerson();

  //EXPECT_THROW(personHandler.addPerson(), PersonHandlerException*);
}

//} /* namespace CuEira_Test */
//} /* namespace CuEira */

