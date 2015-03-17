#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>

#include <Person.h>
#include <Sex.h>
#include <Id.h>
#include <Phenotype.h>

namespace CuEira {

/**
 * Test for the Person class
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PersonTest: public ::testing::Test {
protected:
  PersonTest();
  virtual ~PersonTest();
  virtual void SetUp();
  virtual void TearDown();
};

PersonTest::PersonTest() {

}

PersonTest::~PersonTest() {

}

void PersonTest::SetUp() {

}

void PersonTest::TearDown() {

}

TEST_F(PersonTest, Getters) {
  Id id1("Person1");
  Person person1(id1, MALE, UNAFFECTED, true);

  ASSERT_EQ(id1, person1.getId());
  ASSERT_EQ(MALE, person1.getSex());
  ASSERT_EQ(UNAFFECTED, person1.getPhenotype());
  ASSERT_TRUE(person1.getInclude());
  person1.setInclude(false);
  ASSERT_FALSE(person1.getInclude());

  Id id2("Person2");
  Person person2(id2, FEMALE, AFFECTED, false);

  ASSERT_EQ(id2, person2.getId());
  ASSERT_EQ(FEMALE, person2.getSex());
  ASSERT_EQ(AFFECTED, person2.getPhenotype());
  ASSERT_FALSE(person2.getInclude());
  person2.setInclude(true);
  ASSERT_TRUE(person2.getInclude());

  Id id3("Person3");
  Person person3(id3, MALE, MISSING, true);

  ASSERT_EQ(id3, person3.getId());
  ASSERT_EQ(MALE, person3.getSex());
  ASSERT_EQ(MISSING, person3.getPhenotype());
  ASSERT_TRUE(person3.getInclude());
}

TEST_F(PersonTest, Operators) {
  Id id1("Person1");
  Person person1(id1, MALE, UNAFFECTED, true);
  Person person2(id1, MALE, UNAFFECTED, true);

  Id id3("Person3");
  Person person3(id3, FEMALE, AFFECTED, true);

  ASSERT_TRUE(person1 == person2);
  ASSERT_FALSE(person1 < person2);
  ASSERT_FALSE(person2 < person1);

  if(id1 < id3){
    ASSERT_TRUE(person1 < person3);
  }else{
    ASSERT_TRUE(person3 < person1);
  }

}

} /* namespace CuEira */

