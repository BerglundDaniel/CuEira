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
namespace CuEira_Test {

/**
 * Test for testing ....
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

TEST_F(PersonTest, Getters){
  Id id1("Person1");
  Person person1(id1,MALE,UNAFFECTED);

  ASSERT_EQ(id1, person1.getId());
  ASSERT_TRUE(person1.getInclude());
  ASSERT_EQ(MALE, person1.getSex());
  ASSERT_EQ(UNAFFECTED, person1.getPhenotype());

  Id id2("Person2");
  Person person2(id2,FEMALE,AFFECTED);

  ASSERT_EQ(id2, person2.getId());
  ASSERT_TRUE(person2.getInclude());
  ASSERT_EQ(FEMALE, person2.getSex());
  ASSERT_EQ(AFFECTED, person2.getPhenotype());

  Id id3("Person3");
  Person person3(id3,MALE,MISSING);

  ASSERT_EQ(id3, person3.getId());
  ASSERT_FALSE(person3.getInclude());
  ASSERT_EQ(MALE, person3.getSex());
  ASSERT_EQ(MISSING, person3.getPhenotype());
}

TEST_F(PersonTest, Include){
  Id id1("Person1");
  Person person1(id1,FEMALE,UNAFFECTED);
  ASSERT_TRUE(person1.getInclude());

  Id id2("Person2");
  Person person2(id2,MALE,MISSING);
  ASSERT_FALSE(person2.getInclude());
}

TEST_F(PersonTest, Operators){
  Id id1("Person1");
  Person person1(id1,MALE,UNAFFECTED);
  Person person2(id1,MALE,UNAFFECTED);

  Id id3("Person3");
  Person person3(id3,FEMALE,AFFECTED);

  ASSERT_TRUE(person1==person2);
  ASSERT_FALSE(person1<person2);
  ASSERT_FALSE(person2<person1);

 if(id1<id3){
   ASSERT_TRUE(person1<person3);
 } else{
   ASSERT_TRUE(person3<person1);
 }

}

}
/* namespace CuEira_Test */
} /* namespace CuEira */

