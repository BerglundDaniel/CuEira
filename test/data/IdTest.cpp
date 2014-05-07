#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>

#include <Id.h>

namespace CuEira {
namespace CuEira_Test {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class IdTest: public ::testing::Test {
protected:
  IdTest();
  virtual ~IdTest();
  virtual void SetUp();
  virtual void TearDown();
};

IdTest::IdTest() {

}

IdTest::~IdTest() {

}

void IdTest::SetUp() {

}

void IdTest::TearDown() {

}

TEST_F(IdTest, Getters) {
  std::string id1Str("Person1");
  Id id1(id1Str);

  ASSERT_EQ(id1Str, id1.getString());
}

TEST_F(IdTest, Operators) {
  std::string id1Str("Person1");
  Id id1(id1Str);

  std::string id2Str("Person2");
  Id id2(id2Str);

  Id id3(id1Str);

  ASSERT_TRUE(id1 == id3);

  ASSERT_FALSE(id1 < id3);
  ASSERT_FALSE(id3 < id1);

  if(id1Str < id2Str){
    ASSERT_TRUE(id1 < id2);
  }else{
    ASSERT_TRUE(id2 < id1);
  }
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */

