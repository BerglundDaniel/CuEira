#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>

#include <EnvironmentFactor.h>
#include <Id.h>
#include <VariableType.h>

namespace CuEira {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class EnvironmentFactorTest: public ::testing::Test {
protected:
  EnvironmentFactorTest();
  virtual ~EnvironmentFactorTest();
  virtual void SetUp();
  virtual void TearDown();
};

EnvironmentFactorTest::EnvironmentFactorTest() {

}

EnvironmentFactorTest::~EnvironmentFactorTest() {

}

void EnvironmentFactorTest::SetUp() {

}

void EnvironmentFactorTest::TearDown() {

}

TEST_F(EnvironmentFactorTest, Getters) {
  Id id1("env1");
  EnvironmentFactor envFactor1(id1);
  EXPECT_EQ(id1, envFactor1.getId());
  EXPECT_TRUE(envFactor1.getInclude());

  envFactor1.setVariableType(OTHER);
  EXPECT_EQ(OTHER, envFactor1.getVariableType());

  envFactor1.setVariableType(BINARY);
  EXPECT_EQ(BINARY, envFactor1.getVariableType());
}

TEST_F(EnvironmentFactorTest, Operators) {
  Id id1("env1");
  EnvironmentFactor envFactor1(id1);

  Id id2("env2");
  EnvironmentFactor envFactor2(id2);

  Id id3("a_env");
  EnvironmentFactor envFactor3(id3);

  EnvironmentFactor envFactor4(id1);

  EXPECT_EQ(envFactor4, envFactor1);
  EXPECT_FALSE(envFactor1 == envFactor2);
  EXPECT_FALSE(envFactor4 == envFactor2);
  EXPECT_FALSE(envFactor1 == envFactor3);
  EXPECT_FALSE(envFactor4 == envFactor3);
  EXPECT_FALSE(envFactor3 == envFactor2);

  if(id1 < id2){
    EXPECT_TRUE(envFactor1 < envFactor2);
  }else{
    EXPECT_TRUE(envFactor2 < envFactor1);
  }

  if(id1 < id3){
    EXPECT_TRUE(envFactor1 < envFactor3);
  }else{
    EXPECT_TRUE(envFactor3 < envFactor1);
  }

  if(id2 < id3){
    EXPECT_TRUE(envFactor2 < envFactor3);
  }else{
    EXPECT_TRUE(envFactor3 < envFactor2);
  }
}

} /* namespace CuEira */

