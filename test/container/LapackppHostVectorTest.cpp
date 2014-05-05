#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>
#include <lapackpp/lavd.h>

#include <LapackppHostVector.h>
#include <HostVector.h>

namespace CuEira {
namespace CuEira_Test {

using namespace CuEira::Container;

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LapackppHostVectorTest: public ::testing::Test {
protected:
  LapackppHostVectorTest();
  virtual ~LapackppHostVectorTest();
  virtual void SetUp();
  virtual void TearDown();
};

LapackppHostVectorTest::LapackppHostVectorTest() {

}

LapackppHostVectorTest::~LapackppHostVectorTest() {

}

void LapackppHostVectorTest::SetUp() {

}

void LapackppHostVectorTest::TearDown() {

}

TEST_F(LapackppHostVectorTest, Getters){
  const int size=5;
  LaVectorDouble laVector(size);
  LapackppHostVector hostVector(laVector);

  ASSERT_EQ(1,hostVector.getNumberOfColumns());
  ASSERT_EQ(size,hostVector.getNumberOfRows());
}

TEST_F(LapackppHostVectorTest, GetLapackpp){
  const int size=5;
  LaVectorDouble laVector(size);
  LapackppHostVector hostVector(laVector);

  LaVectorDouble& laVectorGet=hostVector.getLapackpp();

  ASSERT_EQ(size, laVectorGet.size());

  int a=5;
  int b=3.2;
  laVectorGet(0)=a;
  laVectorGet(2)=b;

  ASSERT_EQ(laVectorGet(0), hostVector(0));
  ASSERT_EQ(laVectorGet(2), hostVector(2));
  ASSERT_EQ(a, hostVector(0));
  ASSERT_EQ(b, hostVector(2));
}

TEST_F(LapackppHostVectorTest, AccessOperator){
  const int size=5;
  LaVectorDouble laVector(size);
  LapackppHostVector hostVector(laVector);

  int a=5;
  int b=3.2;

  hostVector(0)=a;
  hostVector(3)=b;

  ASSERT_EQ(a, hostVector(0));
  ASSERT_EQ(b, hostVector(3));
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */

