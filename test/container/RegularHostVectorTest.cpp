#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>
#include <math.h>

#include <RegularHostVector.h>
#include <HostVector.h>
#include <DimensionMismatch.h>

namespace CuEira {
namespace CuEira_Test {

using namespace CuEira::Container;

/**
 * Test for testing RegularHostVector
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class RegularHostVectorTest: public ::testing::Test {
protected:
  RegularHostVectorTest();
  virtual ~RegularHostVectorTest();
  virtual void SetUp();
  virtual void TearDown();
};

RegularHostVectorTest::RegularHostVectorTest() {

}

RegularHostVectorTest::~RegularHostVectorTest() {

}

void RegularHostVectorTest::SetUp() {

}

void RegularHostVectorTest::TearDown() {

}

TEST_F(RegularHostVectorTest, Getters) {
  const int size = 5;
  RegularHostVector hostVector(size);

  ASSERT_EQ(1, hostVector.getNumberOfColumns());
  ASSERT_EQ(size, hostVector.getNumberOfRows());
}

TEST_F(RegularHostVectorTest, AccessOperator) {
  const int size = 5;
  RegularHostVector hostVector(size);

  PRECISION a = 5;
  PRECISION b = 3.2;

  hostVector(0) = a;
  hostVector(3) = b;

  ASSERT_EQ(a, hostVector(0));
  ASSERT_EQ(b, hostVector(3));
}

TEST_F(RegularHostVectorTest, Exceptions) {
  ASSERT_THROW(RegularHostVector(-1), DimensionMismatch);
  ASSERT_THROW(RegularHostVector(0), DimensionMismatch);

  const int size = 5;
  RegularHostVector hostVector(size);

  ASSERT_THROW(hostVector(-1), DimensionMismatch);
  ASSERT_THROW(hostVector(5), DimensionMismatch);
}

TEST_F(RegularHostVectorTest, Size) {
  const int size = 8;
  RegularHostVector hostVector(size);

  ASSERT_EQ(1, hostVector.getNumberOfColumns());
  ASSERT_EQ(size, hostVector.getNumberOfRows());

  const int newSize = 5;
  hostVector.updateSize(newSize);
  ASSERT_EQ(newSize, hostVector.getNumberOfRows());

  const int realSize = ceil(((double) size) / CPU_UNROLL) * CPU_UNROLL;
  ASSERT_EQ(realSize, hostVector.getRealNumberOfRows());
  ASSERT_EQ(1, hostVector.getRealNumberOfColumns());
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */

