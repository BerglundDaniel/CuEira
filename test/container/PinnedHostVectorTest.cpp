#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>

#include <PinnedHostVector.h>
#include <HostVector.h>

namespace CuEira {
namespace Container {

using namespace CuEira::Container;

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PinnedHostVectorTest: public ::testing::Test {
protected:
  PinnedHostVectorTest();
  virtual ~PinnedHostVectorTest();
  virtual void SetUp();
  virtual void TearDown();
};

PinnedHostVectorTest::PinnedHostVectorTest() {

}

PinnedHostVectorTest::~PinnedHostVectorTest() {

}

void PinnedHostVectorTest::SetUp() {

}

void PinnedHostVectorTest::TearDown() {

}

TEST_F(PinnedHostVectorTest, Getters) {
  const int size = 5;
  PinnedHostVector pinnedVector(size);

  ASSERT_EQ(1, pinnedVector.getNumberOfColumns());
  ASSERT_EQ(size, pinnedVector.getNumberOfRows());
}

TEST_F(PinnedHostVectorTest, AccessOperator) {
  const int size = 5;
  PinnedHostVector pinnedVector(size);

  PRECISION a = 5;
  PRECISION b = 3.2;

  pinnedVector(0) = a;
  pinnedVector(3) = b;

  ASSERT_EQ(a, pinnedVector(0));
  ASSERT_EQ(b, pinnedVector(3));
}

}
/* namespace Container */
} /* namespace CuEira */

