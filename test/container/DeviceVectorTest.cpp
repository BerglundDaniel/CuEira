#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>

#include <DeviceVector.h>

namespace CuEira {
namespace Container {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DeviceVectorTest: public ::testing::Test {
protected:
  DeviceVectorTest();
  virtual ~DeviceVectorTest();
  virtual void SetUp();
  virtual void TearDown();
};

DeviceVectorTest::DeviceVectorTest() {

}

DeviceVectorTest::~DeviceVectorTest() {

}

void DeviceVectorTest::SetUp() {

}

void DeviceVectorTest::TearDown() {

}

TEST_F(DeviceVectorTest, Getters) {
  const int size = 5;
  DeviceVector deviceVector(size);

  ASSERT_EQ(1, deviceVector.getNumberOfColumns());
  ASSERT_EQ(size, deviceVector.getNumberOfRows());
}

TEST_F(DeviceVectorTest, AccessOperator) {
  const int size = 5;
  DeviceVector deviceVector(size);
  PRECISION* memPtr=deviceVector.getMemoryPointer();

  ASSERT_EQ(memPtr, deviceVector(0));
  ASSERT_EQ(memPtr+3, deviceVector(3));
}

}
/* namespace Container */
} /* namespace CuEira */

