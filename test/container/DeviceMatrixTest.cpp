#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>

#include <DeviceMatrix.h>
#include <DeviceVector.h>

namespace CuEira {
namespace Container {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DeviceMatrixTest: public ::testing::Test {
protected:
  DeviceMatrixTest();
  virtual ~DeviceMatrixTest();
  virtual void SetUp();
  virtual void TearDown();
};

DeviceMatrixTest::DeviceMatrixTest() {

}

DeviceMatrixTest::~DeviceMatrixTest() {

}

void DeviceMatrixTest::SetUp() {

}

void DeviceMatrixTest::TearDown() {

}

TEST_F(DeviceMatrixTest, Getters) {
  const int numberOfRows = 5;
  const int numberOfColumns = 3;

  DeviceMatrix deviceMatrix(numberOfRows, numberOfColumns);

  ASSERT_EQ(numberOfColumns, deviceMatrix.getNumberOfColumns());
  ASSERT_EQ(numberOfRows, deviceMatrix.getNumberOfRows());
}

TEST_F(DeviceMatrixTest, AccessOperator) {
  const int numberOfRows = 5;
  const int numberOfColumns = 3;

  DeviceMatrix deviceMatrix(numberOfRows, numberOfColumns);
  PRECISION* memPtr = deviceMatrix.getMemoryPointer();

  ASSERT_EQ(memPtr, deviceMatrix(0, 0));
  ASSERT_EQ(memPtr + 3, deviceMatrix(3, 0));
  ASSERT_EQ(memPtr + (3 * numberOfRows) + 1, deviceMatrix(1, 3));
}

TEST_F(DeviceMatrixTest, AccessOperatorColumn) {
  const int numberOfRows = 5;
  const int numberOfColumns = 3;

  DeviceMatrix deviceMatrix(numberOfRows, numberOfColumns);
  DeviceVector* deviceVector1 = deviceMatrix(1);

  ASSERT_EQ(numberOfRows, deviceVector1->getNumberOfRows());
  ASSERT_EQ(1, deviceVector1->getNumberOfColumns());

  EXPECT_EQ((*deviceVector1)(0), deviceMatrix(0, 1));
  EXPECT_EQ((*deviceVector1)(3), deviceMatrix(3, 1));
  EXPECT_EQ((*deviceVector1)(4), deviceMatrix(4, 1));
}

}
/* namespace Container */
} /* namespace CuEira */

