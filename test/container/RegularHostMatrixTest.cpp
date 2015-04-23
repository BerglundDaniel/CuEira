#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>

#include <RegularHostVector.h>
#include <RegularHostMatrix.h>
#include <HostVector.h>
#include <HostMatrix.h>
#include <DimensionMismatch.h>

namespace CuEira {
namespace CuEira_Test {

using namespace CuEira::Container;

/**
 * Test for testing RegularHostMatrix
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class RegularHostMatrixTest: public ::testing::Test {
protected:
  RegularHostMatrixTest();
  virtual ~RegularHostMatrixTest();
  virtual void SetUp();
  virtual void TearDown();

  int rows;
  int cols;
  RegularHostMatrix* hostMatrix;
};

RegularHostMatrixTest::RegularHostMatrixTest() :
    rows(5), cols(3), hostMatrix(new RegularHostMatrix(rows, cols)) {

}

RegularHostMatrixTest::~RegularHostMatrixTest() {

}

void RegularHostMatrixTest::SetUp() {
  hostMatrix = new RegularHostMatrix(rows, cols);
}

void RegularHostMatrixTest::TearDown() {
  delete hostMatrix;
}

TEST_F(RegularHostMatrixTest, Getters) {
  EXPECT_EQ(cols, hostMatrix->getNumberOfColumns());
  EXPECT_EQ(rows, hostMatrix->getNumberOfRows());
}

TEST_F(RegularHostMatrixTest, AccessOperator) {
  PRECISION a = 5;
  PRECISION b = 3.2;

  (*hostMatrix)(0, 1) = a;
  (*hostMatrix)(3, 2) = b;

  EXPECT_EQ(a, (*hostMatrix)(0, 1));
  EXPECT_EQ(b, (*hostMatrix)(3, 2));
}

TEST_F(RegularHostMatrixTest, AccessOperatorColumn) {
  PRECISION a = 5;
  PRECISION b = 3.2;
  PRECISION c = 2.1;

  HostVector* colVector = (*hostMatrix)(1);
  (*hostMatrix)(0, 1) = a;
  (*hostMatrix)(3, 1) = b;

  EXPECT_EQ(a, (*colVector)(0));
  EXPECT_EQ(b, (*colVector)(3));

  (*colVector)(3) = c;
  EXPECT_EQ(c, (*hostMatrix)(3, 1));

  delete colVector;
}

TEST_F(RegularHostMatrixTest, AccessOperatorColumnMemory) {
  PRECISION a = 5;
  PRECISION b = 3.2;
  PRECISION c = 2.1;

  HostVector* colVector = (*hostMatrix)(1);
  (*hostMatrix)(0, 1) = a;
  (*hostMatrix)(3, 1) = b;
  (*colVector)(2) = c;

  delete colVector;

  EXPECT_EQ(a, (*hostMatrix)(0, 1));
  EXPECT_EQ(b, (*hostMatrix)(3, 1));
  EXPECT_EQ(c, (*hostMatrix)(2, 1));
}

TEST_F(RegularHostMatrixTest, Exception) {
  EXPECT_THROW(RegularHostMatrix(0, 1), DimensionMismatch);
  EXPECT_THROW(RegularHostMatrix(1, 0), DimensionMismatch);
  EXPECT_THROW(RegularHostMatrix(-1, 1), DimensionMismatch);
  EXPECT_THROW(RegularHostMatrix(1, -1), DimensionMismatch);

  const int row = 3;
  const int col = 5;

  RegularHostMatrix hostMatrix(row, col);

  EXPECT_THROW(hostMatrix(-1), DimensionMismatch);
  EXPECT_THROW(hostMatrix(1, -1), DimensionMismatch);
  EXPECT_THROW(hostMatrix(-1, 1), DimensionMismatch);
}

TEST_F(RegularHostMatrixTest, Size) {
  const int row = 8;
  const int col = 5;
  RegularHostMatrix hostMatrix(row, col);

  ASSERT_EQ(col, hostMatrix.getNumberOfColumns());
  ASSERT_EQ(size, hostMatrix.getNumberOfRows());

  const int newSize = 5;
  hostMatrix.updateSize(newSize);
  ASSERT_EQ(newSize, hostMatrix.getNumberOfRows());

  const int realSize = ceil(((double) size) / CPU_UNROLL) * CPU_UNROLL;
  ASSERT_EQ(realSize, hostMatrix.getRealNumberOfRows());
  ASSERT_EQ(col, hostMatrix.getRealNumberOfColumns());
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */

