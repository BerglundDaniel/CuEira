#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>

#include <HostVector.h>
#include <HostMatrix.h>
#include <PinnedHostMatrix.h>
#include <PinnedHostVector.h>

namespace CuEira {
namespace Container {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PinnedHostMatrixTest: public ::testing::Test {
protected:
  PinnedHostMatrixTest();
  virtual ~PinnedHostMatrixTest();
  virtual void SetUp();
  virtual void TearDown();

  int rows;
  int cols;
  PinnedHostMatrix* hostMatrix;
};

PinnedHostMatrixTest::PinnedHostMatrixTest() :
    rows(5), cols(3) {

}

PinnedHostMatrixTest::~PinnedHostMatrixTest() {

}

void PinnedHostMatrixTest::SetUp() {
  hostMatrix = new PinnedHostMatrix(rows, cols);
}

void PinnedHostMatrixTest::TearDown() {
  delete hostMatrix;
}

TEST_F(PinnedHostMatrixTest, Getters) {
  EXPECT_EQ(cols, hostMatrix->getNumberOfColumns());
  EXPECT_EQ(rows, hostMatrix->getNumberOfRows());
}

TEST_F(PinnedHostMatrixTest, AccessOperator) {
  PRECISION a = 5;
  PRECISION b = 3.2;

  (*hostMatrix)(0, 1) = a;
  (*hostMatrix)(3, 2) = b;

  EXPECT_EQ(a, (*hostMatrix)(0, 1));
  EXPECT_EQ(b, (*hostMatrix)(3, 2));
}

TEST_F(PinnedHostMatrixTest, AccessOperatorColumn) {
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

TEST_F(PinnedHostMatrixTest, AccessOperatorColumnMemory) {
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

TEST_F(PinnedHostMatrixTest, Size) {
  const int row = 8;
  const int col = 5;
  PinnedHostMatrix hostMatrix(row, col);

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
/* namespace Container */
} /* namespace CuEira */

