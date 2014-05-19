#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>
#include <lapackpp/lavd.h>
#include <lapackpp/gmd.h>
#include <LapackppHostVector.h>
#include <LapackppHostMatrix.h>
#include <HostVector.h>
#include <HostMatrix.h>

namespace CuEira {
namespace CuEira_Test {

using namespace CuEira::Container;

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LapackppHostMatrixTest: public ::testing::Test {
protected:
  LapackppHostMatrixTest();
  virtual ~LapackppHostMatrixTest();
  virtual void SetUp();
  virtual void TearDown();

  int rows;
  int cols;
  LaGenMatDouble* laMatrix;
  LapackppHostMatrix* hostMatrix;
};

LapackppHostMatrixTest::LapackppHostMatrixTest() :
    rows(5), cols(3), laMatrix(new LaGenMatDouble(rows, cols)), hostMatrix(new LapackppHostMatrix(laMatrix)) {

}

LapackppHostMatrixTest::~LapackppHostMatrixTest() {

}

void LapackppHostMatrixTest::SetUp() {
  laMatrix = new LaGenMatDouble(rows, cols);
  hostMatrix = new LapackppHostMatrix(laMatrix);
}

void LapackppHostMatrixTest::TearDown() {
  delete hostMatrix;
}

TEST_F(LapackppHostMatrixTest, Getters) {
  EXPECT_EQ(cols, hostMatrix->getNumberOfColumns());
  EXPECT_EQ(rows, hostMatrix->getNumberOfRows());
}

TEST_F(LapackppHostMatrixTest, GetLapackpp) {
  LaGenMatDouble& laMatrixGet = hostMatrix->getLapackpp();

  EXPECT_EQ(rows, laMatrixGet.rows());
  EXPECT_EQ(cols, laMatrixGet.cols());

  double a = 5;
  double b = 3.2;
  laMatrixGet(0, 0) = a;
  laMatrixGet(2, 1) = b;

  EXPECT_EQ(laMatrixGet(0, 0), (*hostMatrix)(0, 0));
  EXPECT_EQ(laMatrixGet(2, 1), (*hostMatrix)(2, 1));
  EXPECT_EQ(a, (*hostMatrix)(0, 0));
  EXPECT_EQ(b, (*hostMatrix)(2, 1));
}

TEST_F(LapackppHostMatrixTest, AccessOperator) {
  double a = 5;
  double b = 3.2;

  (*hostMatrix)(0, 1) = a;
  (*hostMatrix)(3, 2) = b;

  EXPECT_EQ(a, (*hostMatrix)(0, 1));
  EXPECT_EQ(b, (*hostMatrix)(3, 2));
}

TEST_F(LapackppHostMatrixTest, AccessOperatorColumn) {
  double a = 5;
  double b = 3.2;
  double c = 2.1;

  HostVector* colVector = (*hostMatrix)(1);
  (*hostMatrix)(0, 1) = a;
  (*hostMatrix)(3, 1) = b;

  EXPECT_EQ(a, (*colVector)(0));
  EXPECT_EQ(b, (*colVector)(3));

  (*colVector)(3) = c;
  EXPECT_EQ(c, (*hostMatrix)(3, 1));

  delete colVector;
}

TEST_F(LapackppHostMatrixTest, AccessOperatorColumnMemory) {
  double a = 5;
  double b = 3.2;
  double c = 2.1;

  HostVector* colVector = (*hostMatrix)(1);
  (*hostMatrix)(0, 1) = a;
  (*hostMatrix)(3, 1) = b;
  (*colVector)(2) = c;

  delete colVector;

  EXPECT_EQ(a, (*hostMatrix)(0, 1));
  EXPECT_EQ(b, (*hostMatrix)(3, 1));
  EXPECT_EQ(c, (*hostMatrix)(2, 1));
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */

