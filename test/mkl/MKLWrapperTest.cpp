#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>

#include <MKLWrapper.h>
#include <HostVector.h>
#include <HostMatrix.h>
#include <MKLException.h>
#include <DimensionMismatch.h>

#ifdef CPU
#include <lapackpp/gmd.h>
#include <lapackpp/lavd.h>
#include <LapackppHostMatrix.h>
#include <LapackppHostVector.h>
#else
#include <PinnedHostMatrix.h>
#include <PinnedHostVector.h>
#endif

namespace CuEira {
namespace CuEira_Test {

using namespace CuEira::Container;

/**
 * Testing MKLWrapper
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class MKLWrapperTest: public ::testing::Test {
protected:
  MKLWrapperTest();
  virtual ~MKLWrapperTest();
  virtual void SetUp();
  virtual void TearDown();

  MKLWrapper mklWrapper;
};

MKLWrapperTest::MKLWrapperTest() {

}

MKLWrapperTest::~MKLWrapperTest() {

}

void MKLWrapperTest::SetUp() {

}

void MKLWrapperTest::TearDown() {

}

TEST_F(MKLWrapperTest, CopyVector) {
  const int size = 5;
#ifdef CPU
  LapackppHostVector vectorFrom(new LaVectorDouble(size));
  LapackppHostVector vectorTo(new LaVectorDouble(size));
#else
  PinnedHostVector vectorFrom(size);
  PinnedHostVector vectorTo(size);
#endif

  for(int i = 0; i < size; ++i){
    vectorFrom(i) = i - size / 2 + 0.01;
  }

  mklWrapper.copyVector(vectorFrom, vectorTo);

  for(int i = 0; i < size; ++i){
    EXPECT_EQ(vectorFrom(i), vectorTo(i));
  }

}

TEST_F(MKLWrapperTest, SVD) {

}

TEST_F(MKLWrapperTest, matrixVectorMultiply) {
  const int numberOfRows = 3;
  const int numberOfColumns = 5;
#ifdef CPU
  LapackppHostVector vector(new LaVectorDouble(numberOfColumns));
  LapackppHostVector vectorRes(new LaVectorDouble(numberOfColumns));
  LapackppHostMatrix matrix1(new LaGenMatDouble(numberOfRows, numberOfColumns));
#else
  PinnedHostVector vector(numberOfColumns);
  PinnedHostVector vectorRes(numberOfColumns);
  PinnedHostMatrix matrix1(numberOfRows, numberOfColumns);
#endif

  vector(0) = 1;
  vector(1) = 2;
  vector(2) = 3;
  vector(3) = 4;
  vector(4) = 5;

  matrix1(0, 0) = 1;
  matrix1(0, 1) = 2;
  matrix1(0, 2) = 3;
  matrix1(0, 3) = 4;
  matrix1(0, 4) = 5;

  matrix1(1, 0) = 10;
  matrix1(1, 1) = 20;
  matrix1(1, 2) = 30;
  matrix1(1, 3) = 40;
  matrix1(1, 4) = 50;

  matrix1(2, 0) = 1.1;
  matrix1(2, 1) = 2.2;
  matrix1(2, 2) = 3.3;
  matrix1(2, 3) = 4.4;
  matrix1(2, 4) = 5.5;

  mklWrapper.matrixVectorMultiply(matrix1, vector, vectorRes, 1, 0);

  EXPECT_EQ(55, vectorRes(0));
  EXPECT_EQ(550, vectorRes(1));
  EXPECT_EQ(60.5, vectorRes(2));
}

TEST_F(MKLWrapperTest, matrixTransVectorMultiply) {
  const int numberOfRows = 3;
  const int numberOfColumns = 5;
#ifdef CPU
  LapackppHostVector vector(new LaVectorDouble(numberOfColumns));
  LapackppHostVector vectorRes(new LaVectorDouble(numberOfColumns));
  LapackppHostMatrix matrixT(new LaGenMatDouble(numberOfColumns, numberOfRows));
#else
  PinnedHostVector vector(numberOfColumns);
  PinnedHostVector vectorRes(numberOfColumns);
  PinnedHostMatrix matrixT(numberOfColumns, numberOfRows);
#endif

  vector(0) = 1;
  vector(1) = 2;
  vector(2) = 3;
  vector(3) = 4;
  vector(4) = 5;

  matrix1(0, 0) = 1;
  matrix1(1, 0) = 2;
  matrix1(2, 0) = 3;
  matrix1(3, 0) = 4;
  matrix1(4, 0) = 5;

  matrix1(0, 1) = 10;
  matrix1(1, 1) = 20;
  matrix1(2, 1) = 30;
  matrix1(3, 1) = 40;
  matrix1(4, 1) = 50;

  matrix1(0, 2) = 1.1;
  matrix1(1, 2) = 2.2;
  matrix1(2, 2) = 3.3;
  matrix1(3, 2) = 4.4;
  matrix1(4, 2) = 5.5;

  mklWrapper.matrixTransVectorMultiply(matrix1, vector, vectorRes, 1, 0);

  EXPECT_EQ(55, vectorRes(0));
  EXPECT_EQ(550, vectorRes(1));
  EXPECT_EQ(60.5, vectorRes(2));
}

TEST_F(MKLWrapperTest, matrixMatrixMultiply) {
  const int numberOfRows = 3;
  const int numberOfColumns = 5;
  const int numberOfRows2 = 5;
  const int numberOfColumns2 = 4;
#ifdef CPU
  LapackppHostMatrix matrix1(new LaGenMatDouble(numberOfRows, numberOfColumns));
  LapackppHostMatrix matrix2(new LaGenMatDouble(numberOfRows2, numberOfColumns2));
  LapackppHostMatrix matrixRes(new LaGenMatDouble(numberOfRows, numberOfColumns2));
#else
  PinnedHostMatrix matrix1(numberOfRows, numberOfColumns);
  PinnedHostMatrix matrix2(numberOfRows2, numberOfColumns2);
  PinnedHostMatrix matrixRes(numberOfRows, numberOfColumns2);
#endif

  matrix1(0, 0) = 1;
  matrix1(0, 1) = 2;
  matrix1(0, 2) = 3;
  matrix1(0, 3) = 4;
  matrix1(0, 4) = 5;

  matrix1(1, 0) = 10;
  matrix1(1, 1) = 20;
  matrix1(1, 2) = 30;
  matrix1(1, 3) = 40;
  matrix1(1, 4) = 50;

  matrix1(2, 0) = 1.1;
  matrix1(2, 1) = 2.2;
  matrix1(2, 2) = 3.3;
  matrix1(2, 3) = 4.4;
  matrix1(2, 4) = 5.5;

  for(int i = 0; i < numberOfRows2; ++i){
    matrix2(i, 0) = 6;
  }

  for(int i = 0; i < numberOfRows2; ++i){
    matrix2(i, 1) = 7;
  }

  for(int i = 0; i < numberOfRows2; ++i){
    matrix2(i, 2) = 8;
  }

  for(int i = 0; i < numberOfRows2; ++i){
    matrix2(i, 3) = 9;
  }

  mklWrapper.matrixMatrixMultiply(matrix1, matrix2, matrixRes, 1, 0);

  EXPECT_EQ(0, matrixRes(90, 1));
  EXPECT_EQ(0, matrixRes(105, 2));
  EXPECT_EQ(0, matrixRes(120, 3));
  EXPECT_EQ(0, matrixRes(135, 4));

  EXPECT_EQ(0, matrixRes(900, 1));
  EXPECT_EQ(0, matrixRes(1050, 2));
  EXPECT_EQ(0, matrixRes(1200, 3));
  EXPECT_EQ(0, matrixRes(1350, 4));

  EXPECT_EQ(0, matrixRes(99, 1));
  EXPECT_EQ(0, matrixRes(115.5, 2));
  EXPECT_EQ(0, matrixRes(132, 3));
  EXPECT_EQ(0, matrixRes(148.5, 4));
}

TEST_F(MKLWrapperTest, matrixTransMatrixMultiply) {
  const int numberOfRows = 5;
  const int numberOfColumns = 3;
  const int numberOfRows2 = 5;
  const int numberOfColumns2 = 4;
#ifdef CPU
  LapackppHostMatrix matrix1(new LaGenMatDouble(numberOfRows, numberOfColumns));
  LapackppHostMatrix matrix2(new LaGenMatDouble(numberOfRows2, numberOfColumns2));
  LapackppHostMatrix matrixRes(new LaGenMatDouble(numberOfColumns, numberOfColumns2));
#else
  PinnedHostMatrix matrix1(numberOfRows, numberOfColumns);
  PinnedHostMatrix matrix2(numberOfRows2, numberOfColumns2);
  PinnedHostMatrix matrixRes(numberOfColumns, numberOfColumns2);
#endif

  matrix1(0, 0) = 1;
  matrix1(1, 0) = 2;
  matrix1(2, 0) = 3;
  matrix1(3, 0) = 4;
  matrix1(4, 0) = 5;

  matrix1(0, 1) = 10;
  matrix1(1, 1) = 20;
  matrix1(2, 1) = 30;
  matrix1(3, 1) = 40;
  matrix1(4, 1) = 50;

  matrix1(0, 2) = 1.1;
  matrix1(1, 2) = 2.2;
  matrix1(2, 2) = 3.3;
  matrix1(3, 2) = 4.4;
  matrix1(4, 2) = 5.5;

  for(int i = 0; i < numberOfRows2; ++i){
    matrix2(i, 0) = 6;
  }

  for(int i = 0; i < numberOfRows2; ++i){
    matrix2(i, 1) = 7;
  }

  for(int i = 0; i < numberOfRows2; ++i){
    matrix2(i, 2) = 8;
  }

  for(int i = 0; i < numberOfRows2; ++i){
    matrix2(i, 3) = 9;
  }

  mklWrapper.matrixTransMatrixMultiply(matrix1, matrix2, matrixRes, 1, 0);

  EXPECT_EQ(0, matrixRes(90, 1));
  EXPECT_EQ(0, matrixRes(105, 2));
  EXPECT_EQ(0, matrixRes(120, 3));
  EXPECT_EQ(0, matrixRes(135, 4));

  EXPECT_EQ(0, matrixRes(900, 1));
  EXPECT_EQ(0, matrixRes(1050, 2));
  EXPECT_EQ(0, matrixRes(1200, 3));
  EXPECT_EQ(0, matrixRes(1350, 4));

  EXPECT_EQ(0, matrixRes(99, 1));
  EXPECT_EQ(0, matrixRes(115.5, 2));
  EXPECT_EQ(0, matrixRes(132, 3));
  EXPECT_EQ(0, matrixRes(148.5, 4));
}

TEST_F(MKLWrapperTest, differenceElememtWise) {
  const int size = 5;
#ifdef CPU
  LapackppHostVector vector1(new LaVectorDouble(size));
  LapackppHostVector vector2(new LaVectorDouble(size));
  LapackppHostVector vectorRes(new LaVectorDouble(size));
#else
  PinnedHostVector vector1(size);
  PinnedHostVector vector2(size);
  PinnedHostVector vectorRes(size);
#endif

  for(int i = 0; i < size; ++i){
    vector1(i) = i - size / 2 + 0.01;
    vector2(i) = i - 1;

    vectorRes(i) = vector2(i) - vector1(i);
  }

  mklWrapper.differenceElememtWise(vector1, vector2);

  for(int i = 0; i < size; ++i){
    EXPECT_EQ(vectorRes(i), vector2(i));
  }
}

TEST_F(MKLWrapperTest, absoluteSum) {
  const int size = 5;
#ifdef CPU
  LapackppHostVector vector(new LaVectorDouble(size));
#else
  PinnedHostVector vector(size);
#endif

  for(int i = 0; i < size; ++i){
    vector(i) = i - size / 2 + 0.01;
  }

  PRECISION* res = new PRECISION(0);
  mklWrapper.absoluteSum(vector, res);

  PRECISION sum = 0;
  for(int i = 0; i < size; ++i){
    sum += abs(vector(i));
  }

  EXPECT_EQ(sum, *res);
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */
