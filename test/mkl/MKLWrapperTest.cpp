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

using ::testing::Ge;
using ::testing::Le;

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
  double e;
};

MKLWrapperTest::MKLWrapperTest() :
    e(1e-5) {

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
  const int size = 3;
  double l;
  double h;
  double x;
#ifdef CPU
  LapackppHostVector sigma(new LaVectorDouble(size));
  LapackppHostMatrix matrix(new LaGenMatDouble(size, size));
  LapackppHostMatrix uSVD(new LaGenMatDouble(size, size));
  LapackppHostMatrix vtSVD(new LaGenMatDouble(size, size));
#else
  PinnedHostVector sigma(size);
  PinnedHostMatrix matrix(size, size);
  PinnedHostMatrix uSVD(size, size);
  PinnedHostMatrix vtSVD(size, size);
#endif

  matrix(0, 0) = 1;
  matrix(1, 0) = 2;
  matrix(2, 0) = 3;

  matrix(0, 1) = 4;
  matrix(1, 1) = 5;
  matrix(2, 1) = 6;

  matrix(0, 2) = 7;
  matrix(1, 2) = 8;
  matrix(2, 2) = 9;

  mklWrapper.svd(matrix, uSVD, sigma, vtSVD);

  ///
  ///sigma
  ///
  x = 16.84810;
  l = x - e;
  h = x + e;
  EXPECT_THAT(sigma(0), Ge(l));
  EXPECT_THAT(sigma(0), Le(h));

  x = 1.068370;
  l = x - e;
  h = x + e;
  EXPECT_THAT(sigma(1), Ge(l));
  EXPECT_THAT(sigma(1), Le(h));

  //x = 5.543107e-16;
  //Checking that it's around 0.
  l = -1e-4;
  h = 1e-4;
  EXPECT_THAT(sigma(2), Ge(l));
  EXPECT_THAT(sigma(2), Le(h));

  /*
   $u
   [,1]        [,2]       [,3]
   [1,] -0.4796712  0.77669099  0.4082483
   [2,] -0.5723678  0.07568647 -0.8164966
   [3,] -0.6650644 -0.62531805  0.4082483



   $vt
   [,1]       [,2]       [,3]
   [1,] -0.2148372 -0.5205874 -0.8263375
   [2,] -0.8872307 -0.2496440  0.3879428
   [3,]  0.4082483 -0.8164966  0.4082483

   */

  ///
  ///uSVD
  ///
  x = -0.4796712;
  l = x - e;
  h = x + e;
  EXPECT_THAT(uSVD(0, 0), Ge(l));
  EXPECT_THAT(uSVD(0, 0), Le(h));

  x = 0.77669099;
  l = x - e;
  h = x + e;
  EXPECT_THAT(uSVD(0, 1), Ge(l));
  EXPECT_THAT(uSVD(0, 1), Le(h));

  x = 0.4082483;
  l = x - e;
  h = x + e;
  EXPECT_THAT(uSVD(0, 2), Ge(l));
  EXPECT_THAT(uSVD(0, 2), Le(h));

  x = -0.5723678;
  l = x - e;
  h = x + e;
  EXPECT_THAT(uSVD(1, 0), Ge(l));
  EXPECT_THAT(uSVD(1, 0), Le(h));

  x = 0.07568647;
  l = x - e;
  h = x + e;
  EXPECT_THAT(uSVD(1, 1), Ge(l));
  EXPECT_THAT(uSVD(1, 1), Le(h));

  x = -0.8164966;
  l = x - e;
  h = x + e;
  EXPECT_THAT(uSVD(1, 2), Ge(l));
  EXPECT_THAT(uSVD(1, 2), Le(h));

  x = -0.6650644;
  l = x - e;
  h = x + e;
  EXPECT_THAT(uSVD(2, 0), Ge(l));
  EXPECT_THAT(uSVD(2, 0), Le(h));

  x = -0.62531805;
  l = x - e;
  h = x + e;
  EXPECT_THAT(uSVD(2, 1), Ge(l));
  EXPECT_THAT(uSVD(2, 1), Le(h));

  x = 0.4082483;
  l = x - e;
  h = x + e;
  EXPECT_THAT(uSVD(2, 2), Ge(l));
  EXPECT_THAT(uSVD(2, 2), Le(h));

  ///
  ///vtSVD
  ///
  x = -0.2148372;
  l = x - e;
  h = x + e;
  EXPECT_THAT(vtSVD(0, 0), Ge(l));
  EXPECT_THAT(vtSVD(0, 0), Le(h));

  x = -0.5205874;
  l = x - e;
  h = x + e;
  EXPECT_THAT(vtSVD(0, 1), Ge(l));
  EXPECT_THAT(vtSVD(0, 1), Le(h));

  x = -0.8263375;
  l = x - e;
  h = x + e;
  EXPECT_THAT(vtSVD(0, 2), Ge(l));
  EXPECT_THAT(vtSVD(0, 2), Le(h));

  x = -0.8872307;
  l = x - e;
  h = x + e;
  EXPECT_THAT(vtSVD(1, 0), Ge(l));
  EXPECT_THAT(vtSVD(1, 0), Le(h));

  x = -0.2496440;
  l = x - e;
  h = x + e;
  EXPECT_THAT(vtSVD(1, 1), Ge(l));
  EXPECT_THAT(vtSVD(1, 1), Le(h));

  x = 0.3879428;
  l = x - e;
  h = x + e;
  EXPECT_THAT(vtSVD(1, 2), Ge(l));
  EXPECT_THAT(vtSVD(1, 2), Le(h));

  x = 0.4082483;
  l = x - e;
  h = x + e;
  EXPECT_THAT(vtSVD(2, 0), Ge(l));
  EXPECT_THAT(vtSVD(2, 0), Le(h));

  x = -0.8164966;
  l = x - e;
  h = x + e;
  EXPECT_THAT(vtSVD(2, 1), Ge(l));
  EXPECT_THAT(vtSVD(2, 1), Le(h));

  x = 0.4082483;
  l = x - e;
  h = x + e;
  EXPECT_THAT(vtSVD(2, 2), Ge(l));
  EXPECT_THAT(vtSVD(2, 2), Le(h));
}

TEST_F(MKLWrapperTest, matrixVectorMultiply) {
  const int numberOfRows = 3;
  const int numberOfColumns = 5;
#ifdef CPU
  LapackppHostVector vector(new LaVectorDouble(numberOfColumns));
  LapackppHostVector vectorRes(new LaVectorDouble(numberOfRows));
  LapackppHostMatrix matrix1(new LaGenMatDouble(numberOfRows, numberOfColumns));
#else
  PinnedHostVector vector(numberOfColumns);
  PinnedHostVector vectorRes(numberOfRows);
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

  vectorRes(0) = 0;
  vectorRes(1) = 0;
  vectorRes(2) = 0;

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
  LapackppHostVector vectorRes(new LaVectorDouble(numberOfRows));
  LapackppHostMatrix matrixT(new LaGenMatDouble(numberOfColumns, numberOfRows));
#else
  PinnedHostVector vector(numberOfColumns);
  PinnedHostVector vectorRes(numberOfRows);
  PinnedHostMatrix matrixT(numberOfColumns, numberOfRows);
#endif

  vector(0) = 1;
  vector(1) = 2;
  vector(2) = 3;
  vector(3) = 4;
  vector(4) = 5;

  matrixT(0, 0) = 1;
  matrixT(1, 0) = 2;
  matrixT(2, 0) = 3;
  matrixT(3, 0) = 4;
  matrixT(4, 0) = 5;

  matrixT(0, 1) = 10;
  matrixT(1, 1) = 20;
  matrixT(2, 1) = 30;
  matrixT(3, 1) = 40;
  matrixT(4, 1) = 50;

  matrixT(0, 2) = 1.1;
  matrixT(1, 2) = 2.2;
  matrixT(2, 2) = 3.3;
  matrixT(3, 2) = 4.4;
  matrixT(4, 2) = 5.5;

  vectorRes(0) = 0;
  vectorRes(1) = 0;
  vectorRes(2) = 0;

  mklWrapper.matrixTransVectorMultiply(matrixT, vector, vectorRes, 1, 0);

  EXPECT_EQ(55, vectorRes(0));
  EXPECT_EQ(550, vectorRes(1));
  EXPECT_EQ(60.5, vectorRes(2));
}

TEST_F(MKLWrapperTest, matrixMatrixMultiply) {
  const int numberOfRows = 3;
  const int numberOfColumns = 5;
  const int numberOfRows2 = numberOfColumns;
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

  EXPECT_EQ(90, matrixRes(0, 0));
  EXPECT_EQ(105, matrixRes(0, 1));
  EXPECT_EQ(120, matrixRes(0, 2));
  EXPECT_EQ(135, matrixRes(0, 3));

  EXPECT_EQ(900, matrixRes(1, 0));
  EXPECT_EQ(1050, matrixRes(1, 1));
  EXPECT_EQ(1200, matrixRes(1, 2));
  EXPECT_EQ(1350, matrixRes(1, 3));

  EXPECT_EQ(99, matrixRes(2, 0));
  EXPECT_EQ(115.5, matrixRes(2, 1));
  EXPECT_EQ(132, matrixRes(2, 2));
  EXPECT_EQ(148.5, matrixRes(2, 3));
}

TEST_F(MKLWrapperTest, matrixTransMatrixMultiply) {
  const int numberOfRows = 5;
  const int numberOfColumns = 3;
  const int numberOfRows2 = 5;
  const int numberOfColumns2 = 4;
#ifdef CPU
  LapackppHostMatrix matrixT(new LaGenMatDouble(numberOfRows, numberOfColumns));
  LapackppHostMatrix matrix2(new LaGenMatDouble(numberOfRows2, numberOfColumns2));
  LapackppHostMatrix matrixRes(new LaGenMatDouble(numberOfColumns, numberOfColumns2));
#else
  PinnedHostMatrix matrixT(numberOfRows, numberOfColumns);
  PinnedHostMatrix matrix2(numberOfRows2, numberOfColumns2);
  PinnedHostMatrix matrixRes(numberOfColumns, numberOfColumns2);
#endif

  matrixT(0, 0) = 1;
  matrixT(1, 0) = 2;
  matrixT(2, 0) = 3;
  matrixT(3, 0) = 4;
  matrixT(4, 0) = 5;

  matrixT(0, 1) = 10;
  matrixT(1, 1) = 20;
  matrixT(2, 1) = 30;
  matrixT(3, 1) = 40;
  matrixT(4, 1) = 50;

  matrixT(0, 2) = 1.1;
  matrixT(1, 2) = 2.2;
  matrixT(2, 2) = 3.3;
  matrixT(3, 2) = 4.4;
  matrixT(4, 2) = 5.5;

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

  mklWrapper.matrixTransMatrixMultiply(matrixT, matrix2, matrixRes, 1, 0);

  EXPECT_EQ(90, matrixRes(0, 0));
  EXPECT_EQ(105, matrixRes(0, 1));
  EXPECT_EQ(120, matrixRes(0, 2));
  EXPECT_EQ(135, matrixRes(0, 3));

  EXPECT_EQ(900, matrixRes(1, 0));
  EXPECT_EQ(1050, matrixRes(1, 1));
  EXPECT_EQ(1200, matrixRes(1, 2));
  EXPECT_EQ(1350, matrixRes(1, 3));

  EXPECT_EQ(99, matrixRes(2, 0));
  EXPECT_EQ(115.5, matrixRes(2, 1));
  EXPECT_EQ(132, matrixRes(2, 2));
  EXPECT_EQ(148.5, matrixRes(2, 3));
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

TEST_F(MKLWrapperTest, absoluteSumInt) {
  const int size = 5;
#ifdef CPU
  LapackppHostVector vector(new LaVectorDouble(size));
#else
  PinnedHostVector vector(size);
#endif

  for(int i = 0; i < size; ++i){
    vector(i) = i - 2;
  }

  PRECISION* res = new PRECISION(0);
  mklWrapper.absoluteSum(vector, res);

  PRECISION sum = 0;
  for(int i = 0; i < size; ++i){
    sum += abs(vector(i));
  }

  EXPECT_EQ(sum, *res);
}

TEST_F(MKLWrapperTest, absoluteSumFloat) {
  const int size = 5;
#ifdef CPU
  LapackppHostVector vector(new LaVectorDouble(size));
#else
  PinnedHostVector vector(size);
#endif

  for(int i = 0; i < size; ++i){
    vector(i) = i - size / 2;
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
