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

}

TEST_F(MKLWrapperTest, matrixTransVectorMultiply) {

}

TEST_F(MKLWrapperTest, matrixTransMatrixMultiply) {

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
