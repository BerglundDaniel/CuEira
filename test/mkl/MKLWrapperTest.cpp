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
  LapackppHostVector vectorFromWrongSize(new LaVectorDouble(size - 1));
#else
  PinnedHostVector vectorFrom(size);
  PinnedHostVector vectorTo(size);
  PinnedHostVector vectorFromWrongSize(size - 1);
#endif

  for(int i = 0; i < size; ++i){
    vectorFrom(i) = i - size / 2 + 0.01;
  }

  mklWrapper.copyVector(vectorFrom, vectorTo);

  for(int i = 0; i < size; ++i){
    EXPECT_EQ(vectorFrom(i), vectorTo(i));
  }

  ASSERT_THROW(mklWrapper.copyVector(vectorFromWrongSize, vectorTo), DimensionMismatch);
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */
