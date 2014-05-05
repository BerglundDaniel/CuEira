#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>

#include <LapackppHostVector.h>
#include <HostVector.h>

namespace CuEira {
namespace CuEira_Test {

using namespace CuEira::Container;

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LapackppHostVectorTest: public ::testing::Test {
protected:
  LapackppHostVectorTest();
  virtual ~LapackppHostVectorTest();
  virtual void SetUp();
  virtual void TearDown();
};

LapackppHostVectorTest::LapackppHostVectorTest() {

}

LapackppHostVectorTest::~LapackppHostVectorTest() {

}

void LapackppHostVectorTest::SetUp() {

}

void LapackppHostVectorTest::TearDown() {

}

TEST_F(LapackppHostVectorTest, Construct){
  const int size=5;
  LapackppHostVector hostVector(size);
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */

