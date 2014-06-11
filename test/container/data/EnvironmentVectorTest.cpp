#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include <Recode.h>
#include <EnvironmentVector.h>
#include <SNPVector.h>
#include <HostVector.h>
#include <VariableType.h>
#include <StatisticModel.h>

#ifdef CPU
#include <lapackpp/lavd.h>
#include <LapackppHostVector.h>
#else
#include <PinnedHostVector.h>
#endif

namespace CuEira {
namespace Container {

/**
 * Test for testing EnvironmentVector
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class EnvironmentVectorTest: public ::testing::Test {
protected:
  EnvironmentVectorTest();
  virtual ~EnvironmentVectorTest();
  virtual void SetUp();
  virtual void TearDown();

  const int numberOfIndividuals = 6;

};

EnvironmentVectorTest::EnvironmentVectorTest() {

}

EnvironmentVectorTest::~EnvironmentVectorTest() {

}

void EnvironmentVectorTest::SetUp() {

}

void EnvironmentVectorTest::TearDown() {

}

TEST_F(EnvironmentVectorTest, ConstructAndGet) {

}

} /* namespace Container */
} /* namespace CuEira */

