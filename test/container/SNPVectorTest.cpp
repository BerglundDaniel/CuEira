#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include <SNP.h>
#include <Id.h>
#include <Recode.h>
#include <SNPVector.h>
#include <HostVector.h>
#include <InvalidState.h>
#include <GeneticModel.h>
#include <RiskAllele.h>

namespace CuEira {
namespace Container {

/**
 * Test for testing SNPVector
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class SNPVectorTest: public ::testing::Test {
protected:
  SNPVectorTest();
  virtual ~SNPVectorTest();
  virtual void SetUp();
  virtual void TearDown();
};

SNPVectorTest::SNPVectorTest() {

}

SNPVectorTest::~SNPVectorTest() {

}

void SNPVectorTest::SetUp() {

}

void SNPVectorTest::TearDown() {

}

TEST_F(SNPVectorTest, Getters) {

}

} /* namespace Container */
} /* namespace CuEira */

