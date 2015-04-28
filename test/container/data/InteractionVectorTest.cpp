#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include <InteractionVector.h>
#include <RegularHostVector.h>
#include <InvalidState.h>
#include <InvalidArgument.h>

namespace CuEira {
namespace Container {

/**
 * Test for testing InteractionVector
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class InteractionVectorTest: public ::testing::Test {
protected:
  InteractionVectorTest();
  virtual ~InteractionVectorTest();
  virtual void SetUp();
  virtual void TearDown();

};

InteractionVectorTest::InteractionVectorTest() {

}

InteractionVectorTest::~InteractionVectorTest() {

}

void InteractionVectorTest::SetUp() {

}

void InteractionVectorTest::TearDown() {

}

#ifdef DEBUG
TEST_F(InteractionVectorTest, Exception){
  const int numberOfIndvidualsTotal = 10;
  InteractionVector<RegularHostVector> interactionVector(numberOfIndvidualsTotal);

  EXPECT_THROW(interactionVector.getInteractionData(), InvalidState);
  EXPECT_THROW(interactionVector.getNumberOfIndividualsToInclude(), InvalidState);
}
#endif

TEST_F(InteractionVectorTest, UpdateAndGet) {
  const int numberOfIndvidualsTotal = 10;
  InteractionVector<RegularHostVector> interactionVector(numberOfIndvidualsTotal);

  const int size1 = 4;
  const int size2 = 8;

  interactionVector.updateSize(size1);
  EXPECT_EQ(size1, interactionVector.getNumberOfIndividualsToInclude());
  const RegularHostVector& vector1 = interactionVector.getInteractionData();
  EXPECT_EQ(size1, vector1.getNumberOfRows());

  interactionVector.updateSize(size2);
  EXPECT_EQ(size2, interactionVector.getNumberOfIndividualsToInclude());
  const RegularHostVector& vector2 = interactionVector.getInteractionData();
  EXPECT_EQ(size2, vector2.getNumberOfRows());

  EXPECT_THROW(interactionVector.updateSize(numberOfIndvidualsTotal + 1), InvalidState);
}

} /* namespace Container */
} /* namespace CuEira */

