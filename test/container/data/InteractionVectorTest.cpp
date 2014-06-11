#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include <Recode.h>
#include <InteractionVector.h>
#include <EnvironmentVector.h>
#include <SNPVector.h>
#include <HostVector.h>
#include <SNPVectorMock.h>
#include <EnvironmentVectorMock.h>
#include <ConstructorHelpers.h>

#ifdef CPU
#include <lapackpp/lavd.h>
#include <LapackppHostVector.h>
#else
#include <PinnedHostVector.h>
#endif

using testing::Return;
using testing::_;
using testing::ReturnRef;

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

  const int numberOfIndividuals = 6;
  CuEira_Test::ConstructorHelpers constructorHelpers;
  EnvironmentVectorMock* environmentVectorMock;
  SNPVectorMock* snpVectorMock;
  HostVector* envData;
  HostVector* snpData;
  std::vector<PRECISION> interact;
};

InteractionVectorTest::InteractionVectorTest() :
    environmentVectorMock(constructorHelpers.constructEnvironmentVectorMock()), snpVectorMock(
        constructorHelpers.constructSNPVectorMock()), interact(numberOfIndividuals),
#ifdef CPU
        envData(new LapackppHostVector(new LaVectorDouble(numberOfIndividuals))),
        snpData(new LapackppHostVector(new LaVectorDouble(numberOfIndividuals)))
#else
        envData(new PinnedHostVector(numberOfIndividuals)), snpData(new PinnedHostVector(numberOfIndividuals))
#endif
{
  for(int i = 0; i < numberOfIndividuals; ++i){
    (*envData)(i) = 0;
    (*snpData)(i) = 0;

    interact[i] = (*envData)(i) * (*snpData)(i);
  }
}

InteractionVectorTest::~InteractionVectorTest() {
  delete envData;
  delete snpData;
  delete environmentVectorMock;
  delete snpVectorMock;
}

void InteractionVectorTest::SetUp() {

}

void InteractionVectorTest::TearDown() {

}

TEST_F(InteractionVectorTest, ConstructAndGet) {
  EXPECT_CALL(*environmentVectorMock, getRecodedData()).Times(1).WillRepeatedly(ReturnRef(*envData));
  EXPECT_CALL(*snpVectorMock, getRecodedData()).Times(1).WillRepeatedly(ReturnRef(*snpData));
  EXPECT_CALL(*snpVectorMock, getNumberOfIndividualsToInclude()).Times(1).WillRepeatedly(Return(numberOfIndividuals));

  InteractionVector interactionVector(*environmentVectorMock, *snpVectorMock);

  ASSERT_EQ(numberOfIndividuals, interactionVector.getNumberOfIndividualsToInclude());

  const Container::HostVector& recodeData = interactionVector.getRecodedData();
  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ(interact[i], recodeData(i));
  }
}

} /* namespace Container */
} /* namespace CuEira */
