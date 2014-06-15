#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include <Recode.h>
#include <EnvironmentVector.h>
#include <EnvironmentFactor.h>
#include <HostVector.h>
#include <VariableType.h>
#include <StatisticModel.h>
#include <Id.h>
#include <ConstructorHelpers.h>
#include <EnvironmentFactorHandlerMock.h>

#ifdef CPU
#include <lapackpp/lavd.h>
#include <LapackppHostVector.h>
#else
#include <PinnedHostVector.h>
#endif

using testing::Return;
using testing::AtLeast;
using testing::ReturnRef;

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

  const int numberOfIndividuals;
  CuEira_Test::ConstructorHelpers constructorHelpers;
  EnvironmentFactorHandlerMock* environmentFactorHandlerMock;
  EnvironmentFactor startFactor;
  EnvironmentFactor binaryFactor;
  Container::HostVector* orgData;
  Container::HostVector* binaryData;
};

EnvironmentVectorTest::EnvironmentVectorTest() :
    numberOfIndividuals(6), environmentFactorHandlerMock(constructorHelpers.constructEnvironmentFactorHandlerMock()), startFactor(
        Id("startFactor")), binaryFactor(Id("binaryFactor")),
#ifdef CPU
        orgData(new LapackppHostVector(new LaVectorDouble(numberOfIndividuals))),
        binaryData(new LapackppHostVector(new LaVectorDouble(numberOfIndividuals)))
#else
        orgData(new PinnedHostVector(numberOfIndividuals)), binaryData(new PinnedHostVector(numberOfIndividuals))
#endif
{

  startFactor.setVariableType(OTHER);
  binaryFactor.setVariableType(BINARY);

  for(int i = 0; i < numberOfIndividuals; ++i){
    (*orgData)(i) = i;

    if(i < 5 && i > 2){
      (*binaryData)(i) = 1;
    }else{
      (*binaryData)(i) = 0;
    }
  }
}

EnvironmentVectorTest::~EnvironmentVectorTest() {
  delete environmentFactorHandlerMock;
  delete orgData;
  delete binaryData;
}

void EnvironmentVectorTest::SetUp() {
  EXPECT_CALL(*environmentFactorHandlerMock, getData(startFactor)).Times(1).WillRepeatedly(ReturnRef(*orgData));
  EXPECT_CALL(*environmentFactorHandlerMock, getData(binaryFactor)).Times(AtLeast(0)).WillRepeatedly(
      ReturnRef(*binaryData));
}

void EnvironmentVectorTest::TearDown() {

}

TEST_F(EnvironmentVectorTest, ConstructAndGet) {
  EnvironmentVector environmentVector(*environmentFactorHandlerMock, startFactor);

  ASSERT_EQ(ALL_RISK, environmentVector.currentRecode);
  ASSERT_EQ(startFactor, environmentVector.getCurrentEnvironmentFactor());

  const Container::HostVector& recodedData = environmentVector.getRecodedData();
  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());
  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*orgData)(i), recodedData(i));
  }
}

TEST_F(EnvironmentVectorTest, Switch) {
  EnvironmentVector environmentVector(*environmentFactorHandlerMock, startFactor);

  environmentVector.switchEnvironmentFactor(binaryFactor);
  ASSERT_EQ(binaryFactor, environmentVector.getCurrentEnvironmentFactor());

  const Container::HostVector& recodedData = environmentVector.getRecodedData();
  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());
  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*binaryData)(i), recodedData(i));
  }
}

TEST_F(EnvironmentVectorTest, RecodeNonBinary) {
  EnvironmentVector environmentVector(*environmentFactorHandlerMock, startFactor);
  ASSERT_EQ(ALL_RISK, environmentVector.currentRecode);

  environmentVector.recode(ALL_RISK);
  ASSERT_EQ(ALL_RISK, environmentVector.currentRecode);

  const HostVector* recodedData = &environmentVector.getRecodedData();
  ASSERT_EQ(numberOfIndividuals, recodedData->getNumberOfRows());
  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*orgData)(i), (*recodedData)(i));
  }

  environmentVector.recode(ENVIRONMENT_PROTECT);
  ASSERT_EQ(ENVIRONMENT_PROTECT, environmentVector.currentRecode);

  recodedData = &environmentVector.getRecodedData();
  ASSERT_EQ(numberOfIndividuals, recodedData->getNumberOfRows());
  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*orgData)(i) * -1, (*recodedData)(i));
  }

  environmentVector.recode(ENVIRONMENT_PROTECT);
  ASSERT_EQ(ENVIRONMENT_PROTECT, environmentVector.currentRecode);

  recodedData = &environmentVector.getRecodedData();
  ASSERT_EQ(numberOfIndividuals, recodedData->getNumberOfRows());
  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*orgData)(i) * -1, (*recodedData)(i));
  }

  environmentVector.recode(ALL_RISK);
  ASSERT_EQ(ALL_RISK, environmentVector.currentRecode);

  recodedData = &environmentVector.getRecodedData();
  ASSERT_EQ(numberOfIndividuals, recodedData->getNumberOfRows());
  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*orgData)(i), (*recodedData)(i));
  }

  environmentVector.recode(INTERACTION_PROTECT);
  ASSERT_EQ(INTERACTION_PROTECT, environmentVector.currentRecode);

  recodedData = &environmentVector.getRecodedData();
  ASSERT_EQ(numberOfIndividuals, recodedData->getNumberOfRows());
  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*orgData)(i) * -1, (*recodedData)(i));
  }
}

TEST_F(EnvironmentVectorTest, RecodeBinary) {
  Container::HostVector* binaryDataInvert;
#ifdef CPU
  binaryDataInvert=new LapackppHostVector(new LaVectorDouble(numberOfIndividuals));
#else
  binaryDataInvert = new PinnedHostVector(numberOfIndividuals);
#endif
  for(int i = 0; i < numberOfIndividuals; ++i){
    if((*binaryData)(i) == 0){
      (*binaryDataInvert)(i) = 1;
    }else{
      (*binaryDataInvert)(i) = 0;
    }
  }

  EnvironmentVector environmentVector(*environmentFactorHandlerMock, startFactor);
  environmentVector.switchEnvironmentFactor(binaryFactor);
  ASSERT_EQ(ALL_RISK, environmentVector.currentRecode);

  environmentVector.recode(ALL_RISK);
  ASSERT_EQ(ALL_RISK, environmentVector.currentRecode);

  const HostVector* recodedData = &environmentVector.getRecodedData();
  ASSERT_EQ(numberOfIndividuals, recodedData->getNumberOfRows());
  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*binaryData)(i), (*recodedData)(i));
  }

  environmentVector.recode(ENVIRONMENT_PROTECT);
  ASSERT_EQ(ENVIRONMENT_PROTECT, environmentVector.currentRecode);

  recodedData = &environmentVector.getRecodedData();
  ASSERT_EQ(numberOfIndividuals, recodedData->getNumberOfRows());
  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*binaryDataInvert)(i), (*recodedData)(i));
  }

  environmentVector.recode(ENVIRONMENT_PROTECT);
  ASSERT_EQ(ENVIRONMENT_PROTECT, environmentVector.currentRecode);

  recodedData = &environmentVector.getRecodedData();
  ASSERT_EQ(numberOfIndividuals, recodedData->getNumberOfRows());
  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*binaryDataInvert)(i), (*recodedData)(i));
  }

  environmentVector.recode(ALL_RISK);
  ASSERT_EQ(ALL_RISK, environmentVector.currentRecode);

  recodedData = &environmentVector.getRecodedData();
  ASSERT_EQ(numberOfIndividuals, recodedData->getNumberOfRows());
  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*binaryData)(i), (*recodedData)(i));
  }

  environmentVector.recode(INTERACTION_PROTECT);
  ASSERT_EQ(INTERACTION_PROTECT, environmentVector.currentRecode);

  recodedData = &environmentVector.getRecodedData();
  ASSERT_EQ(numberOfIndividuals, recodedData->getNumberOfRows());
  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*binaryDataInvert)(i), (*recodedData)(i));
  }

  delete binaryDataInvert;
}

TEST_F(EnvironmentVectorTest, StatisticModel) {
  EnvironmentVector environmentVector(*environmentFactorHandlerMock, startFactor);
  Container::HostVector* interactionVector;
#ifdef CPU
  interactionVector=new LapackppHostVector(new LaVectorDouble(numberOfIndividuals));
#else
  interactionVector = new PinnedHostVector(numberOfIndividuals);
#endif

  for(int i = 0; i < numberOfIndividuals; ++i){
    if(i < 3){
      (*interactionVector)(i) = 0;
    }else{
      (*interactionVector)(i) = 1;
    }
  }

  environmentVector.applyStatisticModel(MULTIPLICATIVE, *interactionVector);
  const Container::HostVector* recodedData = &environmentVector.getRecodedData();
  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*orgData)(i), (*recodedData)(i));
  }

  environmentVector.switchEnvironmentFactor(binaryFactor);
  environmentVector.applyStatisticModel(ADDITIVE, *interactionVector);
  recodedData = &environmentVector.getRecodedData();
  for(int i = 0; i < numberOfIndividuals; ++i){
    if((*interactionVector)(i) != 0){
      EXPECT_EQ(0, (*recodedData)(i));
    }else{
      EXPECT_EQ((*binaryData)(i), (*recodedData)(i));
    }
  }

  delete interactionVector;
}

TEST_F(EnvironmentVectorTest, RecodeDifferentOrder) {
  EnvironmentVector environmentVector(*environmentFactorHandlerMock, startFactor);

  ASSERT_EQ(ALL_RISK, environmentVector.currentRecode);
  ASSERT_EQ(startFactor, environmentVector.getCurrentEnvironmentFactor());

  const Container::HostVector* recodedData = &environmentVector.getRecodedData();
  ASSERT_EQ(numberOfIndividuals, recodedData->getNumberOfRows());
  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*orgData)(i), (*recodedData)(i));
  }

  environmentVector.recode(ENVIRONMENT_PROTECT);
  ASSERT_EQ(ENVIRONMENT_PROTECT, environmentVector.currentRecode);

  recodedData = &environmentVector.getRecodedData();
  ASSERT_EQ(numberOfIndividuals, recodedData->getNumberOfRows());
  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*orgData)(i) * -1, (*recodedData)(i));
  }
}

} /* namespace Container */
} /* namespace CuEira */

