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
#include <InvalidState.h>

#ifdef CPU
#include <lapackpp/lavd.h>
#include <LapackppHostVector.h>
#else
#include <PinnedHostVector.h>
#endif

using testing::Return;
using testing::AtLeast;
using testing::ReturnRef;
using testing::Eq;
using testing::ByRef;

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
        Id("startFactor")), binaryFactor(Id("binaryFactor")), orgData(nullptr), binaryData(nullptr)

{
  startFactor.setVariableType(OTHER);
  binaryFactor.setVariableType(BINARY);
}

EnvironmentVectorTest::~EnvironmentVectorTest() {
  delete environmentFactorHandlerMock;
}

void EnvironmentVectorTest::SetUp() {
  EXPECT_CALL(*environmentFactorHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillRepeatedly(
      Return(numberOfIndividuals));

#ifdef CPU
  orgData = new LapackppHostVector(new LaVectorDouble(numberOfIndividuals));
  binaryData = new LapackppHostVector(new LaVectorDouble(numberOfIndividuals));
#else
  orgData = new PinnedHostVector(numberOfIndividuals);
  binaryData = new PinnedHostVector(numberOfIndividuals);
#endif

  for(int i = 0; i < numberOfIndividuals; ++i){
    (*orgData)(i) = i;

    if(i < 5 && i > 2){
      (*binaryData)(i) = 1;
    }else{
      (*binaryData)(i) = 0;
    }
  }
}

void EnvironmentVectorTest::TearDown() {
  delete orgData;
  delete binaryData;
}

#ifdef DEBUG
TEST_F(EnvironmentVectorTest, ConstructAndGetException){
  Container::HostVector* interactionVector;
#ifdef CPU
  interactionVector=new LapackppHostVector(new LaVectorDouble(numberOfIndividuals));
#else
  interactionVector = new PinnedHostVector(numberOfIndividuals);
#endif

  EnvironmentVector environmentVector(*environmentFactorHandlerMock);

  EXPECT_THROW(environmentVector.getCurrentEnvironmentFactor(), InvalidState);
  EXPECT_THROW(environmentVector.applyStatisticModel(MULTIPLICATIVE, *interactionVector), InvalidState);
  EXPECT_THROW(environmentVector.applyStatisticModel(ADDITIVE, *interactionVector), InvalidState);
  EXPECT_THROW(environmentVector.getRecodedData(), InvalidState);
  EXPECT_THROW(environmentVector.recode(ALL_RISK), InvalidState);
  EXPECT_THROW(environmentVector.recode(ENVIRONMENT_PROTECT), InvalidState);
  EXPECT_THROW(environmentVector.recode(SNP_PROTECT), InvalidState);
  EXPECT_THROW(environmentVector.recode(INTERACTION_PROTECT), InvalidState);

  delete interactionVector;
}
#endif

TEST_F(EnvironmentVectorTest, ConstructAndGet) {
  EnvironmentVector environmentVector(*environmentFactorHandlerMock);

  EXPECT_CALL(*environmentFactorHandlerMock, getData(Eq(ByRef(startFactor)))).Times(1).WillRepeatedly(Return(orgData));
  environmentVector.switchEnvironmentFactor(startFactor);

  ASSERT_EQ(ALL_RISK, environmentVector.currentRecode);
  ASSERT_EQ(startFactor, environmentVector.getCurrentEnvironmentFactor());

  const Container::HostVector& recodedData = environmentVector.getRecodedData();
  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());
  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*orgData)(i), recodedData(i));
  }

  orgData = nullptr;
}

TEST_F(EnvironmentVectorTest, Switch) {
  EnvironmentVector environmentVector(*environmentFactorHandlerMock);

  EXPECT_CALL(*environmentFactorHandlerMock, getData(Eq(ByRef(startFactor)))).Times(1).WillRepeatedly(Return(orgData));
  environmentVector.switchEnvironmentFactor(startFactor);

  EXPECT_CALL(*environmentFactorHandlerMock, getData(Eq(ByRef(binaryFactor)))).Times(1).WillRepeatedly(Return(binaryData));
  environmentVector.switchEnvironmentFactor(binaryFactor);

  ASSERT_EQ(binaryFactor, environmentVector.getCurrentEnvironmentFactor());

  const Container::HostVector& recodedData = environmentVector.getRecodedData();
  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());
  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*binaryData)(i), recodedData(i));
  }

  orgData = nullptr;
  binaryData = nullptr;
}

TEST_F(EnvironmentVectorTest, RecodeNonBinary) {
  EnvironmentVector environmentVector(*environmentFactorHandlerMock);

  EXPECT_CALL(*environmentFactorHandlerMock, getData(Eq(ByRef(startFactor)))).Times(1).WillRepeatedly(Return(orgData));
  environmentVector.switchEnvironmentFactor(startFactor);

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

  orgData = nullptr;
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

  EnvironmentVector environmentVector(*environmentFactorHandlerMock);

  EXPECT_CALL(*environmentFactorHandlerMock, getData(Eq(ByRef(binaryFactor)))).Times(1).WillRepeatedly(Return(binaryData));
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
  binaryData = nullptr;
}

TEST_F(EnvironmentVectorTest, StatisticModel) {
  EnvironmentVector environmentVector(*environmentFactorHandlerMock);

  EXPECT_CALL(*environmentFactorHandlerMock, getData(Eq(ByRef(startFactor)))).Times(1).WillRepeatedly(Return(orgData));
  environmentVector.switchEnvironmentFactor(startFactor);

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

  EXPECT_CALL(*environmentFactorHandlerMock, getData(Eq(ByRef(binaryFactor)))).Times(1).WillRepeatedly(Return(binaryData));
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
  orgData = nullptr;
  binaryData = nullptr;
}

TEST_F(EnvironmentVectorTest, RecodeDifferentOrder) {
  EnvironmentVector environmentVector(*environmentFactorHandlerMock);

  EXPECT_CALL(*environmentFactorHandlerMock, getData(Eq(ByRef(startFactor)))).Times(1).WillRepeatedly(Return(orgData));
  environmentVector.switchEnvironmentFactor(startFactor);

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

  orgData = nullptr;
}

} /* namespace Container */
} /* namespace CuEira */

