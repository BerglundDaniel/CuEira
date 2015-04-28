#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include <Recode.h>
#include <CpuEnvironmentVector.h>
#include <EnvironmentFactor.h>
#include <VariableType.h>
#include <Id.h>
#include <EnvironmentFactorHandlerMock.h>
#include <MissingDataHandlerMock.h>
#include <InvalidState.h>
#include <HostVector.h>
#include <RegularHostVector.h>

using testing::Return;
using testing::AtLeast;
using testing::ReturnRef;
using testing::Eq;
using testing::ByRef;
using testing::Invoke;

namespace CuEira {

namespace CuEira_Test {

void FillExMissingVector(const Container::RegularHostVector& fromVector, Container::RegularHostVector& toVector) {
  const int size = toVector.getNumberOfRows();
  for(int i = 0; i < size; ++i){
    toVector(i) = fromVector(i);
  }
}

}/* namespace CuEira_Test */

namespace Container {
namespace CPU {

/**
 * Test for testing CpuEnvironmentVector
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuEnvironmentVectorTest: public ::testing::Test {
protected:
  CpuEnvironmentVectorTest();
  virtual ~CpuEnvironmentVectorTest();
  virtual void SetUp();
  virtual void TearDown();

  const int numberOfIndividualsTotal;
  EnvironmentFactorHandlerMock<RegularHostVector> environmentFactorHandlerMock;
  EnvironmentFactor envFactor;
  EnvironmentFactor binaryEnvFactor;
  Container::RegularHostVector normalData;
  Container::RegularHostVector binaryData;
  MKLWrapper mklWrapper;
};

CpuEnvironmentVectorTest::CpuEnvironmentVectorTest() :
    numberOfIndividualsTotal(9), envFactor(Id("startFactor")), binaryEnvFactor(Id("binaryFactor")), normalData(
        numberOfIndividualsTotal), binaryData(numberOfIndividualsTotal)

{
  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    normalData(i) = i;

    if(i < 5 && i > 2){
      binaryData(i) = 1;
    }else{
      binaryData(i) = 0;
    }
  }

  envFactor.setVariableType(OTHER);
  binaryEnvFactor.setVariableType(BINARY);

  envFactor.setMax(numberOfIndividualsTotal - 1);
  envFactor.setMin(-1);
  normalData(3) = -1;

  binaryEnvFactor.setMax(1);
  binaryEnvFactor.setMin(0);
}

CpuEnvironmentVectorTest::~CpuEnvironmentVectorTest() {

}

void CpuEnvironmentVectorTest::SetUp() {
  EXPECT_CALL(environmentFactorHandlerMock, getNumberOfIndividualsTotal()).Times(1).WillRepeatedly(
      Return(numberOfIndividualsTotal));
}

void CpuEnvironmentVectorTest::TearDown() {

}

#ifdef DEBUG
TEST_F(CpuEnvironmentVectorTest, ConstructAndGetException){
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentFactor()).Times(1).WillRepeatedly(ReturnRef(envFactor));
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentData()).Times(1).WillRepeatedly(ReturnRef(normalData));

  CpuEnvironmentVector cpuEnvironmentVector(environmentFactorHandlerMock, mklWrapper);

  EXPECT_THROW(cpuEnvironmentVector.getNumberOfIndividualsToInclude(), InvalidState);
  EXPECT_THROW(cpuEnvironmentVector.getEnvironmentData(), InvalidState);
}
#endif

TEST_F(CpuEnvironmentVectorTest, RecodeBinaryNoMissing) {
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentFactor()).Times(1).WillRepeatedly(ReturnRef(binaryEnvFactor));
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentData()).Times(1).WillRepeatedly(ReturnRef(binaryData));

  CpuEnvironmentVector cpuEnvironmentVector(environmentFactorHandlerMock, mklWrapper);
  EXPECT_EQ(binaryEnvFactor, cpuEnvironmentVector.getEnvironmentFactor());
  EXPECT_EQ(numberOfIndividualsTotal, cpuEnvironmentVector.getNumberOfIndividualsTotal());

  //ALL_RISK
  cpuEnvironmentVector.recode(ALL_RISK);
  EXPECT_EQ(numberOfIndividualsTotal, cpuEnvironmentVector.getNumberOfIndividualsToInclude());
  const RegularHostVector& allRiskVector1 = cpuEnvironmentVector.getEnvironmentData();
  EXPECT_EQ(numberOfIndividualsTotal, allRiskVector1.getNumberOfRows());

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    EXPECT_EQ(binaryData(i), allRiskVector1(i));
  }

  //ENVIRONMENT_PROTECT
  cpuEnvironmentVector.recode(ENVIRONMENT_PROTECT);
  EXPECT_EQ(numberOfIndividualsTotal, cpuEnvironmentVector.getNumberOfIndividualsToInclude());
  const RegularHostVector& envProctVector1 = cpuEnvironmentVector.getEnvironmentData();
  EXPECT_EQ(numberOfIndividualsTotal, envProctVector1.getNumberOfRows());

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    EXPECT_EQ(1 - binaryData(i), envProctVector1(i));
  }

  //INTERACTION_PROTECT
  cpuEnvironmentVector.recode(INTERACTION_PROTECT);
  EXPECT_EQ(numberOfIndividualsTotal, cpuEnvironmentVector.getNumberOfIndividualsToInclude());
  const RegularHostVector& interProctVector1 = cpuEnvironmentVector.getEnvironmentData();
  EXPECT_EQ(numberOfIndividualsTotal, interProctVector1.getNumberOfRows());

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    EXPECT_EQ(1 - binaryData(i), interProctVector1(i));
  }

  //ALL_RISK
  cpuEnvironmentVector.recode(ALL_RISK);
  EXPECT_EQ(numberOfIndividualsTotal, cpuEnvironmentVector.getNumberOfIndividualsToInclude());
  const RegularHostVector& allRiskVector2 = cpuEnvironmentVector.getEnvironmentData();
  EXPECT_EQ(numberOfIndividualsTotal, allRiskVector2.getNumberOfRows());

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    EXPECT_EQ(binaryData(i), allRiskVector2(i));
  }
}

TEST_F(CpuEnvironmentVectorTest, RecodeBinaryMissing) {
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentFactor()).Times(1).WillRepeatedly(ReturnRef(binaryEnvFactor));
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentData()).Times(1).WillRepeatedly(ReturnRef(binaryData));

  CpuEnvironmentVector cpuEnvironmentVector(environmentFactorHandlerMock, mklWrapper);
  EXPECT_EQ(binaryEnvFactor, cpuEnvironmentVector.getEnvironmentFactor());
  EXPECT_EQ(numberOfIndividualsTotal, cpuEnvironmentVector.getNumberOfIndividualsTotal());

  MissingDataHandlerMock<RegularHostVector> missingDataHandlerMock;

  EXPECT_CALL(missingDataHandlerMock, copyNonMissing(_,_)).Times(2).WillRepeatedly(
      Invoke(CuEira_Test::FillExMissingVector));

  //INTERACTION_PROTECT
  const int numberOfIndvidualsExMissing1 = numberOfIndividualsTotal - 4;
  EXPECT_CALL(missingDataHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillOnce(
      Return(numberOfIndvidualsExMissing1));

  cpuEnvironmentVector.recode(INTERACTION_PROTECT, missingDataHandlerMock);

  const RegularHostVector& interProctVector1 = cpuEnvironmentVector.getEnvironmentData();
  ASSERT_EQ(numberOfIndvidualsExMissing1, interProctVector1.getNumberOfRows());
  for(int i = 0; i < numberOfIndvidualsExMissing1; ++i){
    EXPECT_EQ(1 - binaryData(i), interProctVector1(i));
  }

  //ALL_RISK
  const int numberOfIndvidualsExMissing2 = numberOfIndividualsTotal - 3;
  EXPECT_CALL(missingDataHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillOnce(
      Return(numberOfIndvidualsExMissing2));

  cpuEnvironmentVector.recode(ALL_RISK, missingDataHandlerMock);

  const RegularHostVector& allRiskVector1 = cpuEnvironmentVector.getEnvironmentData();
  ASSERT_EQ(numberOfIndvidualsExMissing2, allRiskVector1.getNumberOfRows());
  for(int i = 0; i < numberOfIndvidualsExMissing2; ++i){
    EXPECT_EQ(binaryData(i), allRiskVector1(i));
  }
}

TEST_F(CpuEnvironmentVectorTest, RecodeNonBinaryNoMissing) {
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentFactor()).Times(1).WillRepeatedly(ReturnRef(envFactor));
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentData()).Times(1).WillRepeatedly(ReturnRef(normalData));

  const int c = envFactor.getMax() - envFactor.getMin();

  CpuEnvironmentVector cpuEnvironmentVector(environmentFactorHandlerMock, mklWrapper);
  EXPECT_EQ(envFactor, cpuEnvironmentVector.getEnvironmentFactor());
  EXPECT_EQ(numberOfIndividualsTotal, cpuEnvironmentVector.getNumberOfIndividualsTotal());

  //ALL_RISK
  cpuEnvironmentVector.recode(ALL_RISK);
  EXPECT_EQ(numberOfIndividualsTotal, cpuEnvironmentVector.getNumberOfIndividualsToInclude());
  const RegularHostVector& allRiskVector1 = cpuEnvironmentVector.getEnvironmentData();
  EXPECT_EQ(numberOfIndividualsTotal, allRiskVector1.getNumberOfRows());

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    EXPECT_EQ(normalData(i), allRiskVector1(i));
  }

  //ENVIRONMENT_PROTECT
  cpuEnvironmentVector.recode(ENVIRONMENT_PROTECT);
  EXPECT_EQ(numberOfIndividualsTotal, cpuEnvironmentVector.getNumberOfIndividualsToInclude());
  const RegularHostVector& envProctVector1 = cpuEnvironmentVector.getEnvironmentData();
  EXPECT_EQ(numberOfIndividualsTotal, envProctVector1.getNumberOfRows());

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    EXPECT_EQ(c - normalData(i), envProctVector1(i));
  }

  //INTERACTION_PROTECT
  cpuEnvironmentVector.recode(INTERACTION_PROTECT);
  EXPECT_EQ(numberOfIndividualsTotal, cpuEnvironmentVector.getNumberOfIndividualsToInclude());
  const RegularHostVector& interProctVector1 = cpuEnvironmentVector.getEnvironmentData();
  EXPECT_EQ(numberOfIndividualsTotal, interProctVector1.getNumberOfRows());

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    EXPECT_EQ(c - normalData(i), interProctVector1(i));
  }

  //ALL_RISK
  cpuEnvironmentVector.recode(ALL_RISK);
  EXPECT_EQ(numberOfIndividualsTotal, cpuEnvironmentVector.getNumberOfIndividualsToInclude());
  const RegularHostVector& allRiskVector2 = cpuEnvironmentVector.getEnvironmentData();
  EXPECT_EQ(numberOfIndividualsTotal, allRiskVector2.getNumberOfRows());

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    EXPECT_EQ(normalData(i), allRiskVector2(i));
  }
}

TEST_F(CpuEnvironmentVectorTest, RecodeNonBinaryMissing) {
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentFactor()).Times(1).WillRepeatedly(ReturnRef(envFactor));
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentData()).Times(1).WillRepeatedly(ReturnRef(normalData));

  const int c = envFactor.getMax() - envFactor.getMin();

  CpuEnvironmentVector cpuEnvironmentVector(environmentFactorHandlerMock, mklWrapper);
  EXPECT_EQ(envFactor, cpuEnvironmentVector.getEnvironmentFactor());
  EXPECT_EQ(numberOfIndividualsTotal, cpuEnvironmentVector.getNumberOfIndividualsTotal());

  MissingDataHandlerMock<RegularHostVector> missingDataHandlerMock;

  EXPECT_CALL(missingDataHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillRepeatedly(
      Return(numberOfIndvidualsExMissing));
  EXPECT_CALL(missingDataHandlerMock, copyNonMissing(_,_)).Times(1);

  EXPECT_CALL(missingDataHandlerMock, copyNonMissing(_,_)).Times(2).WillRepeatedly(
      Invoke(CuEira_Test::FillExMissingVector));

  //INTERACTION_PROTECT
  const int numberOfIndvidualsExMissing1 = numberOfIndividualsTotal - 4;
  EXPECT_CALL(missingDataHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillOnce(
      Return(numberOfIndvidualsExMissing1));

  cpuEnvironmentVector.recode(INTERACTION_PROTECT, missingDataHandlerMock);

  const RegularHostVector& interProctVector1 = cpuEnvironmentVector.getEnvironmentData();
  ASSERT_EQ(numberOfIndvidualsExMissing1, interProctVector1.getNumberOfRows());
  for(int i = 0; i < numberOfIndvidualsExMissing1; ++i){
    EXPECT_EQ(c - normalData(i), interProctVector1(i));
  }

  //ALL_RISK
  const int numberOfIndvidualsExMissing2 = numberOfIndividualsTotal - 3;
  EXPECT_CALL(missingDataHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillOnce(
      Return(numberOfIndvidualsExMissing2));

  cpuEnvironmentVector.recode(ALL_RISK, missingDataHandlerMock);

  const RegularHostVector& allRiskVector1 = cpuEnvironmentVector.getEnvironmentData();
  ASSERT_EQ(numberOfIndvidualsExMissing2, allRiskVector1.getNumberOfRows());
  for(int i = 0; i < numberOfIndvidualsExMissing2; ++i){
    EXPECT_EQ(normalData(i), allRiskVector1(i));
  }
}

TEST_F(CpuEnvironmentVectorTest, RecodeBinaryMixed) {
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentFactor()).Times(1).WillRepeatedly(ReturnRef(binaryEnvFactor));
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentData()).Times(1).WillRepeatedly(ReturnRef(binaryData));

  CpuEnvironmentVector cpuEnvironmentVector(environmentFactorHandlerMock, mklWrapper);
  EXPECT_EQ(binaryEnvFactor, cpuEnvironmentVector.getEnvironmentFactor());
  EXPECT_EQ(numberOfIndividualsTotal, cpuEnvironmentVector.getNumberOfIndividualsTotal());

  MissingDataHandlerMock<RegularHostVector> missingDataHandlerMock;

  EXPECT_CALL(missingDataHandlerMock, copyNonMissing(_,_)).Times(2).WillRepeatedly(
      Invoke(CuEira_Test::FillExMissingVector));

  //INTERACTION_PROTECT
  const int numberOfIndvidualsExMissing1 = numberOfIndividualsTotal - 4;
  EXPECT_CALL(missingDataHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillOnce(
      Return(numberOfIndvidualsExMissing1));

  cpuEnvironmentVector.recode(INTERACTION_PROTECT, missingDataHandlerMock);

  const RegularHostVector& interProctVector1 = cpuEnvironmentVector.getEnvironmentData();
  ASSERT_EQ(numberOfIndvidualsExMissing1, interProctVector1.getNumberOfRows());
  for(int i = 0; i < numberOfIndvidualsExMissing1; ++i){
    EXPECT_EQ(1 - binaryData(i), interProctVector1(i));
  }

  //ALL_RISK
  cpuEnvironmentVector.recode(ALL_RISK);

  const RegularHostVector& allRiskVector1 = cpuEnvironmentVector.getEnvironmentData();
  ASSERT_EQ(numberOfIndividualsTotal, allRiskVector1.getNumberOfRows());
  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    EXPECT_EQ(binaryData(i), allRiskVector1(i));
  }

  //ENVIRONMENT_PROTECT
  const int numberOfIndvidualsExMissing3 = numberOfIndividualsTotal - 1;
  EXPECT_CALL(missingDataHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillOnce(
      Return(numberOfIndvidualsExMissing3));

  cpuEnvironmentVector.recode(ENVIRONMENT_PROTECT, missingDataHandlerMock);

  const RegularHostVector& envProctVector1 = cpuEnvironmentVector.getEnvironmentData();
  ASSERT_EQ(numberOfIndvidualsExMissing3, envProctVector1.getNumberOfRows());
  for(int i = 0; i < numberOfIndvidualsExMissing3; ++i){
    EXPECT_EQ(1 - binaryData(i), envProctVector1(i));
  }
}

} /* namespace CPU */
} /* namespace Container */
} /* namespace CuEira */

