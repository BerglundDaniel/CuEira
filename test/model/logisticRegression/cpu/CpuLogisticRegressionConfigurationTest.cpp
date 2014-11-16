#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sstream>

#include <HostVector.h>
#include <HostMatrix.h>
#include <RegularHostMatrix.h>
#include <RegularHostVector.h>
#include <CpuLogisticRegressionConfiguration.h>
#include <ConfigurationMock.h>
#include <MKLWrapper.h>

using testing::Return;
using testing::_;
using testing::AtLeast;

namespace CuEira {
namespace Model {
namespace LogisticRegression {
namespace CPU {

using namespace CuEira::Container;

/**
 * Test for ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuLogisticRegressionConfigurationTest: public ::testing::Test {
protected:
  CpuLogisticRegressionConfigurationTest();
  virtual ~CpuLogisticRegressionConfigurationTest();
  virtual void SetUp();
  virtual void TearDown();

  ConfigurationMock configMock;
  const double convergenceThreshold;
  const int numberOfMaxLRIterations;
  const int numberOfRows;
  const int numberOfCov;
  const int numberOfPredictorsNoCov;
  const int numberOfPredictorsWithCov;

  MKLWrapper blasWrapper;

  RegularHostVector outcomes;
  RegularHostMatrix covariates;
  RegularHostVector snpData;
  RegularHostVector environmentData;
  RegularHostVector interactionVector;
};

CpuLogisticRegressionConfigurationTest::CpuLogisticRegressionConfigurationTest() :
    convergenceThreshold(10e-5), numberOfMaxLRIterations(500), numberOfRows(10), numberOfCov(3), numberOfPredictorsNoCov(
        4), numberOfPredictorsWithCov(numberOfPredictorsNoCov + numberOfCov), outcomes(numberOfRows), covariates(
        numberOfRows, numberOfCov), snpData(numberOfRows), environmentData(numberOfRows), interactionVector(
        numberOfRows), blasWrapper() {

  EXPECT_CALL(configMock, getLRConvergenceThreshold()).Times(AtLeast(0)).WillRepeatedly(Return(convergenceThreshold));
  EXPECT_CALL(configMock, getNumberOfMaxLRIterations()).Times(AtLeast(0)).WillRepeatedly(
      Return(numberOfMaxLRIterations));

  for(int i = 0; i < numberOfRows; ++i){
    outcomes(i) = i;

    snpData(i) = i * 10;
    environmentData(i) = i * 100;
    interactionVector(i) = i * 1000;
  }

  for(int j = 0; j < numberOfCov; ++j){
    for(int i = 0; i < numberOfRows; ++i){
      covariates(i, j) = (j * numberOfRows) + i;
    }
  }
}

CpuLogisticRegressionConfigurationTest::~CpuLogisticRegressionConfigurationTest() {

}

void CpuLogisticRegressionConfigurationTest::SetUp() {

}

void CpuLogisticRegressionConfigurationTest::TearDown() {

}

TEST_F(CpuLogisticRegressionConfigurationTest, Constructor) {
  CpuLogisticRegressionConfiguration lrConfig(configMock, outcomes, blasWrapper);

  EXPECT_EQ(numberOfMaxLRIterations, lrConfig.getNumberOfMaxIterations());
  EXPECT_EQ(convergenceThreshold, lrConfig.getConvergenceThreshold());
  EXPECT_EQ(numberOfRows, lrConfig.getNumberOfRows());
  EXPECT_EQ(numberOfPredictorsNoCov, lrConfig.getNumberOfPredictors());

  const HostVector& outcomesReturn = lrConfig.getOutcomes();

  ASSERT_EQ(numberOfRows, outcomesReturn.getNumberOfRows());
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(outcomes(i), outcomesReturn(i));
  }

  const HostMatrix& predictorsReturn = lrConfig.getPredictors();
  ASSERT_EQ(numberOfRows, predictorsReturn.getNumberOfRows());
  ASSERT_EQ(numberOfPredictorsNoCov, predictorsReturn.getNumberOfColumns());

  //Intercept
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(1, predictorsReturn(i, 0));
  }
}

TEST_F(CpuLogisticRegressionConfigurationTest, ConstructorCov) {
  CpuLogisticRegressionConfiguration lrConfig(configMock, outcomes, blasWrapper, covariates);

  EXPECT_EQ(numberOfMaxLRIterations, lrConfig.getNumberOfMaxIterations());
  EXPECT_EQ(convergenceThreshold, lrConfig.getConvergenceThreshold());
  EXPECT_EQ(numberOfRows, lrConfig.getNumberOfRows());
  EXPECT_EQ(numberOfPredictorsNoCov, lrConfig.getNumberOfPredictors());

  const HostVector& outcomesReturn = lrConfig.getOutcomes();

  ASSERT_EQ(numberOfRows, outcomesReturn.getNumberOfRows());
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(outcomes(i), outcomesReturn(i));
  }

  const HostMatrix& predictorsReturn = lrConfig.getPredictors();
  ASSERT_EQ(numberOfRows, predictorsReturn.getNumberOfRows());
  ASSERT_EQ(numberOfPredictorsWithCov, predictorsReturn.getNumberOfColumns());

  //Intercept
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(1, predictorsReturn(i, 0));
  }

  //Covariates
  for(int j = 0; j < numberOfCov; ++j){
    for(int i = 0; i < numberOfRows; ++i){
      EXPECT_EQ((j * numberOfRows + i), predictorsReturn(i, j + 4));
    }
  }
}

TEST_F(CpuLogisticRegressionConfigurationTest, GetSNPEnvInteract) {
  CpuLogisticRegressionConfiguration lrConfig(configMock, outcomes, blasWrapper);

  EXPECT_EQ(numberOfMaxLRIterations, lrConfig.getNumberOfMaxIterations());
  EXPECT_EQ(convergenceThreshold, lrConfig.getConvergenceThreshold());
  EXPECT_EQ(numberOfRows, lrConfig.getNumberOfRows());
  EXPECT_EQ(numberOfPredictorsNoCov, lrConfig.getNumberOfPredictors());

  //Set snp
  lrConfig.setSNP(snpData);

  //Set env
  lrConfig.setEnvironmentFactor(environmentData);

  //Set interact
  lrConfig.setInteraction(interactionVector);

  const HostVector& outcomesReturn = lrConfig.getOutcomes();
  ASSERT_EQ(numberOfRows, outcomesReturn.getNumberOfRows());
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(outcomes(i), outcomesReturn(i));
  }

  const HostMatrix& predictorsReturn = lrConfig.getPredictors();
  ASSERT_EQ(numberOfRows, predictorsReturn.getNumberOfRows());
  ASSERT_EQ(numberOfPredictorsNoCov, predictorsReturn.getNumberOfColumns());

  //Intercept
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(1, predictorsReturn(i, 0));
  }

  //SNP
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i * 10, predictorsReturn(i, 1));
  }

  //Env
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i * 100, predictorsReturn(i, 2));
  }

  //Interact
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i * 1000, predictorsReturn(i, 3));
  }
}

TEST_F(CpuLogisticRegressionConfigurationTest, CovGetSNPEnvInteract) {
  CpuLogisticRegressionConfiguration lrConfig(configMock, outcomes, blasWrapper, covariates);

  EXPECT_EQ(numberOfMaxLRIterations, lrConfig.getNumberOfMaxIterations());
  EXPECT_EQ(convergenceThreshold, lrConfig.getConvergenceThreshold());
  EXPECT_EQ(numberOfRows, lrConfig.getNumberOfRows());
  EXPECT_EQ(numberOfPredictorsNoCov, lrConfig.getNumberOfPredictors());

  //Set snp
  lrConfig.setSNP(snpData);

  //Set env
  lrConfig.setEnvironmentFactor(environmentData);

  //Set interact
  lrConfig.setInteraction(interactionVector);

  const HostVector& outcomesReturn = lrConfig.getOutcomes();
  ASSERT_EQ(numberOfRows, outcomesReturn.getNumberOfRows());
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(outcomes(i), outcomesReturn(i));
  }

  const HostMatrix& predictorsReturn = lrConfig.getPredictors();
  ASSERT_EQ(numberOfRows, predictorsReturn.getNumberOfRows());
  ASSERT_EQ(numberOfPredictorsWithCov, predictorsReturn.getNumberOfColumns());

  //Intercept
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(1, predictorsReturn(i, 0));
  }

  //SNP
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i * 10, predictorsReturn(i, 1));
  }

  //Env
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i * 100, predictorsReturn(i, 2));
  }

  //Interact
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i * 1000, predictorsReturn(i, 3));
  }

  //Covariates
  for(int j = 0; j < numberOfCov; ++j){
    for(int i = 0; i < numberOfRows; ++i){
      EXPECT_EQ((j * numberOfRows + i), predictorsReturn(i, j + 4));
    }
  }
}

} /* namespace CPU */
} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */
