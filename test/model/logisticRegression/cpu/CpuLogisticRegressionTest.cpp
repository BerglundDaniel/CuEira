#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sstream>

#include <CpuLogisticRegression.h>
#include <CpuLogisticRegressionConfiguration.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <RegularHostMatrix.h>
#include <RegularHostVector.h>
#include <ConfigurationMock.h>

using testing::Return;
using testing::_;
using testing::AtLeast;
using testing::Ge;
using testing::Le;

namespace CuEira {
namespace Model {
namespace LogisticRegression {
namespace CPU {

/**
 * Test for CpuLogisticRegression
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuLogisticRegressionTest: public ::testing::Test {
protected:
  CpuLogisticRegressionTest();
  virtual ~CpuLogisticRegressionTest();
  virtual void SetUp();
  virtual void TearDown();

  ConfigurationMock configMock;
  const double convergenceThreshold;
  const int numberOfMaxLRIterations;
  const int numberOfCov;
  const int numberOfPredictorsNoCov;
  const int numberOfPredictorsWithCov;

  MKLWrapper blasWrapper;
};

CpuLogisticRegressionTest::CpuLogisticRegressionTest() :
    convergenceThreshold(1e-3), numberOfMaxLRIterations(500), numberOfCov(2), numberOfPredictorsNoCov(4), numberOfPredictorsWithCov(
        numberOfPredictorsNoCov + numberOfCov), blasWrapper() {

  EXPECT_CALL(configMock, getLRConvergenceThreshold()).Times(AtLeast(0)).WillRepeatedly(Return(convergenceThreshold));
  EXPECT_CALL(configMock, getNumberOfMaxLRIterations()).Times(AtLeast(0)).WillRepeatedly(
      Return(numberOfMaxLRIterations));

}

CpuLogisticRegressionTest::~CpuLogisticRegressionTest() {

}

void CpuLogisticRegressionTest::SetUp() {

}

void CpuLogisticRegressionTest::TearDown() {

}

TEST_F(CpuLogisticRegressionTest, calcuateProbabilites) {
  double e = 1e-5;
  double x, h, l;
  const int numberOfRows = 3;
  const int numberOfPredictors = 3;

  RegularHostVector outcomes(numberOfRows);

  CpuLogisticRegressionConfiguration* lrConfig = new CpuLogisticRegressionConfiguration(configMock, outcomes,
      blasWrapper);

  CpuLogisticRegression logisticRegression(lrConfig);

  RegularHostMatrix predictors(numberOfRows, numberOfPredictors);
  RegularHostVector beta(numberOfPredictors);
  RegularHostVector probabilites(numberOfRows);
  RegularHostVector workVectorNx1(numberOfRows);

  predictors(0, 0) = 1;
  predictors(1, 0) = 1;
  predictors(2, 0) = 1;

  predictors(0, 1) = 1;
  predictors(1, 1) = 2;
  predictors(2, 1) = 0.3;

  predictors(0, 2) = 0.1;
  predictors(1, 2) = 0.2;
  predictors(2, 2) = 0.5;

  beta(0) = 1;
  beta(1) = 2;
  beta(2) = 3;

  logisticRegression.calcuateProbabilites(predictors, beta, probabilites, workVectorNx1);

  x = 0.9644288;
  l = x - e;
  h = x + e;
  EXPECT_THAT(probabilites(0), Ge(l));
  EXPECT_THAT(probabilites(0), Le(h));

  x = 0.9963157;
  l = x - e;
  h = x + e;
  EXPECT_THAT(probabilites(1), Ge(l));
  EXPECT_THAT(probabilites(1), Le(h));

  x = 0.9568927;
  l = x - e;
  h = x + e;
  EXPECT_THAT(probabilites(2), Ge(l));
  EXPECT_THAT(probabilites(2), Le(h));
}

TEST_F(CpuLogisticRegressionTest, calculateScores) {
  double e = 1e-5;
  double x, h, l;
  const int numberOfRows = 3;
  const int numberOfPredictors = 3;

  RegularHostVector outcomes(numberOfRows);

  CpuLogisticRegressionConfiguration* lrConfig = new CpuLogisticRegressionConfiguration(configMock, outcomes,
      blasWrapper);

  CpuLogisticRegression logisticRegression(lrConfig);

  RegularHostMatrix predictors(numberOfRows, numberOfPredictors);
  RegularHostVector beta(numberOfPredictors);
  RegularHostVector probabilites(numberOfRows);
  RegularHostVector scores(numberOfPredictors);
  RegularHostVector workVectorNx1(numberOfRows);

  predictors(0, 0) = 1;
  predictors(1, 0) = 1;
  predictors(2, 0) = 1;

  predictors(0, 1) = 1;
  predictors(1, 1) = 2;
  predictors(2, 1) = 0.3;

  predictors(0, 2) = 0.1;
  predictors(1, 2) = 0.2;
  predictors(2, 2) = 0.5;

  beta(0) = 1;
  beta(1) = 2;
  beta(2) = 3;

  probabilites(0) = 0.9644288;
  probabilites(1) = 0.9963157;
  probabilites(2) = 0.9568927;

  outcomes(0) = 1;
  outcomes(1) = 1;
  outcomes(2) = 0;

  logisticRegression.calculateScores(predictors, outcomes, probabilites, scores, workVectorNx1);

  x = -0.9176373;
  l = x - e;
  h = x + e;
  EXPECT_THAT(scores(0), Ge(l));
  EXPECT_THAT(scores(0), Le(h));

  x = -0.2441281;
  l = x - e;
  h = x + e;
  EXPECT_THAT(scores(1), Ge(l));
  EXPECT_THAT(scores(1), Le(h));

  x = -0.4741524;
  l = x - e;
  h = x + e;
  EXPECT_THAT(scores(2), Ge(l));
  EXPECT_THAT(scores(2), Le(h));
}

TEST_F(CpuLogisticRegressionTest, calculateInformationMatrix) {
  double e = 1e-5;
  double x, h, l;
  const int numberOfRows = 3;
  const int numberOfPredictors = 4;

  RegularHostVector outcomes(numberOfRows);

  CpuLogisticRegressionConfiguration* lrConfig = new CpuLogisticRegressionConfiguration(configMock, outcomes,
      blasWrapper);

  CpuLogisticRegression logisticRegression(lrConfig);

  RegularHostMatrix predictors(numberOfRows, numberOfPredictors);
  RegularHostVector probabilites(numberOfRows);
  RegularHostVector workVectorNx1(numberOfRows);
  RegularHostMatrix informationMatrix(numberOfPredictors, numberOfPredictors);
  RegularHostMatrix workMatrixNxM(numberOfRows, numberOfPredictors);

  predictors(0, 0) = 1;
  predictors(1, 0) = 1;
  predictors(2, 0) = 1;

  predictors(0, 1) = 1;
  predictors(1, 1) = 2;
  predictors(2, 1) = 0.3;

  predictors(0, 2) = 0.1;
  predictors(1, 2) = 0.2;
  predictors(2, 2) = 0.5;

  predictors(0, 3) = 2;
  predictors(1, 3) = 5;
  predictors(2, 3) = 6;

  probabilites(0) = 0.9;
  probabilites(1) = 0.3;
  probabilites(2) = 0.5;

  logisticRegression.calculateInformationMatrix(predictors, probabilites, workVectorNx1, informationMatrix,
      workMatrixNxM);

  x = 0.55;
  l = x - e;
  h = x + e;
  EXPECT_THAT(informationMatrix(0, 0), Ge(l));
  EXPECT_THAT(informationMatrix(0, 0), Le(h));

  x = 0.585;
  l = x - e;
  h = x + e;
  EXPECT_THAT(informationMatrix(0, 1), Ge(l));
  EXPECT_THAT(informationMatrix(0, 1), Le(h));

  x = 0.176;
  l = x - e;
  h = x + e;
  EXPECT_THAT(informationMatrix(0, 2), Ge(l));
  EXPECT_THAT(informationMatrix(0, 2), Le(h));

  x = 1; //FIXME
  l = x - e;
  h = x + e;
  EXPECT_THAT(informationMatrix(0, 3), Ge(l));
  EXPECT_THAT(informationMatrix(0, 3), Le(h));

  x = 0.9525;
  l = x - e;
  h = x + e;
  EXPECT_THAT(informationMatrix(1, 1), Ge(l));
  EXPECT_THAT(informationMatrix(1, 1), Le(h));

  x = 0.1305;
  l = x - e;
  h = x + e;
  EXPECT_THAT(informationMatrix(1, 2), Ge(l));
  EXPECT_THAT(informationMatrix(1, 2), Le(h));

  x = 1; //FIXME
  l = x - e;
  h = x + e;
  EXPECT_THAT(informationMatrix(1, 3), Ge(l));
  EXPECT_THAT(informationMatrix(1, 3), Le(h));

  x = 0.0718;
  l = x - e;
  h = x + e;
  EXPECT_THAT(informationMatrix(2, 2), Ge(l));
  EXPECT_THAT(informationMatrix(2, 2), Le(h));

  x = 1; //FIXME
  l = x - e;
  h = x + e;
  EXPECT_THAT(informationMatrix(2, 3), Ge(l));
  EXPECT_THAT(informationMatrix(2, 3), Le(h));

  x = 1; //FIXME
  l = x - e;
  h = x + e;
  EXPECT_THAT(informationMatrix(3, 3), Ge(l));
  EXPECT_THAT(informationMatrix(3, 3), Le(h));

  EXPECT_EQ(informationMatrix(1, 0), informationMatrix(0, 1));
  EXPECT_EQ(informationMatrix(2, 0), informationMatrix(0, 2));
  EXPECT_EQ(informationMatrix(2, 1), informationMatrix(1, 2));
  EXPECT_EQ(informationMatrix(3, 0), informationMatrix(0, 3));
  EXPECT_EQ(informationMatrix(3, 1), informationMatrix(1, 3));
  EXPECT_EQ(informationMatrix(3, 2), informationMatrix(2, 3));
}

TEST_F(CpuLogisticRegressionTest, calculateLogLikelihood) {
  double e = 1e-5;
  double x, h, l;
  const int numberOfRows = 3;
  const int numberOfPredictors = 3;

  RegularHostVector outcomes(numberOfRows);

  CpuLogisticRegressionConfiguration* lrConfig = new CpuLogisticRegressionConfiguration(configMock, outcomes,
      blasWrapper);

  CpuLogisticRegression logisticRegression(lrConfig);

  RegularHostMatrix predictors(numberOfRows, numberOfPredictors);
  RegularHostVector probabilites(numberOfRows);
  PRECISION logLikelihood = 0;

  predictors(0, 0) = 1;
  predictors(1, 0) = 1;
  predictors(2, 0) = 1;

  predictors(0, 1) = 1;
  predictors(1, 1) = 2;
  predictors(2, 1) = 0.3;

  predictors(0, 2) = 0.1;
  predictors(1, 2) = 0.2;
  predictors(2, 2) = 0.5;

  probabilites(0) = 0.9644288;
  probabilites(1) = 0.9963157;
  probabilites(2) = 0.9568927;

  outcomes(0) = 1;
  outcomes(1) = 1;
  outcomes(2) = 0;

  logisticRegression.calculateLogLikelihood(outcomes, probabilites, logLikelihood);

  x = -3.18397427;
  l = x - e;
  h = x + e;
  EXPECT_THAT(logLikelihood, Ge(l));
  EXPECT_THAT(logLikelihood, Le(h));
}

TEST_F(CpuLogisticRegressionTest, SmallTestNoCov) {
  double e = 1e-4;
  const int numberOfRows = 10;
  RegularHostVector outcomes(numberOfRows);
  RegularHostVector snpData(numberOfRows);
  RegularHostVector environmentData(numberOfRows);
  RegularHostVector interactionVector(numberOfRows);

  //SNP
  snpData(0) = 1.33;
  snpData(1) = -0.2;
  snpData(2) = 0.29;
  snpData(3) = 0.49;
  snpData(4) = -0.57;
  snpData(5) = -1;
  snpData(6) = 0.1;
  snpData(7) = -0.38;
  snpData(8) = 0.25;
  snpData(9) = -0.69;

  //Env
  environmentData(0) = 1.49;
  environmentData(1) = -0.99;
  environmentData(2) = 1.16;
  environmentData(3) = 0.49;
  environmentData(4) = 0.76;
  environmentData(5) = -0.3;
  environmentData(6) = -0.92;
  environmentData(7) = -0.6;
  environmentData(8) = -0.6;
  environmentData(9) = 0.32;

  //Interaction
  interactionVector(0) = 1;
  interactionVector(1) = 1;
  interactionVector(2) = 1;
  interactionVector(3) = 1;
  interactionVector(4) = 1;
  interactionVector(5) = 0;
  interactionVector(6) = 0;
  interactionVector(7) = 0;
  interactionVector(8) = 0;
  interactionVector(9) = 0;

  outcomes(0) = 1;
  outcomes(1) = 1;
  outcomes(2) = 0;
  outcomes(3) = 0;
  outcomes(4) = 1;
  outcomes(5) = 0;
  outcomes(6) = 1;
  outcomes(7) = 0;
  outcomes(8) = 1;
  outcomes(9) = 1;

  std::vector<PRECISION> correctBeta(numberOfPredictorsNoCov);
  correctBeta[0] = 0.4563;
  correctBeta[1] = 0.7382;
  correctBeta[2] = -0.5478;
  correctBeta[3] = 0.0867;

  CpuLogisticRegressionConfiguration* lrConfig = new CpuLogisticRegressionConfiguration(configMock, outcomes,
      blasWrapper);
  lrConfig->setSNP(snpData);
  lrConfig->setEnvironmentFactor(environmentData);
  lrConfig->setInteraction(interactionVector);

  CpuLogisticRegression logisticRegression(lrConfig);
  LogisticRegressionResult* lrResult = logisticRegression.calculate();
  const HostVector& beta = lrResult->getBeta();

  for(int i = 0; i < numberOfPredictorsNoCov; ++i){
    PRECISION l = correctBeta[i] - e;
    PRECISION h = correctBeta[i] + e;

    EXPECT_THAT(beta(i), Ge(l));
    EXPECT_THAT(beta(i), Le(h));
  }

  delete lrResult;
}

TEST_F(CpuLogisticRegressionTest, SmallTestNoCovIntOnly) {
  double e = 1e-4;
  const int numberOfRows = 10;
  RegularHostVector outcomes(numberOfRows);
  RegularHostVector snpData(numberOfRows);
  RegularHostVector environmentData(numberOfRows);
  RegularHostVector interactionVector(numberOfRows);

  //SNP
  snpData(0) = 1;
  snpData(1) = 0;
  snpData(2) = 2;
  snpData(3) = 4;
  snpData(4) = 7;
  snpData(5) = -1;
  snpData(6) = 1;
  snpData(7) = 3;
  snpData(8) = -2;
  snpData(9) = -6;

  //Env
  environmentData(0) = 1;
  environmentData(1) = 0;
  environmentData(2) = 1;
  environmentData(3) = 4;
  environmentData(4) = 2;
  environmentData(5) = -3;
  environmentData(6) = -1;
  environmentData(7) = 6;
  environmentData(8) = 4;
  environmentData(9) = 3;

  //Interaction
  interactionVector(0) = 1;
  interactionVector(1) = 1;
  interactionVector(2) = 1;
  interactionVector(3) = 1;
  interactionVector(4) = 1;
  interactionVector(5) = 0;
  interactionVector(6) = 0;
  interactionVector(7) = 0;
  interactionVector(8) = 0;
  interactionVector(9) = 0;

  outcomes(0) = 1;
  outcomes(1) = 1;
  outcomes(2) = 0;
  outcomes(3) = 0;
  outcomes(4) = 1;
  outcomes(5) = 0;
  outcomes(6) = 1;
  outcomes(7) = 0;
  outcomes(8) = 1;
  outcomes(9) = 1;

  std::vector<PRECISION> correctBeta(numberOfPredictorsNoCov);
  correctBeta[0] = 0.2500;
  correctBeta[1] = -0.2557;
  correctBeta[2] = -0.0200;
  correctBeta[3] = 0.9337;

  CpuLogisticRegressionConfiguration* lrConfig = new CpuLogisticRegressionConfiguration(configMock, outcomes,
      blasWrapper);

  lrConfig->setSNP(snpData);
  lrConfig->setEnvironmentFactor(environmentData);
  lrConfig->setInteraction(interactionVector);

  CpuLogisticRegression logisticRegression(lrConfig);
  LogisticRegressionResult* lrResult = logisticRegression.calculate();
  const HostVector& beta = lrResult->getBeta();

  for(int i = 0; i < numberOfPredictorsNoCov; ++i){
    PRECISION l = correctBeta[i] - e;
    PRECISION h = correctBeta[i] + e;

    EXPECT_THAT(beta(i), Ge(l));
    EXPECT_THAT(beta(i), Le(h));
  }

  delete lrResult;
}

TEST_F(CpuLogisticRegressionTest, SmallTestCov) {
  //TODO
}

} /* namespace CPU */
} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

