#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sstream>
#include <vector>

#include <HostVector.h>
#include <HostMatrix.h>
#include <LogisticRegressionResult.h>
#include <RegularHostVector.h>
#include <RegularHostMatrix.h>

namespace CuEira {
namespace Model {
namespace LogisticRegression {

using namespace CuEira::Container;

/**
 * Test for LogisticRegressionResultTest
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LogisticRegressionResultTest: public ::testing::Test {
protected:
  LogisticRegressionResultTest();
  virtual ~LogisticRegressionResultTest();
  virtual void SetUp();
  virtual void TearDown();
};

LogisticRegressionResultTest::LogisticRegressionResultTest() {

}

LogisticRegressionResultTest::~LogisticRegressionResultTest() {

}

void LogisticRegressionResultTest::SetUp() {

}

void LogisticRegressionResultTest::TearDown() {

}

TEST_F(LogisticRegressionResultTest, ConstructAndGet) {
  const int numberOfPredictors = 5;
  Container::HostVector* beta = new Container::RegularHostVector(numberOfPredictors);
  Container::HostMatrix* informationMatrix = new Container::RegularHostMatrix(numberOfPredictors, numberOfPredictors);
  Container::HostMatrix* inverseInformationMatrixHost = new Container::RegularHostMatrix(numberOfPredictors,
      numberOfPredictors);

  int numberOfIterations = 10;
  PRECISION logLikelihood = 0;

  LogisticRegressionResult lrResult(beta, informationMatrix, inverseInformationMatrixHost, numberOfIterations,
      logLikelihood);

  EXPECT_EQ(beta, &lrResult.getBeta());
  EXPECT_EQ(informationMatrix, &lrResult.getInformationMatrix());
  EXPECT_EQ(inverseInformationMatrixHost, &lrResult.getInverseInformationMatrix());
  EXPECT_EQ(logLikelihood, lrResult.getLogLikelihood());
  EXPECT_EQ(numberOfIterations, lrResult.getNumberOfIterations());
}

TEST_F(LogisticRegressionResultTest, Recode_ALL_RISK) {
  const int numberOfPredictors = 4;
  Container::HostVector* beta = new Container::RegularHostVector(numberOfPredictors);
  Container::HostMatrix* informationMatrix = new Container::RegularHostMatrix(numberOfPredictors, numberOfPredictors);
  Container::HostMatrix* inverseInformationMatrixHost = new Container::RegularHostMatrix(numberOfPredictors,
      numberOfPredictors);

  int numberOfIterations = 10;
  PRECISION logLikelihood = 0;

  (*beta)(0) = 0; //Intercept so doesn't mater
  (*beta)(1) = 1;
  (*beta)(2) = 1;
  (*beta)(3) = 1;

  LogisticRegressionResult lrResult(beta, informationMatrix, inverseInformationMatrixHost, numberOfIterations,
      logLikelihood);

  EXPECT_EQ(ALL_RISK, lrResult.calculateRecode());
}

TEST_F(LogisticRegressionResultTest, Recode_SNP_PROTECT) {
  const int numberOfPredictors = 4;
  Container::HostVector* beta = new Container::RegularHostVector(numberOfPredictors);
  Container::HostMatrix* informationMatrix = new Container::RegularHostMatrix(numberOfPredictors, numberOfPredictors);
  Container::HostMatrix* inverseInformationMatrixHost = new Container::RegularHostMatrix(numberOfPredictors,
      numberOfPredictors);

  int numberOfIterations = 10;
  PRECISION logLikelihood = 0;

  (*beta)(0) = 0; //Intercept so doesn't mater
  (*beta)(1) = -1;
  (*beta)(2) = 1;
  (*beta)(3) = 1;

  LogisticRegressionResult lrResult(beta, informationMatrix, inverseInformationMatrixHost, numberOfIterations,
      logLikelihood);

  EXPECT_EQ(SNP_PROTECT, lrResult.calculateRecode());
}

TEST_F(LogisticRegressionResultTest, Recode_ENVIRONMENT_PROTECT) {
  const int numberOfPredictors = 4;
  Container::HostVector* beta = new Container::RegularHostVector(numberOfPredictors);
  Container::HostMatrix* informationMatrix = new Container::RegularHostMatrix(numberOfPredictors, numberOfPredictors);
  Container::HostMatrix* inverseInformationMatrixHost = new Container::RegularHostMatrix(numberOfPredictors,
      numberOfPredictors);

  int numberOfIterations = 10;
  PRECISION logLikelihood = 0;

  (*beta)(0) = 0; //Intercept so doesn't mater
  (*beta)(1) = 1;
  (*beta)(2) = -1;
  (*beta)(3) = 1;

  LogisticRegressionResult lrResult(beta, informationMatrix, inverseInformationMatrixHost, numberOfIterations,
      logLikelihood);

  EXPECT_EQ(ENVIRONMENT_PROTECT, lrResult.calculateRecode());
}

TEST_F(LogisticRegressionResultTest, Recode_INTERACTION_PROTECT) {
  const int numberOfPredictors = 4;
  Container::HostVector* beta = new Container::RegularHostVector(numberOfPredictors);
  Container::HostMatrix* informationMatrix = new Container::RegularHostMatrix(numberOfPredictors, numberOfPredictors);
  Container::HostMatrix* inverseInformationMatrixHost = new Container::RegularHostMatrix(numberOfPredictors,
      numberOfPredictors);

  int numberOfIterations = 10;
  PRECISION logLikelihood = 0;

  (*beta)(0) = 0; //Intercept so doesn't mater
  (*beta)(1) = 1;
  (*beta)(2) = 1;
  (*beta)(3) = -1;

  LogisticRegressionResult lrResult(beta, informationMatrix, inverseInformationMatrixHost, numberOfIterations,
      logLikelihood);

  EXPECT_EQ(INTERACTION_PROTECT, lrResult.calculateRecode());
}

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

