#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sstream>
#include <lapackpp/gmd.h>
#include <lapackpp/lavd.h>

#include <MultipleLogisticRegression.h>

using ::testing::Ge;
using ::testing::Le;

namespace CuEira {
namespace Model {
namespace LogisticRegression {
namespace Serial {

/**
 * Test for ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LogisticRegressionTest: public ::testing::Test {
protected:
  LogisticRegressionTest();
  virtual ~LogisticRegressionTest();
  virtual void SetUp();
  virtual void TearDown();

};

LogisticRegressionTest::LogisticRegressionTest() {

}

LogisticRegressionTest::~LogisticRegressionTest() {

}

void LogisticRegressionTest::SetUp() {

}

void LogisticRegressionTest::TearDown() {

}

TEST_F(LogisticRegressionTest, SmallTestNoCov) {
  const int numberOfRows = 10;
  const int numberOfPredictors = 3;
  double e = 1e-4;

  LaVectorDouble outcomes(numberOfRows);
  LaVectorDouble* betaCoefficients = new LaVectorDouble(numberOfPredictors + 1);
  LaGenMatDouble predictors(numberOfRows, numberOfPredictors);

  //SNP
  predictors(0, 0) = 1.33;
  predictors(1, 0) = -0.2;
  predictors(2, 0) = 0.29;
  predictors(3, 0) = 0.49;
  predictors(4, 0) = -0.57;
  predictors(5, 0) = -1;
  predictors(6, 0) = 0.1;
  predictors(7, 0) = -0.38;
  predictors(8, 0) = 0.25;
  predictors(9, 0) = -0.69;

  //Env
  predictors(0, 1) = 1.49;
  predictors(1, 1) = -0.99;
  predictors(2, 1) = 1.16;
  predictors(3, 1) = 0.49;
  predictors(4, 1) = 0.76;
  predictors(5, 1) = -0.3;
  predictors(6, 1) = -0.92;
  predictors(7, 1) = -0.6;
  predictors(8, 1) = -0.6;
  predictors(9, 1) = 0.32;

  //Interaction
  predictors(0, 2) = 1;
  predictors(1, 2) = 1;
  predictors(2, 2) = 1;
  predictors(3, 2) = 1;
  predictors(4, 2) = 1;
  predictors(5, 2) = 0;
  predictors(6, 2) = 0;
  predictors(7, 2) = 0;
  predictors(8, 2) = 0;
  predictors(9, 2) = 0;

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

  //Initial beta
  (*betaCoefficients)(0) = 0;
  (*betaCoefficients)(1) = 0;
  (*betaCoefficients)(2) = 0;
  (*betaCoefficients)(3) = 0;

  std::vector<PRECISION> correctBeta(numberOfPredictors + 1);
  correctBeta[0] = 0.4563;
  correctBeta[1] = 0.7382;
  correctBeta[2] = -0.5478;
  correctBeta[3] = 0.0867;

  MultipleLogisticRegression lr(predictors, outcomes, betaCoefficients);
  lr.calculate();

  const LaVectorDouble& beta = lr.getBeta();

  for(int i = 0; i < numberOfPredictors + 1; ++i){
    PRECISION l = correctBeta[i] - e;
    PRECISION h = correctBeta[i] + e;

    EXPECT_THAT(beta(i), Ge(l));
    EXPECT_THAT(beta(i), Le(h));
  }

  delete betaCoefficients;
}

TEST_F(LogisticRegressionTest, SmallTestNoCovIntOnly) {
  const int numberOfRows = 10;
  const int numberOfPredictors = 3;
  double e = 1e-4;

  LaVectorDouble outcomes(numberOfRows);
  LaVectorDouble* betaCoefficients = new LaVectorDouble(numberOfPredictors + 1);
  LaGenMatDouble predictors(numberOfRows, numberOfPredictors);

  //SNP
  predictors(0, 0) = 1;
  predictors(1, 0) = 0;
  predictors(2, 0) = 2;
  predictors(3, 0) = 4;
  predictors(4, 0) = 7;
  predictors(5, 0) = -1;
  predictors(6, 0) = 1;
  predictors(7, 0) = 3;
  predictors(8, 0) = -2;
  predictors(9, 0) = -6;

  //Env
  predictors(0, 1) = 1;
  predictors(1, 1) = 0;
  predictors(2, 1) = 1;
  predictors(3, 1) = 4;
  predictors(4, 1) = 2;
  predictors(5, 1) = -3;
  predictors(6, 1) = -1;
  predictors(7, 1) = 6;
  predictors(8, 1) = 4;
  predictors(9, 1) = 3;

  //Interaction
  predictors(0, 2) = 1;
  predictors(1, 2) = 1;
  predictors(2, 2) = 1;
  predictors(3, 2) = 1;
  predictors(4, 2) = 1;
  predictors(5, 2) = 0;
  predictors(6, 2) = 0;
  predictors(7, 2) = 0;
  predictors(8, 2) = 0;
  predictors(9, 2) = 0;

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

  //Initial beta
  (*betaCoefficients)(0) = 0;
  (*betaCoefficients)(1) = 0;
  (*betaCoefficients)(2) = 0;
  (*betaCoefficients)(3) = 0;

  std::vector<PRECISION> correctBeta(numberOfPredictors + 1);
  correctBeta[0] = 0.2500;
  correctBeta[1] = -0.2557;
  correctBeta[2] = -0.0200;
  correctBeta[3] = 0.9337;

  MultipleLogisticRegression lr(predictors, outcomes, betaCoefficients);
  lr.calculate();

  const LaVectorDouble& beta = lr.getBeta();

  for(int i = 0; i < numberOfPredictors + 1; ++i){
    PRECISION l = correctBeta[i] - e;
    PRECISION h = correctBeta[i] + e;

    EXPECT_THAT(beta(i), Ge(l));
    EXPECT_THAT(beta(i), Le(h));
  }

  delete betaCoefficients;
}

} /* namespace Serial */
} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

