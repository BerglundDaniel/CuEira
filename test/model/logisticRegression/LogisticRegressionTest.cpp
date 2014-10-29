#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sstream>

#include <HostVector.h>
#include <HostMatrix.h>
#include <RegularHostVector.h>
#include <RegularHostMatrix.h>
#include <LogisticRegressionConfiguration.h>
#include <Configuration.h>
#include <ConfigurationMock.h>
#include <MKLWrapper.h>
#include <LogisticRegression.h>
#include <LogisticRegressionResult.h>

using testing::Return;
using testing::_;
using testing::AtLeast;
using ::testing::Ge;
using ::testing::Le;

namespace CuEira {
namespace CuEira_Test {
namespace Model {
namespace LogisticRegression {

using namespace CuEira::Container;
using namespace CuEira::Model;
using namespace CuEira::Model::LogisticRegression;

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

  ConfigurationMock configMock;
  const double convergenceThreshold;
  const int numberOfMaxLRIterations;
  MKLWrapper blasWrapper;

  /**
   * Class to enable test of the LogisticRegression since it has a pure virtual function
   *
   * @author Daniel Berglund daniel.k.berglund@gmail.com
   */
  class LogisticRegressionTestClass: public CuEira::Model::LogisticRegression::LogisticRegression {
    friend LogisticRegressionTest;FRIEND_TEST(LogisticRegressionTest, invertInformationMatrix);FRIEND_TEST(LogisticRegressionTest, calculateNewBeta);FRIEND_TEST(LogisticRegressionTest, calculateDifference);

    LogisticRegressionTestClass(LogisticRegressionConfiguration* logisticRegressionConfiguration);
    virtual ~LogisticRegressionTestClass();

    virtual LogisticRegressionResult* calculate();
  };

  /**
   * Class to enable test of the LogisticRegressionConfiguration since it has a pure virtual function
   *
   * @author Daniel Berglund daniel.k.berglund@gmail.com
   */
  class LogisticRegressionConfigurationTestClass: public CuEira::Model::LogisticRegression::LogisticRegressionConfiguration {
  public:
    LogisticRegressionConfigurationTestClass(const Configuration& configuration, const MKLWrapper& blasWrapper,
        bool usingCovariates, const int numberOfRows, const int numberOfPredictors);
    virtual ~LogisticRegressionConfigurationTestClass();

    virtual void setEnvironmentFactor(const HostVector& environmentData);
    virtual void setSNP(const HostVector& snpData);
    virtual void setInteraction(const HostVector& interactionVector);
    virtual HostVector& getScoresHost();

    RegularHostVector* vector;
  };

};

LogisticRegressionTest::LogisticRegressionTestClass::LogisticRegressionTestClass(
    LogisticRegressionConfiguration* logisticRegressionConfiguration) :
    LogisticRegression(logisticRegressionConfiguration) {

}

LogisticRegressionTest::LogisticRegressionTestClass::~LogisticRegressionTestClass() {

}

LogisticRegressionResult* LogisticRegressionTest::LogisticRegressionTestClass::calculate() {
  return nullptr;
}

LogisticRegressionTest::LogisticRegressionConfigurationTestClass::LogisticRegressionConfigurationTestClass(
    const Configuration& configuration, const MKLWrapper& blasWrapper, bool usingCovariates, const int numberOfRows,
    const int numberOfPredictors) :
    LogisticRegressionConfiguration(configuration, blasWrapper, usingCovariates, numberOfRows, numberOfPredictors), vector(
        new RegularHostVector(1)) {

}

LogisticRegressionTest::LogisticRegressionConfigurationTestClass::~LogisticRegressionConfigurationTestClass() {

}

void LogisticRegressionTest::LogisticRegressionConfigurationTestClass::setEnvironmentFactor(
    const HostVector& environmentData) {

}

void LogisticRegressionTest::LogisticRegressionConfigurationTestClass::setSNP(const HostVector& snpData) {

}

void LogisticRegressionTest::LogisticRegressionConfigurationTestClass::setInteraction(
    const HostVector& interactionVector) {

}

HostVector& LogisticRegressionTest::LogisticRegressionConfigurationTestClass::getScoresHost() {
  return *vector;
}

LogisticRegressionTest::LogisticRegressionTest() :
    convergenceThreshold(1e-3), numberOfMaxLRIterations(500) {

  EXPECT_CALL(configMock, getLRConvergenceThreshold()).Times(AtLeast(0)).WillRepeatedly(Return(convergenceThreshold));
  EXPECT_CALL(configMock, getNumberOfMaxLRIterations()).Times(AtLeast(0)).WillRepeatedly(
      Return(numberOfMaxLRIterations));
}

LogisticRegressionTest::~LogisticRegressionTest() {

}

void LogisticRegressionTest::SetUp() {

}

void LogisticRegressionTest::TearDown() {

}

TEST_F(LogisticRegressionTest, invertInformationMatrix) {
  const int numberOfPredictors = 3;
  const int numberOfRows = 3;

  PRECISION e = 1e-4;
  PRECISION x, h, l;

  LogisticRegressionConfigurationTestClass* lrConfig = new LogisticRegressionConfigurationTestClass(configMock,
      blasWrapper, false, numberOfRows, numberOfPredictors);

  LogisticRegressionTestClass logisticRegression(lrConfig);

  RegularHostMatrix informationMatrixHost(numberOfPredictors, numberOfPredictors);
  RegularHostMatrix inverseInformationMatrixHost(numberOfPredictors, numberOfPredictors);
  RegularHostMatrix uSVD(numberOfPredictors, numberOfPredictors);
  RegularHostVector sigma(numberOfPredictors);
  RegularHostMatrix vtSVD(numberOfPredictors, numberOfPredictors);
  RegularHostMatrix workMatrixMxMHost(numberOfPredictors, numberOfPredictors);

  informationMatrixHost(0, 0) = 1;
  informationMatrixHost(0, 1) = 2;
  informationMatrixHost(0, 2) = 3;

  informationMatrixHost(1, 0) = 4;
  informationMatrixHost(1, 1) = 5;
  informationMatrixHost(1, 2) = 6;

  informationMatrixHost(2, 0) = 7;
  informationMatrixHost(2, 1) = 8;
  informationMatrixHost(2, 2) = 9;

  logisticRegression.invertInformationMatrix(informationMatrixHost, inverseInformationMatrixHost, uSVD, sigma, vtSVD,
      workMatrixMxMHost);

  /*
   -0.6389   -0.1667    0.3056
   -0.0556    0.0000    0.0556
   0.5278    0.1667   -0.1944
   */

  x = -0.6389;
  l = x - e;
  h = x + e;
  EXPECT_THAT(inverseInformationMatrixHost(0, 0), Ge(l));
  EXPECT_THAT(inverseInformationMatrixHost(0, 0), Le(h));

  x = -0.1667;
  l = x - e;
  h = x + e;
  EXPECT_THAT(inverseInformationMatrixHost(0, 1), Ge(l));
  EXPECT_THAT(inverseInformationMatrixHost(0, 1), Le(h));

  x = 0.3056;
  l = x - e;
  h = x + e;
  EXPECT_THAT(inverseInformationMatrixHost(0, 2), Ge(l));
  EXPECT_THAT(inverseInformationMatrixHost(0, 2), Le(h));

  x = -0.0556;
  l = x - e;
  h = x + e;
  EXPECT_THAT(inverseInformationMatrixHost(1, 0), Ge(l));
  EXPECT_THAT(inverseInformationMatrixHost(1, 0), Le(h));

  x = 0.0;
  l = x - e;
  h = x + e;
  EXPECT_THAT(inverseInformationMatrixHost(1, 1), Ge(l));
  EXPECT_THAT(inverseInformationMatrixHost(1, 1), Le(h));

  x = 0.0556;
  l = x - e;
  h = x + e;
  EXPECT_THAT(inverseInformationMatrixHost(1, 2), Ge(l));
  EXPECT_THAT(inverseInformationMatrixHost(1, 2), Le(h));

  x = 0.5278;
  l = x - e;
  h = x + e;
  EXPECT_THAT(inverseInformationMatrixHost(2, 0), Ge(l));
  EXPECT_THAT(inverseInformationMatrixHost(2, 0), Le(h));

  x = 0.1667;
  l = x - e;
  h = x + e;
  EXPECT_THAT(inverseInformationMatrixHost(2, 1), Ge(l));
  EXPECT_THAT(inverseInformationMatrixHost(2, 1), Le(h));

  x = -0.1944;
  l = x - e;
  h = x + e;
  EXPECT_THAT(inverseInformationMatrixHost(2, 2), Ge(l));
  EXPECT_THAT(inverseInformationMatrixHost(2, 2), Le(h));
}

TEST_F(LogisticRegressionTest, calculateNewBeta) {
  const int numberOfPredictors = 3;
  const int numberOfRows = 3;

  PRECISION e = 1e-5;
  PRECISION x, h, l;

  LogisticRegressionConfigurationTestClass* lrConfig = new LogisticRegressionConfigurationTestClass(configMock,
      blasWrapper, false, numberOfRows, numberOfPredictors);

  LogisticRegressionTestClass logisticRegression(lrConfig);

  RegularHostMatrix inverseInformationMatrixHost(numberOfPredictors, numberOfPredictors);
  RegularHostVector scoresHost(numberOfPredictors);
  RegularHostVector betaCoefficentsHost(numberOfPredictors);

  inverseInformationMatrixHost(0, 0) = 1;
  inverseInformationMatrixHost(0, 1) = 2;
  inverseInformationMatrixHost(0, 2) = 3;

  inverseInformationMatrixHost(1, 0) = 4;
  inverseInformationMatrixHost(1, 1) = 5;
  inverseInformationMatrixHost(1, 2) = 6;

  inverseInformationMatrixHost(2, 0) = 7;
  inverseInformationMatrixHost(2, 1) = 8;
  inverseInformationMatrixHost(2, 2) = 9;

  scoresHost(0) = 0.1;
  scoresHost(1) = 0.2;
  scoresHost(2) = 0.3;

  betaCoefficentsHost(0) = 10;
  betaCoefficentsHost(1) = 100;
  betaCoefficentsHost(2) = 200;

  logisticRegression.calculateNewBeta(inverseInformationMatrixHost, scoresHost, betaCoefficentsHost);

  x = 11.4;
  l = x - e;
  h = x + e;
  EXPECT_THAT(betaCoefficentsHost(0), Ge(l));
  EXPECT_THAT(betaCoefficentsHost(0), Le(h));

  x = 103.2;
  l = x - e;
  h = x + e;
  EXPECT_THAT(betaCoefficentsHost(1), Ge(l));
  EXPECT_THAT(betaCoefficentsHost(1), Le(h));

  x = 205;
  l = x - e;
  h = x + e;
  EXPECT_THAT(betaCoefficentsHost(2), Ge(l));
  EXPECT_THAT(betaCoefficentsHost(2), Le(h));
}

TEST_F(LogisticRegressionTest, calculateDifference) {
  const int numberOfPredictors = 3;
  const int numberOfRows = 3;

  PRECISION e = 1e-5;
  PRECISION x, h, l;

  LogisticRegressionConfigurationTestClass* lrConfig = new LogisticRegressionConfigurationTestClass(configMock,
      blasWrapper, false, numberOfRows, numberOfPredictors);

  LogisticRegressionTestClass logisticRegression(lrConfig);

  RegularHostVector betaCoefficentsHost(numberOfPredictors);
  RegularHostVector betaCoefficentsOldHost(numberOfPredictors);

  PRECISION diffSumHost = 0;

  betaCoefficentsHost(0) = 1;
  betaCoefficentsHost(1) = 2;
  betaCoefficentsHost(2) = 3;

  betaCoefficentsOldHost(0) = -4;
  betaCoefficentsOldHost(1) = 1.2;
  betaCoefficentsOldHost(2) = 5.1;

  PRECISION sum = 0;
  PRECISION diff;
  for(int i = 0; i < numberOfPredictors; ++i){
    diff = betaCoefficentsHost(i) - betaCoefficentsOldHost(i);
    sum += fabs(diff);
  }

  logisticRegression.calculateDifference(betaCoefficentsHost, betaCoefficentsOldHost, diffSumHost);

  x = sum;
  l = x - e;
  h = x + e;
  EXPECT_THAT(diffSumHost, Ge(l));
  EXPECT_THAT(diffSumHost, Le(h));
}

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira_Test */
} /* namespace CuEira */
