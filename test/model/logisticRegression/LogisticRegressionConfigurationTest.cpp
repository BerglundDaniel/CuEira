#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sstream>

#include <HostVector.h>
#include <HostMatrix.h>
#include <RegularHostVector.h>
#include <RegularHostMatrix.h>
#include <LogisticRegressionConfiguration.h>
#include <ConfigurationMock.h>
#include <MKLWrapper.h>

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
class LogisticRegressionConfigurationTest: public ::testing::Test {
protected:
  LogisticRegressionConfigurationTest();
  virtual ~LogisticRegressionConfigurationTest();
  virtual void SetUp();
  virtual void TearDown();

  ConfigurationMock configMock;
  const double convergenceThreshold;
  const int numberOfMaxLRIterations;
  const int numberOfPredictors;
  const int numberOfRows;
  MKLWrapper blasWrapper;

  /**
   * Class to enable test of the LogisticRegressionConfiguration since it has a pure virtual function
   *
   * @author Daniel Berglund daniel.k.berglund@gmail.com
   */
  class LogisticRegressionConfigurationTestClass: public CuEira::Model::LogisticRegression::LogisticRegressionConfiguration {
    friend LogisticRegressionConfigurationTest;FRIEND_TEST(LogisticRegressionConfigurationTest, SetDefaultBeta);

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

LogisticRegressionConfigurationTest::LogisticRegressionConfigurationTestClass::LogisticRegressionConfigurationTestClass(
    const Configuration& configuration, const MKLWrapper& blasWrapper, bool usingCovariates, const int numberOfRows,
    const int numberOfPredictors) :
    LogisticRegressionConfiguration(configuration, blasWrapper, usingCovariates, numberOfRows, numberOfPredictors), vector(
        new RegularHostVector(1)) {

}

LogisticRegressionConfigurationTest::LogisticRegressionConfigurationTestClass::~LogisticRegressionConfigurationTestClass() {

}

void LogisticRegressionConfigurationTest::LogisticRegressionConfigurationTestClass::setEnvironmentFactor(
    const HostVector& environmentData) {

}

void LogisticRegressionConfigurationTest::LogisticRegressionConfigurationTestClass::setSNP(const HostVector& snpData) {

}

void LogisticRegressionConfigurationTest::LogisticRegressionConfigurationTestClass::setInteraction(
    const HostVector& interactionVector) {

}

HostVector& LogisticRegressionConfigurationTest::LogisticRegressionConfigurationTestClass::getScoresHost() {
  return *vector;
}

LogisticRegressionConfigurationTest::LogisticRegressionConfigurationTest() :
    convergenceThreshold(1e-3), numberOfMaxLRIterations(500), numberOfPredictors(4), numberOfRows(10) {

  EXPECT_CALL(configMock, getLRConvergenceThreshold()).Times(AtLeast(0)).WillRepeatedly(Return(convergenceThreshold));
  EXPECT_CALL(configMock, getNumberOfMaxLRIterations()).Times(AtLeast(0)).WillRepeatedly(
      Return(numberOfMaxLRIterations));
}

LogisticRegressionConfigurationTest::~LogisticRegressionConfigurationTest() {

}

void LogisticRegressionConfigurationTest::SetUp() {

}

void LogisticRegressionConfigurationTest::TearDown() {

}

TEST_F(LogisticRegressionConfigurationTest, Getters) {
  LogisticRegressionConfigurationTestClass logisticRegressionConfiguration(configMock, blasWrapper, false, numberOfRows,
      numberOfPredictors);

  EXPECT_EQ(convergenceThreshold, logisticRegressionConfiguration.getConvergenceThreshold());
  EXPECT_EQ(numberOfMaxLRIterations, logisticRegressionConfiguration.getNumberOfMaxIterations());
  EXPECT_EQ(numberOfPredictors, logisticRegressionConfiguration.getNumberOfPredictors());
  EXPECT_EQ(numberOfRows, logisticRegressionConfiguration.getNumberOfRows());
}

TEST_F(LogisticRegressionConfigurationTest, SetDefaultBeta) {
  LogisticRegressionConfigurationTestClass logisticRegressionConfiguration(configMock, blasWrapper, false, numberOfRows,
      numberOfPredictors);
  RegularHostVector beta(numberOfPredictors);

  logisticRegressionConfiguration.setDefaultBeta(beta);

  for(int i = 0; i < numberOfPredictors; ++i){
    EXPECT_EQ(0, beta(i));
  }
}

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira_Test */
} /* namespace CuEira */
