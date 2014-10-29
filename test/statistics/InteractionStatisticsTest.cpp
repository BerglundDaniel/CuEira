#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <sstream>
#include <math.h>

#include <InteractionStatistics.h>
#include <HostVector.h>
#include <HostMatrix.h>
#include <LogisticRegressionResultMock.h>
#include <RegularHostVector.h>
#include <RegularHostMatrix.h>

using testing::Ge;
using testing::Le;
using testing::Return;
using testing::_;
using testing::ReturnRef;

namespace CuEira {
namespace CuEira_Test {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class InteractionStatisticsTest: public ::testing::Test {
protected:
  InteractionStatisticsTest();
  virtual ~InteractionStatisticsTest();
  virtual void SetUp();
  virtual void TearDown();

  const int numberOfPredictors;
  Model::LogisticRegression::LogisticRegressionResultMock* logisticRegressionResultMock;

  Container::HostMatrix* inverseInfoMat;
  Container::HostVector* beta;

  std::vector<double> oddsRatios;
};

InteractionStatisticsTest::InteractionStatisticsTest() :
    numberOfPredictors(4), logisticRegressionResultMock(nullptr), oddsRatios(numberOfPredictors), beta(
        new Container::RegularHostVector(numberOfPredictors)), inverseInfoMat(
        new Container::RegularHostMatrix(numberOfPredictors, numberOfPredictors)) {

  for(int i = 0; i < numberOfPredictors; ++i){
    (*beta)(i) = (i + 7) / 10.3;
  }

  //Only care about the diagonal of the information matrix inverse
  for(int i = 0; i < numberOfPredictors; ++i){
    (*inverseInfoMat)(i, i) = i;
  }

  for(int i = 0; i < numberOfPredictors - 1; ++i){
    oddsRatios[i] = exp((*beta)(i + 1));
  }

}

InteractionStatisticsTest::~InteractionStatisticsTest() {
  //Don't need to LogisticRegressionResult since the class will do that.
  delete inverseInfoMat;
  delete beta;
}

void InteractionStatisticsTest::SetUp() {
  logisticRegressionResultMock = new Model::LogisticRegression::LogisticRegressionResultMock();

  EXPECT_CALL(*logisticRegressionResultMock, getBeta()).Times(1).WillRepeatedly(ReturnRef(*beta));
  EXPECT_CALL(*logisticRegressionResultMock, getInverseInformationMatrix()).Times(1).WillRepeatedly(
      ReturnRef(*inverseInfoMat));
}

void InteractionStatisticsTest::TearDown() {

}

TEST_F(InteractionStatisticsTest, Reri) {
  InteractionStatistics statistics(logisticRegressionResultMock);
  double e = 1e-5;

  double reri = oddsRatios[2] - oddsRatios[1] - oddsRatios[0] + 1;
  double l = reri - e;
  double h = reri + e;

  double statReri = statistics.getReri();
  EXPECT_THAT(statReri, Ge(l));
  EXPECT_THAT(statReri, Le(h));
}

TEST_F(InteractionStatisticsTest, Ap) {
  InteractionStatistics statistics(logisticRegressionResultMock);
  double e = 1e-5;

  double reri = statistics.getReri();
  double or11 = (*beta)(3);

  double l = reri / or11 - e;
  double h = reri / or11 + e;

  double statAp = statistics.getAp();
  EXPECT_THAT(statAp, Ge(l));
  EXPECT_THAT(statAp, Le(h));
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */
