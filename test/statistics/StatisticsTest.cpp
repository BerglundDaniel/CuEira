#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <sstream>
#include <math.h>

#include <Statistics.h>
#include <HostVector.h>
#include <HostMatrix.h>
#include <LogisticRegressionResultMock.h>

#ifdef CPU
#include <lapackpp/lavd.h>
#include <LapackppHostVector.h>
#else
#include <PinnedHostVector.h>
#include <PinnedHostMatrix.h>
#endif

using ::testing::Ge;
using ::testing::Le;
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
class StatisticsTest: public ::testing::Test {
protected:
  StatisticsTest();
  virtual ~StatisticsTest();
  virtual void SetUp();
  virtual void TearDown();

  const int numberOfPredictors;
  Model::LogisticRegression::LogisticRegressionResultMock* logisticRegressionResultMock;

  Container::HostMatrix* inverseInfoMat;
  Container::HostVector* beta;

  std::vector<double> oddsRatios;
  std::vector<double> oddsRatiosLow;
  std::vector<double> oddsRatiosHigh;

};

StatisticsTest::StatisticsTest() :
    numberOfPredictors(4), logisticRegressionResultMock(nullptr), oddsRatios(numberOfPredictors), oddsRatiosLow(
        numberOfPredictors), oddsRatiosHigh(numberOfPredictors),
#ifdef CPU
        beta(new Container::LapackppHostVector(new LaVectorDouble(numberOfPredictors))), inverseInfoMat(
            new Container::LapackppHostMatrix(new LaGenMatDouble(numberOfPredictors, numberOfPredictors)))
#else
        beta(new Container::PinnedHostVector(numberOfPredictors)), inverseInfoMat(
            new Container::PinnedHostMatrix(numberOfPredictors, numberOfPredictors))
#endif
{
  for(int i = 0; i < numberOfPredictors; ++i){
    (*beta)(i) = (i + 7) / 10.3;
  }

  //Only care about the diagonal of the information matrix inverse
  for(int i = 0; i < numberOfPredictors; ++i){
    (*inverseInfoMat)(i, i) = i;
  }

  for(int i = 0; i < numberOfPredictors - 1; ++i){
    oddsRatios[i] = exp((*beta)(i + 1));

    oddsRatiosLow[i] = exp(-1.96 * (*inverseInfoMat)(i + 1, i + 1) + (*beta)(i + 1));
    oddsRatiosHigh[i] = exp(1.96 * (*inverseInfoMat)(i + 1, i + 1) + (*beta)(i + 1));
  }
}

StatisticsTest::~StatisticsTest() {
  //Don't need to LogisticRegressionResult since Statistics class will do that.
  delete inverseInfoMat;
  delete beta;
}

void StatisticsTest::SetUp() {
  logisticRegressionResultMock = new Model::LogisticRegression::LogisticRegressionResultMock();

  EXPECT_CALL(*logisticRegressionResultMock, getBeta()).Times(1).WillRepeatedly(ReturnRef(*beta));
  EXPECT_CALL(*logisticRegressionResultMock, getInverseInformationMatrix()).Times(1).WillRepeatedly(
      ReturnRef(*inverseInfoMat));
}

void StatisticsTest::TearDown() {

}

TEST_F(StatisticsTest, Reri) {
  Statistics statistics(logisticRegressionResultMock);
  double e = 1e-5;

  double reri = oddsRatios[2] - oddsRatios[1] - oddsRatios[0] + 1;
  double l = reri - e;
  double h = reri + e;

  double statReri = statistics.getReri();
  EXPECT_THAT(statReri, Ge(l));
  EXPECT_THAT(statReri, Le(h));
}

TEST_F(StatisticsTest, Ap) {
  Statistics statistics(logisticRegressionResultMock);
  double e = 1e-5;

  double reri = statistics.getReri();
  double or11 = (*beta)(3);

  double l = reri / or11 - e;
  double h = reri / or11 + e;

  double statAp = statistics.getAp();
  EXPECT_THAT(statAp, Ge(l));
  EXPECT_THAT(statAp, Le(h));
}

TEST_F(StatisticsTest, OddsRatios) {
  Statistics statistics(logisticRegressionResultMock);

  std::vector<double> oddsRatiosStat = statistics.getOddsRatios();

  double e = 1e-5;
  for(int i = 0; i < numberOfPredictors - 1; ++i){
    double l = oddsRatios[i] - e;
    double h = oddsRatios[i] + e;

    EXPECT_THAT(oddsRatiosStat[i], Ge(l));
    EXPECT_THAT(oddsRatiosStat[i], Le(h));
  }
}

TEST_F(StatisticsTest, OddsRatiosLowAndHigh) {
  Statistics statistics(logisticRegressionResultMock);

  std::vector<double> oddsRatiosLowStat = statistics.getOddsRatiosLow();
  std::vector<double> oddsRatiosHighStat = statistics.getOddsRatiosHigh();

  double e = 1e-5;
  for(int i = 0; i < numberOfPredictors - 1; ++i){
    EXPECT_THAT(oddsRatiosLowStat[i], Le(oddsRatiosHighStat[i]));

    double l_low = oddsRatiosLow[i] - e;
    double h_low = oddsRatiosLow[i] + e;

    double l_high = oddsRatiosHigh[i] - e;
    double h_high = oddsRatiosHigh[i] + e;

    EXPECT_THAT(oddsRatiosLowStat[i], Ge(l_low));
    EXPECT_THAT(oddsRatiosLowStat[i], Le(h_low));

    EXPECT_THAT(oddsRatiosHighStat[i], Ge(l_high));
    EXPECT_THAT(oddsRatiosHighStat[i], Le(h_high));
  }
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */
