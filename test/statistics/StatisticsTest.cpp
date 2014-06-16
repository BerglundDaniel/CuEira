#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <sstream>
#include <math.h>

#include <Statistics.h>
#include <HostVector.h>
#include <HostMatrix.h>

#ifdef CPU
#include <lapackpp/lavd.h>
#include <LapackppHostVector.h>
#else
#include <PinnedHostVector.h>
#endif

using ::testing::Ge;
using ::testing::Le;

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
  Container::HostVector* beta;
  Container::HostVector* standardError;

#ifdef CPU
  Container::LapackppHostVector oddsRatios;
  Container::LapackppHostVector oddsRatiosLow;
  Container::LapackppHostVector oddsRatiosHigh;
#else
  Container::PinnedHostVector oddsRatios;
  Container::PinnedHostVector oddsRatiosLow;
  Container::PinnedHostVector oddsRatiosHigh;
#endif

};

StatisticsTest::StatisticsTest() :
    numberOfPredictors(4),
#ifdef CPU
        beta(new Container::LapackppHostVector(new LaVectorDouble(numberOfPredictors))), oddsRatios(new LaVectorDouble(numberOfPredictors - 1)),
        standardError(new Container::LapackppHostVector(new LaVectorDouble(numberOfPredictors))),oddsRatiosLow(new LaVectorDouble(numberOfPredictors - 1)),
        oddsRatiosHigh(new LaVectorDouble(numberOfPredictors - 1))
#else
        beta(new Container::PinnedHostVector(numberOfPredictors)), oddsRatios(numberOfPredictors - 1), standardError(
            new Container::PinnedHostVector(numberOfPredictors)), oddsRatiosLow(numberOfPredictors - 1), oddsRatiosHigh(
            numberOfPredictors - 1)
#endif

{

}

StatisticsTest::~StatisticsTest() {
 //Don't need to delete beta and standardError since Statistics class will do that.
}

void StatisticsTest::SetUp() {
#ifdef CPU
  beta = new Container::LapackppHostVector(new LaVectorDouble(numberOfPredictors));
  standardError = new Container::LapackppHostVector(new LaVectorDouble(numberOfPredictors));
#else
  beta = new Container::PinnedHostVector(numberOfPredictors);
  standardError = new Container::PinnedHostVector(numberOfPredictors);
#endif

  for(int i = 0; i < numberOfPredictors; ++i){
    (*beta)(i) = (i + 7) / 10.3;
  }

  for(int i = 0; i < numberOfPredictors; ++i){
    (*standardError)(i) = (i + 3) / 3.1;
  }

  for(int i = 0; i < numberOfPredictors - 1; ++i){
    oddsRatios(i) = exp((*beta)(i + 1));

    oddsRatiosLow(i) = exp(-1.96 * (*standardError)(i + 1) + (*beta)(i + 1));
    oddsRatiosHigh(i) = exp(1.96 * (*standardError)(i + 1) + (*beta)(i + 1));
  }

}

void StatisticsTest::TearDown() {

}

TEST_F(StatisticsTest, Reri) {
  Statistics statistics(beta, standardError);
  double e = 10e-5;

  double reri = oddsRatios(2) - oddsRatios(1) - oddsRatios(0) + 1;
  double l = reri - e;
  double h = reri + e;

  double statReri = statistics.getReri();
  EXPECT_THAT(statReri, Ge(l));
  EXPECT_THAT(statReri, Le(h));
}

TEST_F(StatisticsTest, Ap) {
  Statistics statistics(beta, standardError);
  double e = 10e-5;

  double reri = statistics.getReri();
  double or11 = (*beta)(3);

  double l = reri / or11 - e;
  double h = reri / or11 + e;

  double statAp = statistics.getAp();
  EXPECT_THAT(statAp, Ge(l));
  EXPECT_THAT(statAp, Le(h));
}

TEST_F(StatisticsTest, OddsRatios) {
  Statistics statistics(beta, standardError);

  std::vector<double> oddsRatiosStat = statistics.getOddsRatios();

  double e = 10e-5;
  for(int i = 0; i < numberOfPredictors - 1; ++i){
    double l = oddsRatios(i) - e;
    double h = oddsRatios(i) + e;

    EXPECT_THAT(oddsRatiosStat[i], Ge(l));
    EXPECT_THAT(oddsRatiosStat[i], Le(h));
  }
}

TEST_F(StatisticsTest, OddsRatiosLowAndHigh) {
  Statistics statistics(beta, standardError);

  std::vector<double> oddsRatiosLowStat = statistics.getOddsRatiosLow();
  std::vector<double> oddsRatiosHighStat = statistics.getOddsRatiosHigh();

  double e = 10e-5;
  for(int i = 0; i < numberOfPredictors - 1; ++i){
    EXPECT_THAT(oddsRatiosLowStat[i], Le(oddsRatiosHighStat[i]));

    double l_low = oddsRatiosLow(i) - e;
    double h_low = oddsRatiosLow(i) + e;

    double l_high = oddsRatiosHigh(i) - e;
    double h_high = oddsRatiosHigh(i) + e;

    EXPECT_THAT(oddsRatiosLowStat[i], Ge(l_low));
    EXPECT_THAT(oddsRatiosLowStat[i], Le(h_low));

    EXPECT_THAT(oddsRatiosHighStat[i], Ge(l_high));
    EXPECT_THAT(oddsRatiosHighStat[i], Le(h_high));
  }
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */
