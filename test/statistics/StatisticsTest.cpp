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

using ::testing::AllOf;
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

#ifdef CPU
  Container::LapackppHostVector beta;
  Container::LapackppHostVector standardError;
  Container::LapackppHostVector oddsRatios;
  Container::LapackppHostVector oddsRatiosLow;
  Container::LapackppHostVector oddsRatiosHigh;
#else
  Container::PinnedHostVector beta;
  Container::PinnedHostVector standardError;
  Container::PinnedHostVector oddsRatios;
  Container::PinnedHostVector oddsRatiosLow;
  Container::PinnedHostVector oddsRatiosHigh;
#endif

};

StatisticsTest::StatisticsTest() :
    numberOfPredictors(4),
#ifdef CPU
        beta(new LaVectorDouble(numberOfPredictors)), oddsRatios(new LaVectorDouble(numberOfPredictors - 1)),
        standardError(new LaVectorDouble(numberOfPredictors)),oddsRatiosLow(new LaVectorDouble(numberOfPredictors - 1)),
        oddsRatiosHigh(new LaVectorDouble(numberOfPredictors - 1))
#else
        beta(numberOfPredictors), oddsRatios(numberOfPredictors - 1), standardError(numberOfPredictors), oddsRatiosLow(
            numberOfPredictors - 1), oddsRatiosHigh(numberOfPredictors - 1)
#endif

{

}

StatisticsTest::~StatisticsTest() {

}

void StatisticsTest::SetUp() {
  for(int i = 0; i < numberOfPredictors; ++i){
    beta(i) = (i + 7) / 10;
  }

  for(int i = 0; i < numberOfPredictors; ++i){
    standardError(i) = (i + 3) / 3;
  }

  for(int i = 0; i < numberOfPredictors - 1; ++i){
    oddsRatios(i) = exp(beta(i + 1));

    oddsRatiosLow(i) = exp(-1.96 * standardError(i + 1) + beta(i + 1));
    oddsRatiosHigh(i) = exp(1.96 * standardError(i + 1) + beta(i + 1));
  }

}

void StatisticsTest::TearDown() {

}

TEST_F(StatisticsTest, Reri) {
  Statistics statistics(beta, standardError);

  double reri = oddsRatios(2) - oddsRatios(1) - oddsRatios(0) + 1;
  ASSERT_EQ(reri, statistics.getReri());
}

TEST_F(StatisticsTest, Ap) {
  Statistics statistics(beta, standardError);

  double reri = statistics.getReri();
  double or11 = 1;
  ASSERT_EQ(reri / or11, statistics.getAp());
}

TEST_F(StatisticsTest, OddsRatios) {
  Statistics statistics(beta, standardError);

  std::vector<double> oddsRatiosStat = statistics.getOddsRatios();

  for(int i = 0; i < numberOfPredictors - 1; ++i){
    double e = 10e-5;
    double l = oddsRatios(i) - e;
    double h = oddsRatios(i) + e;

    EXPECT_THAT(oddsRatiosStat[i], AllOf(Ge(l), Le(h)));
  }
}

TEST_F(StatisticsTest, OddsRatiosLowAndHigh) {
  Statistics statistics(beta, standardError);

  std::vector<double> oddsRatiosLowStat = statistics.getOddsRatiosLow();
  std::vector<double> oddsRatiosHighStat = statistics.getOddsRatiosHigh();

  for(int i = 0; i < numberOfPredictors - 1; ++i){
    EXPECT_THAT(oddsRatiosLowStat[i], Le(oddsRatiosHighStat[i]));

    EXPECT_EQ(oddsRatiosLow(i), oddsRatiosLowStat[i]);
    EXPECT_EQ(oddsRatiosHigh(i), oddsRatiosHighStat[i]);
  }
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */
