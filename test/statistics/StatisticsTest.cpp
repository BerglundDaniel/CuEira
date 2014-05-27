#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <sstream>

#include <Statistics.h>
#include <HostVector.h>
#include <HostMatrix.h>

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
};

StatisticsTest::StatisticsTest() {

}

StatisticsTest::~StatisticsTest() {

}

void StatisticsTest::SetUp() {

}

void StatisticsTest::TearDown() {

}

TEST_F(StatisticsTest, Reri) {
  Statistics statistics();

  double reri = statistics.calculateReri();
}

TEST_F(StatisticsTest, Ap) {
  double reri = statistics.calculateReri();
  double ap = statistics.calculateAp();
}

TEST_F(StatisticsTest, OddsRatios) {
  std::vector<double> oddsRatios = statistics.calculateOddsRatios();
}

TEST_F(StatisticsTest, OddsRatiosLow) {
  std::vector<double> oddsRatiosLow = statistics.calculateOddsRatiosLow();
}

TEST_F(StatisticsTest, OddsRatiosHigh) {
  std::vector<double> oddsRatiosHigh = statistics.calculateOddsRatiosHigh();
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */
