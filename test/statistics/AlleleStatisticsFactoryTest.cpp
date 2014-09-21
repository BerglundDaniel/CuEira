#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <sstream>
#include <ostream>

#include <AlleleStatistics.h>
#include <AlleleStatisticsFactory.h>

namespace CuEira {

/**
 * Test for testing AlleleStatistics
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class AlleleStatisticsFactoryTest: public ::testing::Test {
protected:
  AlleleStatisticsFactoryTest();
  virtual ~AlleleStatisticsFactoryTest();
  virtual void SetUp();
  virtual void TearDown();

};

AlleleStatisticsFactoryTest::AlleleStatisticsFactoryTest() {

}

AlleleStatisticsFactoryTest::~AlleleStatisticsFactoryTest() {

}

void AlleleStatisticsFactoryTest::SetUp() {

}

void AlleleStatisticsFactoryTest::TearDown() {

}

TEST_F(AlleleStatisticsFactoryTest, Construct) {
  AlleleStatisticsFactory alleleStatisticsFactory;
  std::vector<int>* numberOfAlleles = new std::vector<int>(6);

  for(int i = 0; i < 6; ++i){
    (*numberOfAlleles)[i] = i;
  }

  AlleleStatistics* alleleStatistics = alleleStatisticsFactory.constructAlleleStatistics(numberOfAlleles);

  ASSERT_EQ(numberOfAlleles, &alleleStatistics->getAlleleNumbers());
  const std::vector<double>& alleleFrequencies = alleleStatistics->getAlleleFrequencies();

  EXPECT_EQ(0, alleleFrequencies[ALLELE_ONE_CASE_POSITION]);
  EXPECT_EQ(1, alleleFrequencies[ALLELE_TWO_CASE_POSITION]);

  EXPECT_EQ((double )2 / 5, alleleFrequencies[ALLELE_ONE_CONTROL_POSITION]);
  EXPECT_EQ((double )3 / 5, alleleFrequencies[ALLELE_TWO_CONTROL_POSITION]);

  EXPECT_EQ((double )4 / 9, alleleFrequencies[ALLELE_ONE_ALL_POSITION]);
  EXPECT_EQ((double )5 / 9, alleleFrequencies[ALLELE_TWO_ALL_POSITION]);

  delete alleleStatistics;
}

} /* namespace CuEira */
