#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <sstream>
#include <ostream>

#include <AlleleStatistics.h>

namespace CuEira {

/**
 * Test for testing AlleleStatistics
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class AlleleStatisticsTest: public ::testing::Test {
protected:
  AlleleStatisticsTest();
  virtual ~AlleleStatisticsTest();
  virtual void SetUp();
  virtual void TearDown();

};

AlleleStatisticsTest::AlleleStatisticsTest() {

}

AlleleStatisticsTest::~AlleleStatisticsTest() {

}

void AlleleStatisticsTest::SetUp() {

}

void AlleleStatisticsTest::TearDown() {

}

TEST_F(AlleleStatisticsTest, Getters) {
  std::vector<int>* numberOfAlleles = new std::vector<int>(6);
  std::vector<double>* alleleFrequencies = new std::vector<double>(6);

  for(int i = 0; i < 6; ++i){
    (*numberOfAlleles)[i] = i;
    (*alleleFrequencies)[i] = i / 10;
  }

  AlleleStatistics alleleStatistics(numberOfAlleles, alleleFrequencies);

  ASSERT_EQ(numberOfAlleles, &alleleStatistics.getAlleleNumbers());
  ASSERT_EQ(alleleFrequencies, &alleleStatistics.getAlleleFrequencies());

  std::ostringstream resultOs;
  resultOs << (*numberOfAlleles)[ALLELE_ONE_CASE_POSITION] << "," << (*numberOfAlleles)[ALLELE_TWO_CASE_POSITION] << ","
      << (*numberOfAlleles)[ALLELE_ONE_CONTROL_POSITION] << "," << (*numberOfAlleles)[ALLELE_TWO_CONTROL_POSITION]
      << "," << (*numberOfAlleles)[ALLELE_ONE_ALL_POSITION] << "," << (*numberOfAlleles)[ALLELE_TWO_ALL_POSITION]
      << ",";

  resultOs << (*alleleFrequencies)[ALLELE_ONE_CASE_POSITION] << "," << (*alleleFrequencies)[ALLELE_TWO_CASE_POSITION]
      << "," << (*alleleFrequencies)[ALLELE_ONE_CONTROL_POSITION] << ","
      << (*alleleFrequencies)[ALLELE_TWO_CONTROL_POSITION] << "," << (*alleleFrequencies)[ALLELE_ONE_ALL_POSITION]
      << "," << (*alleleFrequencies)[ALLELE_TWO_ALL_POSITION];

  std::ostringstream osAlleleStatistics;
  osAlleleStatistics << alleleStatistics;

  EXPECT_EQ(resultOs.str(), osAlleleStatistics.str());
}

//TODO MAF

} /* namespace CuEira */
