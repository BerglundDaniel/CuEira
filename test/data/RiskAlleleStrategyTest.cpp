#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#include <RiskAllele.h>
#include <AlleleStatisticsMock.h>

using testing::ReturnRef;

namespace CuEira {
namespace CuEira_Test {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class RiskAlleleStrategyTest: public ::testing::Test {
protected:
  RiskAlleleStrategyTest();
  virtual ~RiskAlleleStrategyTest();
  virtual void SetUp();
  virtual void TearDown();
};

RiskAlleleStrategyTest::RiskAlleleStrategyTest() {

}

RiskAlleleStrategyTest::~RiskAlleleStrategyTest() {

}

void RiskAlleleStrategyTest::SetUp() {

}

void RiskAlleleStrategyTest::TearDown() {

}

TEST_F(RiskAlleleStrategyTest, RiskAllele_Equal_Case_Control) {
  RiskAlleleStrategy riskAlleleStrategy;
  AlleleStatisticsMock alleleStatisticsMock;

  std::vector<double> alleleFreqs(6);
  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 1;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 1;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;

  EXPECT_CALL(alleleStatisticsMock, getAlleleFrequencies()).Times(1).WillRepeatedly(ReturnRef(alleleFreqs));

  RiskAllele riskAllele = riskAlleleStrategy.calculateRiskAllele(alleleStatisticsMock);
  EXPECT_EQ(ALLELE_ONE, riskAllele);
}

TEST_F(RiskAlleleStrategyTest, RiskAllele_Equal_Case_Control1_larger_Control2) {
  RiskAlleleStrategy riskAlleleStrategy;
  AlleleStatisticsMock alleleStatisticsMock;

  std::vector<double> alleleFreqs(6);
  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 1;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 2;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 1;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;

  EXPECT_CALL(alleleStatisticsMock, getAlleleFrequencies()).Times(1).WillRepeatedly(ReturnRef(alleleFreqs));

  RiskAllele riskAllele = riskAlleleStrategy.calculateRiskAllele(alleleStatisticsMock);
  EXPECT_EQ(ALLELE_TWO, riskAllele);
}

TEST_F(RiskAlleleStrategyTest, RiskAllele_Equal_Case_Control2_larger_Control1) {
  RiskAlleleStrategy riskAlleleStrategy;
  AlleleStatisticsMock alleleStatisticsMock;

  std::vector<double> alleleFreqs(6);
  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 1;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 2;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;

  EXPECT_CALL(alleleStatisticsMock, getAlleleFrequencies()).Times(1).WillRepeatedly(ReturnRef(alleleFreqs));

  RiskAllele riskAllele = riskAlleleStrategy.calculateRiskAllele(alleleStatisticsMock);
  EXPECT_EQ(ALLELE_ONE, riskAllele);
}

TEST_F(RiskAlleleStrategyTest, RiskAllele_Case1_Larger_Case2_Case1_Larger_Control1) {
  RiskAlleleStrategy riskAlleleStrategy;
  AlleleStatisticsMock alleleStatisticsMock;

  std::vector<double> alleleFreqs(6);
  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 2;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 1;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 1;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;

  EXPECT_CALL(alleleStatisticsMock, getAlleleFrequencies()).Times(1).WillRepeatedly(ReturnRef(alleleFreqs));

  RiskAllele riskAllele = riskAlleleStrategy.calculateRiskAllele(alleleStatisticsMock);
  EXPECT_EQ(ALLELE_ONE, riskAllele);
}

TEST_F(RiskAlleleStrategyTest, RiskAllele_Case1_Larger_Case2_Case1_Smaller_Control1) {
  RiskAlleleStrategy riskAlleleStrategy;
  AlleleStatisticsMock alleleStatisticsMock;

  std::vector<double> alleleFreqs(6);
  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 2;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 1;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 3;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 1;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;

  EXPECT_CALL(alleleStatisticsMock, getAlleleFrequencies()).Times(1).WillRepeatedly(ReturnRef(alleleFreqs));

  RiskAllele riskAllele = riskAlleleStrategy.calculateRiskAllele(alleleStatisticsMock);
  EXPECT_EQ(ALLELE_TWO, riskAllele);
}

TEST_F(RiskAlleleStrategyTest, RiskAllele_Case2_Larger_Case1_Case2_Larger_Control2) {
  RiskAlleleStrategy riskAlleleStrategy;
  AlleleStatisticsMock alleleStatisticsMock;

  std::vector<double> alleleFreqs(6);
  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 2;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 1;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;

  EXPECT_CALL(alleleStatisticsMock, getAlleleFrequencies()).Times(1).WillRepeatedly(ReturnRef(alleleFreqs));

  RiskAllele riskAllele = riskAlleleStrategy.calculateRiskAllele(alleleStatisticsMock);
  EXPECT_EQ(ALLELE_TWO, riskAllele);
}

TEST_F(RiskAlleleStrategyTest, RiskAllele_Case2_Larger_Case1_Case2_Smaller_Control2) {
  RiskAlleleStrategy riskAlleleStrategy;
  AlleleStatisticsMock alleleStatisticsMock;

  std::vector<double> alleleFreqs(6);
  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 2;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 3;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;

  EXPECT_CALL(alleleStatisticsMock, getAlleleFrequencies()).Times(1).WillRepeatedly(ReturnRef(alleleFreqs));

  RiskAllele riskAllele = riskAlleleStrategy.calculateRiskAllele(alleleStatisticsMock);
  EXPECT_EQ(ALLELE_ONE, riskAllele);
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */

