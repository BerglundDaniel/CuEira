#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>

#include <SNP.h>
#include <Id.h>
#include <Recode.h>
#include <SNPVector.h>
#include <InvalidState.h>
#include <GeneticModel.h>
#include <RiskAllele.h>
#include <StatisticModel.h>
#include <SNPVectorFactory.h>
#include <ConfigurationMock.h>

using testing::Return;
using testing::_;
using testing::Eq;
using testing::Ge;
using testing::Le;

namespace CuEira {
namespace Container {

/**
 * Test for testing SNPVectorFactory
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class SNPVectorFactoryTest: public ::testing::Test {
protected:
  SNPVectorFactoryTest();
  virtual ~SNPVectorFactoryTest();
  virtual void SetUp();
  virtual void TearDown();

  const int numberOfIndividuals;
  ConfigurationMock configMock;
};

SNPVectorFactoryTest::SNPVectorFactoryTest() :
    numberOfIndividuals(10) {

}

SNPVectorFactoryTest::~SNPVectorFactoryTest() {

}

void SNPVectorFactoryTest::SetUp() {
  EXPECT_CALL(configMock, getMinorAlleleFrequencyThreshold()).Times(1).WillRepeatedly(Return(0.05));
  EXPECT_CALL(configMock, getGeneticModel()).Times(1).WillRepeatedly(Return(DOMINANT));
}

void SNPVectorFactoryTest::TearDown() {

}

TEST_F(SNPVectorFactoryTest, ConstructSNPVector_True) {
  SNPVectorFactory snpVectorFactory(configMock, numberOfIndividuals);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  ASSERT_TRUE(snp.getInclude());

  std::vector<int>* snpData = new std::vector<int>();
  std::vector<int>* alleleNumbers = new std::vector<int>(6);
  bool missingData = false;

  (*alleleNumbers)[ALLELE_ONE_CASE_POSITION] = 100;
  (*alleleNumbers)[ALLELE_TWO_CASE_POSITION] = 100;
  (*alleleNumbers)[ALLELE_ONE_CONTROL_POSITION] = 100;
  (*alleleNumbers)[ALLELE_TWO_CONTROL_POSITION] = 100;
  (*alleleNumbers)[ALLELE_ONE_ALL_POSITION] = 200;
  (*alleleNumbers)[ALLELE_TWO_ALL_POSITION] = 200;

  SNPVector* snpVector = snpVectorFactory.constructSNPVector(snp, snpData, alleleNumbers, missingData);
  ASSERT_TRUE(snp.getInclude());
}

TEST_F(SNPVectorFactoryTest, ConstructSNPVector_False) {
  SNPVectorFactory snpVectorFactory(configMock, numberOfIndividuals);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  ASSERT_TRUE(snp.getInclude());

  std::vector<int>* snpData = new std::vector<int>();
  std::vector<int>* alleleNumbers = new std::vector<int>(6);
  bool missingData = false;

  (*alleleNumbers)[ALLELE_ONE_CASE_POSITION] = 1;
  (*alleleNumbers)[ALLELE_TWO_CASE_POSITION] = 199;
  (*alleleNumbers)[ALLELE_ONE_CONTROL_POSITION] = 100;
  (*alleleNumbers)[ALLELE_TWO_CONTROL_POSITION] = 100;
  (*alleleNumbers)[ALLELE_ONE_ALL_POSITION] = 101;
  (*alleleNumbers)[ALLELE_TWO_ALL_POSITION] = 299;

  SNPVector* snpVector = snpVectorFactory.constructSNPVector(snp, snpData, alleleNumbers, missingData);
  ASSERT_FALSE(snp.getInclude());
}

TEST_F(SNPVectorFactoryTest, AlleleFrequencies) {
  SNPVectorFactory snpVectorFactory(configMock, numberOfIndividuals);

  std::vector<int> alleleNumbers(6);
  alleleNumbers[ALLELE_ONE_CASE_POSITION] = 1;
  alleleNumbers[ALLELE_TWO_CASE_POSITION] = 3;
  alleleNumbers[ALLELE_ONE_CONTROL_POSITION] = 3;
  alleleNumbers[ALLELE_TWO_CONTROL_POSITION] = 5;
  alleleNumbers[ALLELE_ONE_ALL_POSITION] = 4;
  alleleNumbers[ALLELE_TWO_ALL_POSITION] = 8;

  std::vector<double>* alleleFreqs = snpVectorFactory.convertAlleleNumbersToFrequencies(alleleNumbers);
  ASSERT_EQ(6, alleleFreqs->size());

  EXPECT_EQ((double )1 / 4, (*alleleFreqs)[ALLELE_ONE_CASE_POSITION]);
  EXPECT_EQ((double )3 / 4, (*alleleFreqs)[ALLELE_TWO_CASE_POSITION]);
  EXPECT_EQ((double )3 / 8, (*alleleFreqs)[ALLELE_ONE_CONTROL_POSITION]);
  EXPECT_EQ((double )5 / 8, (*alleleFreqs)[ALLELE_TWO_CONTROL_POSITION]);
  EXPECT_EQ((double )4 / 12, (*alleleFreqs)[ALLELE_ONE_ALL_POSITION]);
  EXPECT_EQ((double )8 / 12, (*alleleFreqs)[ALLELE_TWO_ALL_POSITION]);

  delete alleleFreqs;
}

TEST_F(SNPVectorFactoryTest, RiskAllele_Equal_Case_Control) {
  SNPVectorFactory snpVectorFactory(configMock, numberOfIndividuals);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  std::vector<double> alleleFreqs(6);
  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 1;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 1;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;
  snpVectorFactory.setSNPRiskAllele(snp, alleleFreqs);

  EXPECT_EQ(ALLELE_ONE, snp.getRiskAllele());
}

TEST_F(SNPVectorFactoryTest, RiskAllele_Equal_Case_Control1_larger_Control2) {
  SNPVectorFactory snpVectorFactory(configMock, numberOfIndividuals);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  std::vector<double> alleleFreqs(6);
  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 1;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 2;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 1;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;
  snpVectorFactory.setSNPRiskAllele(snp, alleleFreqs);

  EXPECT_EQ(ALLELE_TWO, snp.getRiskAllele());
}

TEST_F(SNPVectorFactoryTest, RiskAllele_Equal_Case_Control2_larger_Control1) {
  SNPVectorFactory snpVectorFactory(configMock, numberOfIndividuals);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  std::vector<double> alleleFreqs(6);
  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 1;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 2;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;
  snpVectorFactory.setSNPRiskAllele(snp, alleleFreqs);

  EXPECT_EQ(ALLELE_ONE, snp.getRiskAllele());
}

TEST_F(SNPVectorFactoryTest, RiskAllele_Case1_Larger_Case2_Case1_Larger_Control1) {
  SNPVectorFactory snpVectorFactory(configMock, numberOfIndividuals);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  std::vector<double> alleleFreqs(6);
  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 2;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 1;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 1;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;
  snpVectorFactory.setSNPRiskAllele(snp, alleleFreqs);

  EXPECT_EQ(ALLELE_ONE, snp.getRiskAllele());
}

TEST_F(SNPVectorFactoryTest, RiskAllele_Case1_Larger_Case2_Case1_Smaller_Control1) {
  SNPVectorFactory snpVectorFactory(configMock, numberOfIndividuals);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  std::vector<double> alleleFreqs(6);
  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 2;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 1;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 3;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 1;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;
  snpVectorFactory.setSNPRiskAllele(snp, alleleFreqs);

  EXPECT_EQ(ALLELE_TWO, snp.getRiskAllele());
}

TEST_F(SNPVectorFactoryTest, RiskAllele_Case2_Larger_Case1_Case2_Larger_Control2) {
  SNPVectorFactory snpVectorFactory(configMock, numberOfIndividuals);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  std::vector<double> alleleFreqs(6);
  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 2;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 1;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;
  snpVectorFactory.setSNPRiskAllele(snp, alleleFreqs);

  EXPECT_EQ(ALLELE_TWO, snp.getRiskAllele());
}

TEST_F(SNPVectorFactoryTest, RiskAllele_Case2_Larger_Case1_Case2_Smaller_Control2) {
  SNPVectorFactory snpVectorFactory(configMock, numberOfIndividuals);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  std::vector<double> alleleFreqs(6);
  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 2;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 3;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;
  snpVectorFactory.setSNPRiskAllele(snp, alleleFreqs);

  EXPECT_EQ(ALLELE_ONE, snp.getRiskAllele());
}

TEST_F(SNPVectorFactoryTest, SNPInclude_MissingData) {
  SNPVectorFactory snpVectorFactory(configMock, numberOfIndividuals);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  std::vector<double> alleleFreqs(6);
  std::vector<int> alleleNumbers(6);
  bool missingData = true;

  snpVectorFactory.setSNPInclude(snp, alleleFreqs, alleleNumbers, missingData);
  EXPECT_FALSE(snp.getInclude());
}

TEST_F(SNPVectorFactoryTest, SNPInclude_True) {
  SNPVectorFactory snpVectorFactory(configMock, numberOfIndividuals);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  std::vector<double> alleleFreqs(6);
  std::vector<int> alleleNumbers(6);
  bool missingData = false;

  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 0.5;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 0.5;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 0.5;

  alleleNumbers[ALLELE_ONE_CASE_POSITION] = 100;
  alleleNumbers[ALLELE_TWO_CASE_POSITION] = 100;
  alleleNumbers[ALLELE_ONE_CONTROL_POSITION] = 100;
  alleleNumbers[ALLELE_TWO_CONTROL_POSITION] = 100;
  alleleNumbers[ALLELE_ONE_ALL_POSITION] = 200;
  alleleNumbers[ALLELE_TWO_ALL_POSITION] = 200;

  snpVectorFactory.setSNPInclude(snp, alleleFreqs, alleleNumbers, missingData);
  EXPECT_TRUE(snp.getInclude());
}

TEST_F(SNPVectorFactoryTest, SNPInclude_ToLowAbsFreq_1) {
  SNPVectorFactory snpVectorFactory(configMock, numberOfIndividuals);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  std::vector<double> alleleFreqs(6);
  std::vector<int> alleleNumbers(6);
  bool missingData = false;

  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 0.5;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 0.5;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 0.5;

  alleleNumbers[ALLELE_ONE_CASE_POSITION] = 1;
  alleleNumbers[ALLELE_TWO_CASE_POSITION] = 100;
  alleleNumbers[ALLELE_ONE_CONTROL_POSITION] = 100;
  alleleNumbers[ALLELE_TWO_CONTROL_POSITION] = 100;
  alleleNumbers[ALLELE_ONE_ALL_POSITION] = 200;
  alleleNumbers[ALLELE_TWO_ALL_POSITION] = 200;

  snpVectorFactory.setSNPInclude(snp, alleleFreqs, alleleNumbers, missingData);
  EXPECT_FALSE(snp.getInclude());
}

TEST_F(SNPVectorFactoryTest, SNPInclude_ToLowAbsFreq_2) {
  SNPVectorFactory snpVectorFactory(configMock, numberOfIndividuals);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  std::vector<double> alleleFreqs(6);
  std::vector<int> alleleNumbers(6);
  bool missingData = false;

  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 0.5;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 0.5;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 0.5;

  alleleNumbers[ALLELE_ONE_CASE_POSITION] = 100;
  alleleNumbers[ALLELE_TWO_CASE_POSITION] = 1;
  alleleNumbers[ALLELE_ONE_CONTROL_POSITION] = 100;
  alleleNumbers[ALLELE_TWO_CONTROL_POSITION] = 100;
  alleleNumbers[ALLELE_ONE_ALL_POSITION] = 200;
  alleleNumbers[ALLELE_TWO_ALL_POSITION] = 200;

  snpVectorFactory.setSNPInclude(snp, alleleFreqs, alleleNumbers, missingData);
  EXPECT_FALSE(snp.getInclude());
}

TEST_F(SNPVectorFactoryTest, SNPInclude_ToLowAbsFreq_3) {
  SNPVectorFactory snpVectorFactory(configMock, numberOfIndividuals);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  std::vector<double> alleleFreqs(6);
  std::vector<int> alleleNumbers(6);
  bool missingData = false;

  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 0.5;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 0.5;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 0.5;

  alleleNumbers[ALLELE_ONE_CASE_POSITION] = 100;
  alleleNumbers[ALLELE_TWO_CASE_POSITION] = 100;
  alleleNumbers[ALLELE_ONE_CONTROL_POSITION] = 1;
  alleleNumbers[ALLELE_TWO_CONTROL_POSITION] = 100;
  alleleNumbers[ALLELE_ONE_ALL_POSITION] = 200;
  alleleNumbers[ALLELE_TWO_ALL_POSITION] = 200;

  snpVectorFactory.setSNPInclude(snp, alleleFreqs, alleleNumbers, missingData);
  EXPECT_FALSE(snp.getInclude());
}

TEST_F(SNPVectorFactoryTest, SNPInclude_ToLowAbsFreq_4) {
  SNPVectorFactory snpVectorFactory(configMock, numberOfIndividuals);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  std::vector<double> alleleFreqs(6);
  std::vector<int> alleleNumbers(6);
  bool missingData = false;

  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 0.5;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 0.5;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 0.5;

  alleleNumbers[ALLELE_ONE_CASE_POSITION] = 100;
  alleleNumbers[ALLELE_TWO_CASE_POSITION] = 100;
  alleleNumbers[ALLELE_ONE_CONTROL_POSITION] = 100;
  alleleNumbers[ALLELE_TWO_CONTROL_POSITION] = 1;
  alleleNumbers[ALLELE_ONE_ALL_POSITION] = 200;
  alleleNumbers[ALLELE_TWO_ALL_POSITION] = 200;

  snpVectorFactory.setSNPInclude(snp, alleleFreqs, alleleNumbers, missingData);
  EXPECT_FALSE(snp.getInclude());
}

TEST_F(SNPVectorFactoryTest, SNPInclude_ToLowMAF_Equal) {
  SNPVectorFactory snpVectorFactory(configMock, numberOfIndividuals);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  std::vector<double> alleleFreqs(6);
  std::vector<int> alleleNumbers(6);
  bool missingData = false;

  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 0.5;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 0.5;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 0.01;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 0.01;

  alleleNumbers[ALLELE_ONE_CASE_POSITION] = 100;
  alleleNumbers[ALLELE_TWO_CASE_POSITION] = 100;
  alleleNumbers[ALLELE_ONE_CONTROL_POSITION] = 100;
  alleleNumbers[ALLELE_TWO_CONTROL_POSITION] = 100;
  alleleNumbers[ALLELE_ONE_ALL_POSITION] = 200;
  alleleNumbers[ALLELE_TWO_ALL_POSITION] = 200;

  snpVectorFactory.setSNPInclude(snp, alleleFreqs, alleleNumbers, missingData);
  EXPECT_FALSE(snp.getInclude());
}

TEST_F(SNPVectorFactoryTest, SNPInclude_ToLowMAF_1Larger2) {
  SNPVectorFactory snpVectorFactory(configMock, numberOfIndividuals);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  std::vector<double> alleleFreqs(6);
  std::vector<int> alleleNumbers(6);
  bool missingData = false;

  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 0.5;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 0.5;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 0.02;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 0.01;

  alleleNumbers[ALLELE_ONE_CASE_POSITION] = 100;
  alleleNumbers[ALLELE_TWO_CASE_POSITION] = 100;
  alleleNumbers[ALLELE_ONE_CONTROL_POSITION] = 100;
  alleleNumbers[ALLELE_TWO_CONTROL_POSITION] = 100;
  alleleNumbers[ALLELE_ONE_ALL_POSITION] = 200;
  alleleNumbers[ALLELE_TWO_ALL_POSITION] = 200;

  snpVectorFactory.setSNPInclude(snp, alleleFreqs, alleleNumbers, missingData);
  EXPECT_FALSE(snp.getInclude());
}

TEST_F(SNPVectorFactoryTest, SNPInclude_ToLowMAF_2Larger1) {
  SNPVectorFactory snpVectorFactory(configMock, numberOfIndividuals);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  std::vector<double> alleleFreqs(6);
  std::vector<int> alleleNumbers(6);
  bool missingData = false;

  alleleFreqs[ALLELE_ONE_CASE_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_CASE_POSITION] = 0.5;
  alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 0.5;
  alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 0.5;
  alleleFreqs[ALLELE_ONE_ALL_POSITION] = 0.01;
  alleleFreqs[ALLELE_TWO_ALL_POSITION] = 0.02;

  alleleNumbers[ALLELE_ONE_CASE_POSITION] = 100;
  alleleNumbers[ALLELE_TWO_CASE_POSITION] = 100;
  alleleNumbers[ALLELE_ONE_CONTROL_POSITION] = 100;
  alleleNumbers[ALLELE_TWO_CONTROL_POSITION] = 100;
  alleleNumbers[ALLELE_ONE_ALL_POSITION] = 200;
  alleleNumbers[ALLELE_TWO_ALL_POSITION] = 200;

  snpVectorFactory.setSNPInclude(snp, alleleFreqs, alleleNumbers, missingData);
  EXPECT_FALSE(snp.getInclude());
}

} /* namespace Container */
} /* namespace CuEira */

