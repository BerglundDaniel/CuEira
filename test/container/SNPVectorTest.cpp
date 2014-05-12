#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include <SNP.h>
#include <Id.h>
#include <Recode.h>
#include <SNPVector.h>
#include <HostVector.h>
#include <InvalidState.h>
#include <GeneticModel.h>
#include <RiskAllele.h>

namespace CuEira {
namespace Container {

/**
 * Test for testing SNPVector
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class SNPVectorTest: public ::testing::Test {
protected:
  SNPVectorTest();
  virtual ~SNPVectorTest();
  virtual void SetUp();
  virtual void TearDown();

  const int numberOfIndividuals = 6;
  std::vector<int>* originalSNPData;
  std::vector<PRECISION>* recodedSNPData;
};

SNPVectorTest::SNPVectorTest() :
    originalSNPData(new std::vector<int>(numberOfIndividuals)), recodedSNPData(
        new std::vector<PRECISION>(numberOfIndividuals)) {
  (*originalSNPData)[0] = 2;
  (*originalSNPData)[1] = 1;
  (*originalSNPData)[2] = 0;
  (*originalSNPData)[3] = 2;
  (*originalSNPData)[4] = 0;
  (*originalSNPData)[5] = 1;

  //For dominant risk allele one
  (*recodedSNPData)[0] = 0;
  (*recodedSNPData)[1] = 1;
  (*recodedSNPData)[2] = 1;
  (*recodedSNPData)[3] = 0;
  (*recodedSNPData)[4] = 1;
  (*recodedSNPData)[5] = 1;
}

SNPVectorTest::~SNPVectorTest() {
  delete originalSNPData;
  delete recodedSNPData;
}

void SNPVectorTest::SetUp() {

}

void SNPVectorTest::TearDown() {

}

TEST_F(SNPVectorTest, ConstructAndGetDominant) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_ONE);

  SNPVector snpVector(originalSNPData, snp1, DOMINANT);

  const std::vector<int>* orgData = snpVector.getOrginalData();
  const HostVector* recodedData = snpVector.getRecodedData();

  ASSERT_EQ(snp1.getId(), snpVector.getAssociatedSNP().getId());
  ASSERT_EQ(ALL_RISK, snpVector.getRecode());

  for(int i = 0; i < numberOfIndividuals; ++i){
    ASSERT_EQ((*originalSNPData)[i], (*orgData)[i]);
    ASSERT_EQ((*recodedSNPData)[i], (*recodedData)(i));
  }
}

TEST_F(SNPVectorTest, ConstructRecessive) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_ONE);

  SNPVector snpVector(originalSNPData, snp1, RECESSIVE);

  const HostVector* recodedData = snpVector.getRecodedData();

  ASSERT_EQ(ALL_RISK, snpVector.getRecode());

  ASSERT_EQ(0, (*recodedData)(0));
  ASSERT_EQ(0, (*recodedData)(1));
  ASSERT_EQ(1, (*recodedData)(2));
  ASSERT_EQ(0, (*recodedData)(3));
  ASSERT_EQ(1, (*recodedData)(4));
  ASSERT_EQ(0, (*recodedData)(5));
}

TEST_F(SNPVectorTest, ReCodeSame) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_ONE);

  SNPVector snpVector(originalSNPData, snp1, DOMINANT);
  snpVector.recode(ALL_RISK);

  const HostVector* recodedData = snpVector.getRecodedData();

  ASSERT_EQ(ALL_RISK, snpVector.getRecode());
  ASSERT_EQ(ALLELE_ONE, snp1.getRiskAllele());

  for(int i = 0; i < numberOfIndividuals; ++i){
    ASSERT_EQ((*recodedSNPData)[i], (*recodedData)(i));
  }
}

TEST_F(SNPVectorTest, ReCodeSNP) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_ONE);

  SNPVector snpVector(originalSNPData, snp1, DOMINANT);
  snpVector.recode(SNP_PROTECT);

  const HostVector* recodedData = snpVector.getRecodedData();

  ASSERT_EQ(SNP_PROTECT, snpVector.getRecode());
  ASSERT_EQ(ALLELE_TWO, snp1.getRiskAllele());

  ASSERT_EQ(1, (*recodedData)(0));
  ASSERT_EQ(0, (*recodedData)(1));
  ASSERT_EQ(0, (*recodedData)(2));
  ASSERT_EQ(1, (*recodedData)(3));
  ASSERT_EQ(0, (*recodedData)(4));
  ASSERT_EQ(0, (*recodedData)(5));
}

TEST_F(SNPVectorTest, ReCodeInteraction) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_TWO);

  SNPVector snpVector(originalSNPData, snp1, DOMINANT);
  snpVector.recode(INTERACTION_PROTECT);

  const HostVector* recodedData = snpVector.getRecodedData();

  ASSERT_EQ(INTERACTION_PROTECT, snpVector.getRecode());
  ASSERT_EQ(ALLELE_ONE, snp1.getRiskAllele());

  ASSERT_EQ(0, (*recodedData)(0));
  ASSERT_EQ(0, (*recodedData)(1));
  ASSERT_EQ(1, (*recodedData)(2));
  ASSERT_EQ(0, (*recodedData)(3));
  ASSERT_EQ(1, (*recodedData)(4));
  ASSERT_EQ(0, (*recodedData)(5));
}

TEST_F(SNPVectorTest, ReCodeEnvironment) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_ONE);

  SNPVector snpVector(originalSNPData, snp1, DOMINANT);
  snpVector.recode(ENVIRONMENT_PROTECT);

  const HostVector* recodedData = snpVector.getRecodedData();

  ASSERT_EQ(ENVIRONMENT_PROTECT, snpVector.getRecode());
  ASSERT_EQ(ALLELE_ONE, snp1.getRiskAllele());

  for(int i = 0; i < numberOfIndividuals; ++i){
    ASSERT_EQ((*recodedSNPData)[i], (*recodedData)(i));
  }
}

TEST_F(SNPVectorTest, DoRecodeDominantAlleleOne) {
  Id id("SNP1");
  unsigned int pos = 1;
  std::string alleOneString("a1_1");
  std::string alleTwoString("a1_2");
  SNP snp1(id, alleOneString, alleTwoString, pos);
  snp1.setRiskAllele(ALLELE_ONE);

  SNPVector snpVector(originalSNPData, snp1, DOMINANT);
  snpVector.currentRiskAllele=ALLELE_TWO;
  snpVector.doRecode();
  const HostVector* recodedData = snpVector.getRecodedData();

  ASSERT_EQ(1, (*recodedData)(0));
  ASSERT_EQ(1, (*recodedData)(1));
  ASSERT_EQ(0, (*recodedData)(2));
  ASSERT_EQ(1, (*recodedData)(3));
  ASSERT_EQ(0, (*recodedData)(4));
  ASSERT_EQ(1, (*recodedData)(5));
}

TEST_F(SNPVectorTest, DoRecodeDominantAlleleTwo) {
  Id id("SNP2");
  unsigned int pos = 2;
  std::string alleOneString("a2_1");
  std::string alleTwoString("a2_2");
  SNP snp(id, alleOneString, alleTwoString, pos);
  snp.setRiskAllele(ALLELE_TWO);

  SNPVector snpVector(originalSNPData, snp, DOMINANT);
  snpVector.currentRiskAllele=ALLELE_ONE;
  snpVector.doRecode();
  const HostVector* recodedData = snpVector.getRecodedData();

  ASSERT_EQ(0, (*recodedData)(0));
  ASSERT_EQ(1, (*recodedData)(1));
  ASSERT_EQ(1, (*recodedData)(2));
  ASSERT_EQ(0, (*recodedData)(3));
  ASSERT_EQ(1, (*recodedData)(4));
  ASSERT_EQ(1, (*recodedData)(5));
}

TEST_F(SNPVectorTest, DoRecodeRecessiveAlleleOne) {
  Id id("SNP1");
  unsigned int pos = 1;
  std::string alleOneString("a1_1");
  std::string alleTwoString("a1_2");
  SNP snp1(id, alleOneString, alleTwoString, pos);
  snp1.setRiskAllele(ALLELE_ONE);

  SNPVector snpVector(originalSNPData, snp1, RECESSIVE);
  snpVector.currentRiskAllele=ALLELE_TWO;
  snpVector.doRecode();
  const HostVector* recodedData = snpVector.getRecodedData();

  ASSERT_EQ(1, (*recodedData)(0));
  ASSERT_EQ(0, (*recodedData)(1));
  ASSERT_EQ(0, (*recodedData)(2));
  ASSERT_EQ(1, (*recodedData)(3));
  ASSERT_EQ(0, (*recodedData)(4));
  ASSERT_EQ(0, (*recodedData)(5));
}

TEST_F(SNPVectorTest, DoRecodeRecessiveAlleleTwo) {
  Id id("SNP2");
  unsigned int pos = 2;
  std::string alleOneString("a2_1");
  std::string alleTwoString("a2_2");
  SNP snp(id, alleOneString, alleTwoString, pos);
  snp.setRiskAllele(ALLELE_TWO);

  SNPVector snpVector(originalSNPData, snp, RECESSIVE);
  snpVector.currentRiskAllele=ALLELE_ONE;
  snpVector.doRecode();
  const HostVector* recodedData = snpVector.getRecodedData();

  ASSERT_EQ(0, (*recodedData)(0));
  ASSERT_EQ(0, (*recodedData)(1));
  ASSERT_EQ(1, (*recodedData)(2));
  ASSERT_EQ(0, (*recodedData)(3));
  ASSERT_EQ(1, (*recodedData)(4));
  ASSERT_EQ(0, (*recodedData)(5));
}

TEST_F(SNPVectorTest, InvertRiskAllele) {
  Id id("SNP2");
  unsigned int pos = 2;
  std::string alleOneString("a2_1");
  std::string alleTwoString("a2_2");
  SNP snp(id, alleOneString, alleTwoString, pos);
  snp.setRiskAllele(ALLELE_TWO);

  SNPVector snpVector(originalSNPData, snp, DOMINANT);

  ASSERT_EQ(ALLELE_TWO, snpVector.invertRiskAllele(ALLELE_ONE));
  ASSERT_EQ(ALLELE_ONE, snpVector.invertRiskAllele(ALLELE_TWO));
}

} /* namespace Container */
} /* namespace CuEira */

