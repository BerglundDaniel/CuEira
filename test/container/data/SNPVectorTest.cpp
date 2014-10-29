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
#include <StatisticModel.h>
#include <RegularHostVector.h>

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

  const static int numberOfIndividuals = 6;
  std::vector<int>* originalSNPData;
  std::vector<PRECISION>* recodedSNPData;
};

SNPVectorTest::SNPVectorTest() :
    originalSNPData(nullptr), recodedSNPData(new std::vector<PRECISION>(numberOfIndividuals)) {

  //For dominant risk allele one
  (*recodedSNPData)[0] = 0;
  (*recodedSNPData)[1] = 1;
  (*recodedSNPData)[2] = 1;
  (*recodedSNPData)[3] = 0;
  (*recodedSNPData)[4] = 1;
  (*recodedSNPData)[5] = 1;
}

SNPVectorTest::~SNPVectorTest() {
  delete recodedSNPData;
}

void SNPVectorTest::SetUp() {
  originalSNPData = new std::vector<int>(numberOfIndividuals);

  (*originalSNPData)[0] = 2;
  (*originalSNPData)[1] = 1;
  (*originalSNPData)[2] = 0;
  (*originalSNPData)[3] = 2;
  (*originalSNPData)[4] = 0;
  (*originalSNPData)[5] = 1;
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

  SNPVector snpVector(snp1, DOMINANT, originalSNPData);
  ASSERT_EQ(ALLELE_ONE, snp1.getRiskAllele());
  ASSERT_EQ(snp1, snpVector.getAssociatedSNP());

  const std::vector<int>& orgData = snpVector.getOrginalData();
  const HostVector& recodedData = snpVector.getRecodedData();

  for(int i = 0; i < numberOfIndividuals; ++i){
    ASSERT_EQ((*originalSNPData)[i], (orgData)[i]);
    ASSERT_EQ((*recodedSNPData)[i], (recodedData)(i));
  }
}

TEST_F(SNPVectorTest, ConstructRecessive) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_ONE);

  SNPVector snpVector(snp1, RECESSIVE, originalSNPData);
  ASSERT_EQ(ALLELE_ONE, snp1.getRiskAllele());
  ASSERT_EQ(snp1, snpVector.getAssociatedSNP());

  const HostVector& recodedData = snpVector.getRecodedData();

  ASSERT_EQ(0, (recodedData)(0));
  ASSERT_EQ(0, (recodedData)(1));
  ASSERT_EQ(1, (recodedData)(2));
  ASSERT_EQ(0, (recodedData)(3));
  ASSERT_EQ(1, (recodedData)(4));
  ASSERT_EQ(0, (recodedData)(5));
}

TEST_F(SNPVectorTest, ReCodeSame) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_ONE);

  SNPVector snpVector(snp1, DOMINANT, originalSNPData);
  snpVector.recode(ALL_RISK);

  const HostVector& recodedData = snpVector.getRecodedData();

  ASSERT_EQ(ALLELE_ONE, snp1.getRiskAllele());

  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*recodedSNPData)[i], (recodedData)(i));
  }
}

TEST_F(SNPVectorTest, ReCodeSNP) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_ONE);

  SNPVector snpVector(snp1, DOMINANT, originalSNPData);
  snpVector.recode(SNP_PROTECT);

  const HostVector& recodedData = snpVector.getRecodedData();

  ASSERT_EQ(ALLELE_TWO, snp1.getRiskAllele());

  ASSERT_EQ(1, (recodedData)(0));
  ASSERT_EQ(0, (recodedData)(1));
  ASSERT_EQ(0, (recodedData)(2));
  ASSERT_EQ(1, (recodedData)(3));
  ASSERT_EQ(0, (recodedData)(4));
  ASSERT_EQ(0, (recodedData)(5));
}

TEST_F(SNPVectorTest, ReCodeInteraction) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_TWO);

  SNPVector snpVector(snp1, DOMINANT, originalSNPData);
  snpVector.recode(INTERACTION_PROTECT);

  const HostVector& recodedData = snpVector.getRecodedData();

  ASSERT_EQ(ALLELE_ONE, snp1.getRiskAllele());

  ASSERT_EQ(0, (recodedData)(0));
  ASSERT_EQ(0, (recodedData)(1));
  ASSERT_EQ(1, (recodedData)(2));
  ASSERT_EQ(0, (recodedData)(3));
  ASSERT_EQ(1, (recodedData)(4));
  ASSERT_EQ(0, (recodedData)(5));
}

TEST_F(SNPVectorTest, ReCodeEnvironment) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_ONE);

  SNPVector snpVector(snp1, DOMINANT, originalSNPData);
  snpVector.recode(ENVIRONMENT_PROTECT);

  const HostVector& recodedData = snpVector.getRecodedData();

  ASSERT_EQ(ALLELE_ONE, snp1.getRiskAllele());

  for(int i = 0; i < numberOfIndividuals; ++i){
    ASSERT_EQ((*recodedSNPData)[i], (recodedData)(i));
  }
}

TEST_F(SNPVectorTest, DoRecodeDominantAlleleOne) {
  Id id("SNP1");
  unsigned int pos = 1;
  std::string alleOneString("a1_1");
  std::string alleTwoString("a1_2");
  SNP snp1(id, alleOneString, alleTwoString, pos);
  snp1.setRiskAllele(ALLELE_ONE);

  SNPVector snpVector(snp1, DOMINANT, originalSNPData);
  snp1.setRiskAllele(ALLELE_TWO);
  snpVector.doRecode();
  const HostVector& recodedData = snpVector.getRecodedData();

  ASSERT_EQ(1, (recodedData)(0));
  ASSERT_EQ(1, (recodedData)(1));
  ASSERT_EQ(0, (recodedData)(2));
  ASSERT_EQ(1, (recodedData)(3));
  ASSERT_EQ(0, (recodedData)(4));
  ASSERT_EQ(1, (recodedData)(5));
}

TEST_F(SNPVectorTest, DoRecodeDominantAlleleTwo) {
  Id id("SNP2");
  unsigned int pos = 2;
  std::string alleOneString("a2_1");
  std::string alleTwoString("a2_2");
  SNP snp(id, alleOneString, alleTwoString, pos);
  snp.setRiskAllele(ALLELE_TWO);

  SNPVector snpVector(snp, DOMINANT, originalSNPData);
  snp.setRiskAllele(ALLELE_ONE);
  snpVector.doRecode();
  const HostVector& recodedData = snpVector.getRecodedData();

  ASSERT_EQ(0, (recodedData)(0));
  ASSERT_EQ(1, (recodedData)(1));
  ASSERT_EQ(1, (recodedData)(2));
  ASSERT_EQ(0, (recodedData)(3));
  ASSERT_EQ(1, (recodedData)(4));
  ASSERT_EQ(1, (recodedData)(5));
}

TEST_F(SNPVectorTest, DoRecodeRecessiveAlleleOne) {
  Id id("SNP1");
  unsigned int pos = 1;
  std::string alleOneString("a1_1");
  std::string alleTwoString("a1_2");
  SNP snp1(id, alleOneString, alleTwoString, pos);
  snp1.setRiskAllele(ALLELE_ONE);

  SNPVector snpVector(snp1, RECESSIVE, originalSNPData);
  snp1.setRiskAllele(ALLELE_TWO);
  snpVector.doRecode();
  const HostVector& recodedData = snpVector.getRecodedData();

  ASSERT_EQ(1, (recodedData)(0));
  ASSERT_EQ(0, (recodedData)(1));
  ASSERT_EQ(0, (recodedData)(2));
  ASSERT_EQ(1, (recodedData)(3));
  ASSERT_EQ(0, (recodedData)(4));
  ASSERT_EQ(0, (recodedData)(5));
}

TEST_F(SNPVectorTest, DoRecodeRecessiveAlleleTwo) {
  Id id("SNP2");
  unsigned int pos = 2;
  std::string alleOneString("a2_1");
  std::string alleTwoString("a2_2");
  SNP snp(id, alleOneString, alleTwoString, pos);
  snp.setRiskAllele(ALLELE_TWO);

  SNPVector snpVector(snp, RECESSIVE, originalSNPData);
  snp.setRiskAllele(ALLELE_ONE);
  snpVector.doRecode();
  const HostVector& recodedData = snpVector.getRecodedData();

  ASSERT_EQ(0, (recodedData)(0));
  ASSERT_EQ(0, (recodedData)(1));
  ASSERT_EQ(1, (recodedData)(2));
  ASSERT_EQ(0, (recodedData)(3));
  ASSERT_EQ(1, (recodedData)(4));
  ASSERT_EQ(0, (recodedData)(5));
}

TEST_F(SNPVectorTest, InvertRiskAllele) {
  Id id("SNP2");
  unsigned int pos = 2;
  std::string alleOneString("a2_1");
  std::string alleTwoString("a2_2");
  SNP snp(id, alleOneString, alleTwoString, pos);
  snp.setRiskAllele(ALLELE_TWO);

  SNPVector snpVector(snp, DOMINANT, originalSNPData);

  ASSERT_EQ(ALLELE_TWO, snpVector.invertRiskAllele(ALLELE_ONE));
  ASSERT_EQ(ALLELE_ONE, snpVector.invertRiskAllele(ALLELE_TWO));
}

TEST_F(SNPVectorTest, StatisticModel) {
  RegularHostVector interactionVector(numberOfIndividuals);

  for(int i = 0; i < numberOfIndividuals; ++i){
    if(i < 5){
      interactionVector(i) = 1;
    }else{
      interactionVector(i) = 0;
    }
  }

  Id id("SNP2");
  unsigned int pos = 2;
  std::string alleOneString("a2_1");
  std::string alleTwoString("a2_2");
  SNP snp(id, alleOneString, alleTwoString, pos);
  snp.setRiskAllele(ALLELE_ONE);

  SNPVector snpVector(snp, DOMINANT, originalSNPData);
  snpVector.applyStatisticModel(ADDITIVE, interactionVector);

  const Container::HostVector& snpData = snpVector.getRecodedData();
  for(int i = 0; i < numberOfIndividuals; ++i){
    if(i < 5){
      EXPECT_EQ(0, snpData(i));
    }else{
      EXPECT_EQ((*recodedSNPData)[i], snpData(i));
    }
  }
}

} /* namespace Container */
} /* namespace CuEira */

