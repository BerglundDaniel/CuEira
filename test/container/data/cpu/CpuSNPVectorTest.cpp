#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <set>

#include <SNP.h>
#include <Id.h>
#include <Recode.h>
#include <CpuSNPVector.h>
#include <HostVector.h>
#include <InvalidState.h>
#include <GeneticModel.h>
#include <RiskAllele.h>
#include <RegularHostVector.h>

namespace CuEira {
namespace Container {
namespace CPU {

/**
 * Test for testing CpuSNPVector
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuSNPVectorTest: public ::testing::Test {
protected:
  CpuSNPVectorTest();
  virtual ~CpuSNPVectorTest();
  virtual void SetUp();
  virtual void TearDown();

  const int numberOfIndividuals;
  RegularHostVector* originalSNPData;
  std::set<int>* snpMissingData;
};

CpuSNPVectorTest::CpuSNPVectorTest() :
    numberOfIndividuals(6), originalSNPData(nullptr), snpMissingData(nullptr) {

}

CpuSNPVectorTest::~CpuSNPVectorTest() {

}

void CpuSNPVectorTest::SetUp() {
  originalSNPData = new RegularHostVector(numberOfIndividuals);
  for(int i = 0; i < numberOfIndividuals; ++i){
    (*originalSNPData)(i) = i % 2;
  }

  snpMissingData = new std::set<int>();
  snpMissingData->insert(3);
  snpMissingData->insert(8);
}

void CpuSNPVectorTest::TearDown() {

}

#ifdef DEBUG
TEST_F(CpuSNPVectorTest, Exceptions){
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  CpuSNPVector cpuSnpVector(snp1, DOMINANT, originalSNPData, snpMissingData);
  EXPECT_THROW(cpuSnpVector.getSNPData(), InvalidState);
}
#endif

TEST_F(CpuSNPVectorTest, ConstructAndGetWithMissing) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  CpuSNPVector cpuSnpVector(snp1, DOMINANT, originalSNPData, snpMissingData);
  ASSERT_EQ(snp1, cpuSnpVector.getAssociatedSNP());

  const RegularHostVector orgData = cpuSnpVector.getOriginalSNPData();
  EXCPECT_EQ(numberOfIndividuals, cpuSnpVector.getNumberOfIndividualsToInclude());
  EXCPECT_FALSE(cpuSnpVector.hasMissing());
  const std::set<int> missingSet = cpuSnpVector.getMissing();

  ASSERT_EQ(numberOfIndividuals, orgData.getNumberOfRows());
  ASSERT_EQ(originalSNPData->getMemoryPointer(), orgData.getMemoryPointer());

  EXPECT_EQ(snpMissingData, &missingSet);
}

TEST_F(CpuSNPVectorTest, ConstructAndGetWithoutMissing) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  delete snpMissingData;
  snpMissingData = new std::set<int>();

  CpuSNPVector cpuSnpVector(snp1, DOMINANT, originalSNPData, snpMissingData);
  ASSERT_EQ(snp1, cpuSnpVector.getAssociatedSNP());

  const RegularHostVector orgData = cpuSnpVector.getOriginalSNPData();
  EXCPECT_EQ(numberOfIndividuals, cpuSnpVector.getNumberOfIndividualsToInclude());
  EXCPECT_TRUE(cpuSnpVector.hasMissing());
  ASSERT_EQ(numberOfIndividuals, orgData.getNumberOfRows());
  ASSERT_EQ(originalSNPData->getMemoryPointer(), orgData.getMemoryPointer());

  const std::set<int> missingSet = cpuSnpVector.getMissing();
  EXPECT_TRUE(missingSet.empty());
}

TEST_F(CpuSNPVectorTest, ReCodeAllRiskDominant) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_ONE);

  CpuSNPVector cpuSnpVector(snp1, DOMINANT, originalSNPData, snpMissingData);
  cpuSnpVector.recode(ALL_RISK);

  const HostVector& recodedData = cpuSnpVector.getSNPData();

  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());
  ASSERT_EQ(ALLELE_ONE, snp1.getRiskAllele());

  EXCPECT_EQ(1, (recodedData)(0));
  EXCPECT_EQ(1, (recodedData)(1));
  EXCPECT_EQ(0, (recodedData)(2));
  EXCPECT_EQ(1, (recodedData)(3));
  EXCPECT_EQ(1, (recodedData)(4));
  EXCPECT_EQ(0, (recodedData)(5));
}

TEST_F(CpuSNPVectorTest, ReCodeEnvironmentDominant) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_TWO);

  CpuSNPVector cpuSnpVector(snp1, DOMINANT, originalSNPData, snpMissingData);
  cpuSnpVector.recode(ENVIRONMENT_PROTECT);

  const HostVector& recodedData = cpuSnpVector.getSNPData();

  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());
  ASSERT_EQ(ALLELE_TWO, snp1.getRiskAllele());

  EXCPECT_EQ(0, (recodedData)(0));
  EXCPECT_EQ(1, (recodedData)(1));
  EXCPECT_EQ(1, (recodedData)(2));
  EXCPECT_EQ(0, (recodedData)(3));
  EXCPECT_EQ(1, (recodedData)(4));
  EXCPECT_EQ(1, (recodedData)(5));
}

TEST_F(CpuSNPVectorTest, ReCodeSNPDominant) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_ONE);

  CpuSNPVector cpuSnpVector(snp1, DOMINANT, originalSNPData, snpMissingData);
  cpuSnpVector.recode(SNP_PROTECT);

  const HostVector& recodedData = cpuSnpVector.getSNPData();

  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());
  ASSERT_EQ(ALLELE_TWO, snp1.getRiskAllele());

  EXCPECT_EQ(0, (recodedData)(0));
  EXCPECT_EQ(1, (recodedData)(1));
  EXCPECT_EQ(1, (recodedData)(2));
  EXCPECT_EQ(0, (recodedData)(3));
  EXCPECT_EQ(1, (recodedData)(4));
  EXCPECT_EQ(1, (recodedData)(5));
}

TEST_F(CpuSNPVectorTest, ReCodeInteractionDominant) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_TWO);

  CpuSNPVector cpuSnpVector(snp1, DOMINANT, originalSNPData, snpMissingData);
  cpuSnpVector.recode(INTERACTION_PROTECT);

  const HostVector& recodedData = cpuSnpVector.getSNPData();

  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());
  ASSERT_EQ(ALLELE_ONE, snp1.getRiskAllele());

  EXCPECT_EQ(1, (recodedData)(0));
  EXCPECT_EQ(1, (recodedData)(1));
  EXCPECT_EQ(0, (recodedData)(2));
  EXCPECT_EQ(1, (recodedData)(3));
  EXCPECT_EQ(1, (recodedData)(4));
  EXCPECT_EQ(0, (recodedData)(5));
}

TEST_F(CpuSNPVectorTest, ReCodeAllRiskRecessive) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_ONE);

  CpuSNPVector cpuSnpVector(snp1, RECESSIVE, originalSNPData, snpMissingData);
  cpuSnpVector.recode(ALL_RISK);

  const HostVector& recodedData = cpuSnpVector.getSNPData();

  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());
  ASSERT_EQ(ALLELE_ONE, snp1.getRiskAllele());

  EXCPECT_EQ(1, (recodedData)(0));
  EXCPECT_EQ(0, (recodedData)(1));
  EXCPECT_EQ(0, (recodedData)(2));
  EXCPECT_EQ(1, (recodedData)(3));
  EXCPECT_EQ(0, (recodedData)(4));
  EXCPECT_EQ(0, (recodedData)(5));
}

TEST_F(CpuSNPVectorTest, ReCodeEnvironmentRecessive) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_TWO);

  CpuSNPVector cpuSnpVector(snp1, RECESSIVE, originalSNPData, snpMissingData);
  cpuSnpVector.recode(ENVIRONMENT_PROTECT);

  const HostVector& recodedData = cpuSnpVector.getSNPData();

  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());
  ASSERT_EQ(ALLELE_TWO, snp1.getRiskAllele());

  EXCPECT_EQ(0, (recodedData)(0));
  EXCPECT_EQ(0, (recodedData)(1));
  EXCPECT_EQ(1, (recodedData)(2));
  EXCPECT_EQ(0, (recodedData)(3));
  EXCPECT_EQ(0, (recodedData)(4));
  EXCPECT_EQ(1, (recodedData)(5));
}

TEST_F(CpuSNPVectorTest, ReCodeSNPRecessive) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_ONE);

  CpuSNPVector cpuSnpVector(snp1, RECESSIVE, originalSNPData, snpMissingData);
  cpuSnpVector.recode(SNP_PROTECT);

  const HostVector& recodedData = cpuSnpVector.getSNPData();

  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());
  ASSERT_EQ(ALLELE_TWO, snp1.getRiskAllele());

  EXCPECT_EQ(0, (recodedData)(0));
  EXCPECT_EQ(0, (recodedData)(1));
  EXCPECT_EQ(1, (recodedData)(2));
  EXCPECT_EQ(0, (recodedData)(3));
  EXCPECT_EQ(0, (recodedData)(4));
  EXCPECT_EQ(1, (recodedData)(5));
}

TEST_F(CpuSNPVectorTest, ReCodeInteractionRecessive) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_TWO);

  CpuSNPVector cpuSnpVector(snp1, RECESSIVE, originalSNPData, snpMissingData);
  cpuSnpVector.recode(INTERACTION_PROTECT);

  const HostVector& recodedData = cpuSnpVector.getSNPData();

  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());
  ASSERT_EQ(ALLELE_ONE, snp1.getRiskAllele());

  EXCPECT_EQ(1, (recodedData)(0));
  EXCPECT_EQ(0, (recodedData)(1));
  EXCPECT_EQ(0, (recodedData)(2));
  EXCPECT_EQ(1, (recodedData)(3));
  EXCPECT_EQ(0, (recodedData)(4));
  EXCPECT_EQ(0, (recodedData)(5));
}

TEST_F(CpuSNPVectorTest, ReCodeDominantStack) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_ONE);

  CpuSNPVector cpuSnpVector(snp1, DOMINANT, originalSNPData, snpMissingData);

  cpuSnpVector.recode(ALL_RISK);
  const HostVector& recodedData1 = cpuSnpVector.getSNPData();
  ASSERT_EQ(numberOfIndividuals, recodedData1.getNumberOfRows());
  ASSERT_EQ(ALLELE_ONE, snp1.getRiskAllele());

  EXCPECT_EQ(1, recodedData1(0));
  EXCPECT_EQ(1, recodedData1(1));
  EXCPECT_EQ(0, recodedData1(2));
  EXCPECT_EQ(1, recodedData1(3));
  EXCPECT_EQ(1, recodedData1(4));
  EXCPECT_EQ(0, recodedData1(5));

  cpuSnpVector.recode(ALL_RISK);
  const HostVector& recodedData2 = cpuSnpVector.getSNPData();
  ASSERT_EQ(ALLELE_ONE, snp1.getRiskAllele());

  EXCPECT_EQ(1, recodedData2(0));
  EXCPECT_EQ(1, recodedData2(1));
  EXCPECT_EQ(0, recodedData2(2));
  EXCPECT_EQ(1, recodedData2(3));
  EXCPECT_EQ(1, recodedData2(4));
  EXCPECT_EQ(0, recodedData2(5));

  cpuSnpVector.recode(SNP_PROTECT);
  const HostVector& recodedData3 = cpuSnpVector.getSNPData();
  ASSERT_EQ(ALLELE_TWO, snp1.getRiskAllele());

  EXCPECT_EQ(0, recodedData3(0));
  EXCPECT_EQ(1, recodedData3(1));
  EXCPECT_EQ(1, recodedData3(2));
  EXCPECT_EQ(0, recodedData3(3));
  EXCPECT_EQ(1, recodedData3(4));
  EXCPECT_EQ(1, recodedData3(5));

  cpuSnpVector.recode(INTERACTION_PROTECT);
  const HostVector& recodedData4 = cpuSnpVector.getSNPData();
  ASSERT_EQ(ALLELE_TWO, snp1.getRiskAllele());

  EXCPECT_EQ(0, recodedData4(0));
  EXCPECT_EQ(1, recodedData4(1));
  EXCPECT_EQ(1, recodedData4(2));
  EXCPECT_EQ(0, recodedData4(3));
  EXCPECT_EQ(1, recodedData4(4));
  EXCPECT_EQ(1, recodedData4(5));
}

} /* namespace CPU */
} /* namespace Container */
} /* namespace CuEira */
