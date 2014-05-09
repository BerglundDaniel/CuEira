#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include <SNP.h>
#include <Id.h>
#include <RiskAllele.h>

namespace CuEira {
namespace CuEira_Test {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class SNPTest: public ::testing::Test {
protected:
  SNPTest();
  virtual ~SNPTest();
  virtual void SetUp();
  virtual void TearDown();
};

SNPTest::SNPTest() {

}

SNPTest::~SNPTest() {

}

void SNPTest::SetUp() {

}

void SNPTest::TearDown() {

}

TEST_F(SNPTest, Getters) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  ASSERT_TRUE(snp1.getInclude());
  ASSERT_EQ(id1, snp1.getId());
  ASSERT_EQ(alleOneString1, snp1.getAlleleOneName());
  ASSERT_EQ(alleTwoString1, snp1.getAlleleTwoName());
  ASSERT_EQ(pos1, snp1.getPosition());

  Id id2("SNP2");
  unsigned int pos2 = 2;
  std::string alleOneString2("a2_1");
  std::string alleTwoString2("a2_2");
  SNP snp2(id2, alleOneString2, alleTwoString2, pos2, false);

  ASSERT_FALSE(snp2.getInclude());
  ASSERT_EQ(id2, snp2.getId());
  ASSERT_EQ(alleOneString2, snp2.getAlleleOneName());
  ASSERT_EQ(alleTwoString2, snp2.getAlleleTwoName());
  ASSERT_EQ(pos2, snp2.getPosition());
}

TEST_F(SNPTest, Include) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  snp1.setInclude(true);
  ASSERT_TRUE(snp1.getInclude());

  snp1.setInclude(false);
  ASSERT_FALSE(snp1.getInclude());
}

TEST_F(SNPTest, MafException) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  ASSERT_THROW(snp1.getMinorAlleleFrequency(), std::invalid_argument);
}

TEST_F(SNPTest, RiskAlleleException) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  ASSERT_THROW(snp1.getRiskAllele(), std::invalid_argument);
}

TEST_F(SNPTest, Maf) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  double maf = 0.5;
  snp1.setMinorAlleleFrequency(maf);
  ASSERT_EQ(maf, snp1.getMinorAlleleFrequency());
}

TEST_F(SNPTest, RiskAllele) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  RiskAllele riskAllele = ALLELE_ONE;
  snp1.setRiskAllele(riskAllele);
  ASSERT_EQ(riskAllele, snp1.getRiskAllele());

  riskAllele = ALLELE_TWO;
  snp1.setRiskAllele(riskAllele);
  ASSERT_EQ(riskAllele, snp1.getRiskAllele());
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */

