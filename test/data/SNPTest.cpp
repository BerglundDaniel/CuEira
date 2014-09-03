#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#include <SNP.h>
#include <Id.h>
#include <RiskAllele.h>
#include <InvalidState.h>
#include <SNPIncludeExclude.h>

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

  ASSERT_TRUE(snp1.shouldInclude());
  ASSERT_EQ(id1, snp1.getId());
  ASSERT_EQ(alleOneString1, snp1.getAlleleOneName());
  ASSERT_EQ(alleTwoString1, snp1.getAlleleTwoName());
  ASSERT_EQ(pos1, snp1.getPosition());

  const std::vector<SNPIncludeExclude>& includeVector1 = snp1.getInclude();
  ASSERT_EQ(includeVector1.size(), 1);
  ASSERT_EQ(includeVector1[0], INCLUDE);

  Id id2("SNP2");
  unsigned int pos2 = 2;
  std::string alleOneString2("a2_1");
  std::string alleTwoString2("a2_2");
  SNP snp2(id2, alleOneString2, alleTwoString2, pos2, MISSING_DATA);

  ASSERT_FALSE(snp2.shouldInclude());
  ASSERT_EQ(id2, snp2.getId());
  ASSERT_EQ(alleOneString2, snp2.getAlleleOneName());
  ASSERT_EQ(alleTwoString2, snp2.getAlleleTwoName());
  ASSERT_EQ(pos2, snp2.getPosition());

  const std::vector<SNPIncludeExclude>& includeVector2 = snp2.getInclude();
  ASSERT_EQ(includeVector2.size(), 1);
  ASSERT_EQ(includeVector2[0], MISSING_DATA);
}

TEST_F(SNPTest, Include) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  snp1.setInclude(INCLUDE);
  ASSERT_TRUE(snp1.shouldInclude());

  snp1.setInclude(MISSING_DATA);
  ASSERT_FALSE(snp1.shouldInclude());

  const std::vector<SNPIncludeExclude>& includeVector1 = snp1.getInclude();

  ASSERT_EQ(includeVector1.size(), 1);
  ASSERT_EQ(includeVector1[0], MISSING_DATA);

  snp1.setInclude(LOW_MAF);
  ASSERT_EQ(includeVector1.size(), 2);
  ASSERT_EQ(includeVector1[1], MISSING_DATA);

  snp1.setInclude(LOW_CELL_NUMBER);
  ASSERT_EQ(includeVector1.size(), 3);
  ASSERT_EQ(includeVector1[2], MISSING_DATA);

  snp1.setInclude(NEGATIVE_POSITION);
  ASSERT_EQ(includeVector1.size(), 4);
  ASSERT_EQ(includeVector1[3], MISSING_DATA);
}

TEST_F(SNPTest, RiskAlleleException) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  ASSERT_THROW(snp1.getRiskAllele(), InvalidState);
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

TEST_F(SNPTest, Operators) {
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");

  Id id1("env1");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  Id id2("env2");
  SNP snp2(id2, alleOneString1, alleTwoString1, pos1);

  Id id3("a_env");
  SNP snp3(id3, alleOneString1, alleTwoString1, pos1);

  SNP snp4(id1, alleOneString1, alleTwoString1, pos1);

  EXPECT_EQ(snp4, snp1);
  EXPECT_FALSE(snp1 == snp2);
  EXPECT_FALSE(snp4 == snp2);
  EXPECT_FALSE(snp1 == snp3);
  EXPECT_FALSE(snp4 == snp3);
  EXPECT_FALSE(snp3 == snp2);

  if(id1 < id2){
    EXPECT_TRUE(snp1 < snp2);
  }else{
    EXPECT_TRUE(snp2 < snp1);
  }

  if(id1 < id3){
    EXPECT_TRUE(snp1 < snp3);
  }else{
    EXPECT_TRUE(snp3 < snp1);
  }

  if(id2 < id3){
    EXPECT_TRUE(snp2 < snp3);
  }else{
    EXPECT_TRUE(snp3 < snp2);
  }
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */

