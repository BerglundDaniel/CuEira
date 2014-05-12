#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>

#include <BimReader.h>
#include <Configuration.h>
#include <SNP.h>
#include <ConfigurationMock.h>

using testing::Return;

namespace CuEira {
namespace CuEira_Test {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class BimReaderTest: public ::testing::Test {
protected:
  BimReaderTest();
  virtual ~BimReaderTest();
  virtual void SetUp();
  virtual void TearDown();
};

BimReaderTest::BimReaderTest() {

}

BimReaderTest::~BimReaderTest() {

}

void BimReaderTest::SetUp() {

}

void BimReaderTest::TearDown() {

}

TEST_F(BimReaderTest, ReadFile) {
  int numberOfSNPs = 10;
  ConfigurationMock configMock;

  EXPECT_CALL(configMock, getBimFilePath()).Times(1).WillRepeatedly(Return("../data/test.bim"));
  EXPECT_CALL(configMock, excludeSNPsWithNegativePosition()).Times(numberOfSNPs).WillRepeatedly(Return(true));

  CuEira::FileIO::BimReader bimReader(configMock);

  ASSERT_EQ(numberOfSNPs, bimReader.getNumberOfSNPs());

  std::vector<SNP*> snpVector = bimReader.getSNPInformation();
  std::vector<std::string> alleleOneVector(numberOfSNPs);
  std::vector<std::string> alleleTwoVector(numberOfSNPs);

  alleleOneVector[0] = "A";
  alleleOneVector[1] = "G";
  alleleOneVector[2] = "A";
  alleleOneVector[3] = "A";
  alleleOneVector[4] = "G";
  alleleOneVector[5] = "G";
  alleleOneVector[6] = "G";
  alleleOneVector[7] = "G";
  alleleOneVector[8] = "G";
  alleleOneVector[9] = "G";

  alleleTwoVector[0] = "G";
  alleleTwoVector[1] = "A";
  alleleTwoVector[2] = "G";
  alleleTwoVector[3] = "G";
  alleleTwoVector[4] = "A";
  alleleTwoVector[5] = "A";
  alleleTwoVector[6] = "A";
  alleleTwoVector[7] = "A";
  alleleTwoVector[8] = "A";
  alleleTwoVector[9] = "A";

  for(int i = 0; i < numberOfSNPs; ++i){
    SNP snp = *snpVector[i];

    //Check id
    std::ostringstream os;
    os << "rs" << i;
    const std::string& tmp = os.str();
    Id id(tmp.c_str());
    ASSERT_EQ(id, snp.getId());

    //Check include
    if(i == 3 || i == 9){
      ASSERT_FALSE(snp.getInclude());
    }else{
      ASSERT_TRUE(snp.getInclude());
    }

    ASSERT_EQ(i, snp.getPosition());

    //Check alleles
    ASSERT_TRUE(alleleOneVector[i] == snp.getAlleleOneName());
    ASSERT_TRUE(alleleTwoVector[i] == snp.getAlleleTwoName());

  } //end for i
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */

