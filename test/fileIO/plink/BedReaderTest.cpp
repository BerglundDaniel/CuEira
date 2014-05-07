#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>

#include <BedReader.h>
#include <Configuration.h>
#include <ConfigurationMock.h>
#include <PersonHandlerMock.h>
#include <GeneticModel.h>
#include <Person.h>
#include <Sex.h>
#include <Id.h>
#include <Phenotype.h>
#include <SNP.h>

using testing::Return;

namespace CuEira {
namespace CuEira_Test {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class BedReaderTest: public ::testing::Test {
protected:
  BedReaderTest();
  virtual ~BedReaderTest();
  virtual void SetUp();
  virtual void TearDown();
};

BedReaderTest::BedReaderTest() {

}

BedReaderTest::~BedReaderTest() {

}

void BedReaderTest::SetUp() {

}

void BedReaderTest::TearDown() {

}

TEST_F(BedReaderTest, ReadDominantInclude) {
  int numberOfSNPs = 10;
  int numberOfIndividualsTotal = 10;
  int numberOfIndividualsToInclude = 8;
  ConfigurationMock configMock;
  PersonHandlerMock personHandlerMock;

  //Expect Configuration
  EXPECT_CALL(configMock, getBedFilePath()).Times(1).WillRepeatedly(Return("../data/test.bed"));
  EXPECT_CALL(configMock, getGeneticModel()).Times(1).WillRepeatedly(Return(DOMINANT));
  EXPECT_CALL(configMock, getMinorAlleleFrequencyThreshold()).WillRepeatedly(Return(0.05));

  //Expect PersonHandler
  //getNumberOfIndividualsToInclude
  //getNumberOfIndividualsTotal
  //getPersonFromRowAll
  //getRowIncludeFromPerson

  //CuEira::FileIO::BedReader bedReader(configMock, personHandlerMock, numberOfSNPs);

  //SNP snp(??);
  //bedReader.readSNP(snp);

}

//TODO read snp ressecive test

}
/* namespace CuEira_Test */
} /* namespace CuEira */

