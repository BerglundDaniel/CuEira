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
#include <ConstructorHelpers.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <FileReaderException.h>

using testing::Return;

namespace CuEira {
namespace FileIO {

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

  static const int numberOfIndividualsTotalStatic = 10;
  static const int numberOfIndividualsToIncludeStatic = 6;
  const int numberOfIndividualsTotal;
  const int numberOfIndividualsToInclude;
  PersonHandlerMock personHandlerMock;
  ConfigurationMock configMock;
  CuEira_Test::ConstructorHelpers constructorHelpers;
  std::string filePath;
  const int notInclude[4] = {1, 2, 5, 7}; //Index 0 based
  const int numberOfSNPs = 10;
  const int numberOfSNPsToInclude = 8;
};

BedReaderTest::BedReaderTest() :
    filePath("../data/test.bed"), numberOfIndividualsTotal(numberOfIndividualsTotalStatic), numberOfIndividualsToInclude(
        numberOfIndividualsToIncludeStatic) {

}

BedReaderTest::~BedReaderTest() {

}

void BedReaderTest::SetUp() {
  //Expect Configuration
  EXPECT_CALL(configMock, getMinorAlleleFrequencyThreshold()).WillRepeatedly(Return(0.05));
  EXPECT_CALL(configMock, getBedFilePath()).Times(1).WillRepeatedly(Return(filePath));

  //Expect PersonHandler
  EXPECT_CALL(personHandlerMock, getNumberOfIndividualsTotal()).Times(1).WillRepeatedly(
      Return(numberOfIndividualsTotal));
  EXPECT_CALL(personHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillRepeatedly(
      Return(numberOfIndividualsToInclude));
}

void BedReaderTest::TearDown() {

}

TEST_F(BedReaderTest, ConstructorCheckMode) {
  EXPECT_CALL(configMock, getGeneticModel()).Times(1).WillRepeatedly(Return(DOMINANT));

  CuEira::FileIO::BedReader bedReader(configMock, personHandlerMock, numberOfSNPs);

  ASSERT_EQ(0, bedReader.mode);
}

TEST_F(BedReaderTest, ReadDominantInclude) {
  EXPECT_CALL(configMock, getGeneticModel()).Times(1).WillRepeatedly(Return(DOMINANT));

  //Expect PersonHandler


  //getPersonFromRowAll
  //getRowIncludeFromPerson

  CuEira::FileIO::BedReader bedReader(configMock, personHandlerMock, numberOfSNPs);

  //SNP snp(??);
  //bedReader.readSNP(snp);

}

//read snp ressecive test

}
/* namespace FileIO */
} /* namespace CuEira */

