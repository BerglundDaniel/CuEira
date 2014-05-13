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
#include <SNPVector.h>

using testing::Return;
using testing::_;
using testing::ReturnRef;
using testing::Eq;

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

  Id* ids[numberOfIndividualsTotalStatic];
  Person* persons[numberOfIndividualsTotalStatic];
  int includePosArr[numberOfIndividualsTotalStatic];
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

TEST_F(BedReaderTest, ReadSnp0) {
  EXPECT_CALL(configMock, getGeneticModel()).Times(1).WillRepeatedly(Return(DOMINANT));

  int j = 0;
  int includePos = 0;
  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    Person* person;
    if(i == notInclude[j]){
      ++j;
      person = constructorHelpers.constructPersonNotInclude(i);
      includePosArr[i] = -1;
    }else{
      if(i < 5){
        person = constructorHelpers.constructPersonInclude(i, AFFECTED);
      }else{
        person = constructorHelpers.constructPersonInclude(i, UNAFFECTED);
      }
      includePosArr[i] = includePos;
      ++includePos;
    }
    persons[i] = person;
    ids[i] = new Id(person->getId().getString());
  }

  //Expect PersonHandler
  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    Person* person = persons[i];
    EXPECT_CALL(personHandlerMock, getPersonFromRowAll(i)).WillRepeatedly(ReturnRef(*person));
    EXPECT_CALL(personHandlerMock, getRowIncludeFromPerson(Eq(*person))).WillRepeatedly(Return(includePosArr[i]));
  }

  CuEira::FileIO::BedReader bedReader(configMock, personHandlerMock, numberOfSNPs);

  Id id1("SNP1");
  unsigned int pos1 = 0; //First SNP
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  Container::SNPVector* snpVector = bedReader.readSNP(snp1);
  ASSERT_TRUE(snp1.getInclude());
  const std::vector<int>* snpData = snpVector->getOrginalData();
  ASSERT_EQ(6, snpData->size());

  //Check maf and all freqs
  EXPECT_EQ(ALLELE_ONE, snp1.getRiskAllele());
  EXPECT_EQ(snp1.getAlleleOneAllFrequency(), snp1.getMinorAlleleFrequency());

  EXPECT_EQ(0.5, snp1.getAlleleOneCaseFrequency());
  EXPECT_EQ(0.5, snp1.getAlleleTwoCaseFrequency());
  EXPECT_EQ(0.5, snp1.getAlleleOneControlFrequency());
  EXPECT_EQ(0.5, snp1.getAlleleTwoControlFrequency());
  EXPECT_EQ(0.5, snp1.getAlleleOneAllFrequency());
  EXPECT_EQ(0.5, snp1.getAlleleTwoAllFrequency());

  //Check data
  EXPECT_EQ(0, (*snpData)[0]);
  EXPECT_EQ(2, (*snpData)[1]);
  EXPECT_EQ(1, (*snpData)[2]);
  EXPECT_EQ(1, (*snpData)[3]);
  EXPECT_EQ(1, (*snpData)[4]);
  EXPECT_EQ(1, (*snpData)[5]);

  delete snpVector;
}
TEST_F(BedReaderTest, ReadSnp1) {
  EXPECT_CALL(configMock, getGeneticModel()).Times(1).WillRepeatedly(Return(DOMINANT));

  int j = 0;
  int includePos = 0;
  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    Person* person;
    if(i == notInclude[j]){
      ++j;
      person = constructorHelpers.constructPersonNotInclude(i);
      includePosArr[i] = -1;
    }else{
      if(i < 5){
        person = constructorHelpers.constructPersonInclude(i, AFFECTED);
      }else{
        person = constructorHelpers.constructPersonInclude(i, UNAFFECTED);
      }
      includePosArr[i] = includePos;
      ++includePos;
    }
    persons[i] = person;
    ids[i] = new Id(person->getId().getString());
  }

  //Expect PersonHandler
  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    Person* person = persons[i];
    EXPECT_CALL(personHandlerMock, getPersonFromRowAll(i)).WillRepeatedly(ReturnRef(*person));
    EXPECT_CALL(personHandlerMock, getRowIncludeFromPerson(Eq(*person))).WillRepeatedly(Return(includePosArr[i]));
  }

  CuEira::FileIO::BedReader bedReader(configMock, personHandlerMock, numberOfSNPs);

  Id id1("SNP1");
  unsigned int pos1 = 1; //Second SNP
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  Container::SNPVector* snpVector = bedReader.readSNP(snp1);
  ASSERT_TRUE(snp1.getInclude());
  const std::vector<int>* snpData = snpVector->getOrginalData();
  ASSERT_EQ(6, snpData->size());

  //Check maf and all freqs
  EXPECT_EQ(ALLELE_TWO, snp1.getRiskAllele());
  EXPECT_EQ(snp1.getAlleleOneAllFrequency(), snp1.getMinorAlleleFrequency());

  EXPECT_EQ(1.0/6.0, snp1.getAlleleOneCaseFrequency());
  EXPECT_EQ(5.0/6.0, snp1.getAlleleTwoCaseFrequency());
  EXPECT_EQ(2.0/6.0, snp1.getAlleleOneControlFrequency());
  EXPECT_EQ(4.0/6.0, snp1.getAlleleTwoControlFrequency());
  EXPECT_EQ(3.0/12.0, snp1.getAlleleOneAllFrequency());
  EXPECT_EQ(9.0/12.0, snp1.getAlleleTwoAllFrequency());

  //Check data
  EXPECT_EQ(2, (*snpData)[0]);
  EXPECT_EQ(2, (*snpData)[1]);
  EXPECT_EQ(1, (*snpData)[2]);
  EXPECT_EQ(2, (*snpData)[3]);
  EXPECT_EQ(2, (*snpData)[4]);
  EXPECT_EQ(0, (*snpData)[5]);

  delete snpVector;
}

}
/* namespace FileIO */
} /* namespace CuEira */

