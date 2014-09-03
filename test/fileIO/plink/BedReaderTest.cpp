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
#include <SNPVectorFactoryMock.h>
#include <SNPVectorFactory.h>
#include <AlleleStatisticsFactoryMock.h>

using testing::Return;
using testing::_;
using testing::ReturnRef;
using testing::Eq;
using testing::SaveArg;
using testing::DoAll;

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
  CuEira_Test::ConstructorHelpers constructorHelpers;
  PersonHandlerMock personHandlerMock;
  ConfigurationMock configMock;
  Container::SNPVectorFactoryMock* snpVectorFactoryMock;
  AlleleStatisticsFactoryMock* alleleStatisticsFactoryMock;
  std::string filePath;
  std::vector<int> notInclude; //Index 0 based
  static const int numberOfSNPs = 10;

  Id* ids[numberOfIndividualsTotalStatic];
  Person* persons[numberOfIndividualsTotalStatic];
  int includePosArr[numberOfIndividualsTotalStatic];
};

BedReaderTest::BedReaderTest() :
    filePath(std::string(CuEira_BUILD_DIR) + std::string("/test.bed")), numberOfIndividualsTotal(
        numberOfIndividualsTotalStatic), numberOfIndividualsToInclude(numberOfIndividualsToIncludeStatic), notInclude(
        4), snpVectorFactoryMock(constructorHelpers.constructSNPVectorFactoryMock()),alleleStatisticsFactoryMock(new AlleleStatisticsFactoryMock) {
  notInclude[0] = 1;
  notInclude[1] = 2;
  notInclude[2] = 5;
  notInclude[3] = 7;
}

BedReaderTest::~BedReaderTest() {
  delete snpVectorFactoryMock;
  delete alleleStatisticsFactoryMock;
}

void BedReaderTest::SetUp() {
  //Expect Configuration
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
  CuEira::FileIO::BedReader bedReader(configMock, *snpVectorFactoryMock, personHandlerMock, numberOfSNPs);

  ASSERT_EQ(0, bedReader.mode);
}

TEST_F(BedReaderTest, ReadSnp0) {
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

  CuEira::FileIO::BedReader bedReader(configMock, *snpVectorFactoryMock, personHandlerMock, numberOfSNPs);

  Id id1("SNP1");
  unsigned int pos1 = 0; //First SNP
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  Container::SNPVectorMock* snpVectorMock = constructorHelpers.constructSNPVectorMock();
  std::vector<int>* originalSNPData = nullptr;
  std::vector<int>* numberOfAlleles = nullptr;
  bool missingData = true;

  EXPECT_CALL(*snpVectorFactoryMock, constructSNPVector(_,_,_,_)).Times(1).WillRepeatedly(
      DoAll(SaveArg<3>(&missingData), SaveArg<1>(&originalSNPData), SaveArg<2>(&numberOfAlleles),
          Return(snpVectorMock)));

  Container::SNPVector* snpVector = bedReader.readSNP(snp1);
  ASSERT_EQ(snpVectorMock, snpVector);
  delete snpVectorMock;

  ASSERT_FALSE(missingData);

  ASSERT_EQ(numberOfIndividualsToInclude, originalSNPData->size());
  ASSERT_EQ(6, numberOfAlleles->size());

  EXPECT_EQ(3, (*numberOfAlleles)[ALLELE_ONE_CASE_POSITION]);
  EXPECT_EQ(3, (*numberOfAlleles)[ALLELE_TWO_CASE_POSITION]);
  EXPECT_EQ(3, (*numberOfAlleles)[ALLELE_ONE_CONTROL_POSITION]);
  EXPECT_EQ(3, (*numberOfAlleles)[ALLELE_TWO_CONTROL_POSITION]);
  EXPECT_EQ(6, (*numberOfAlleles)[ALLELE_ONE_ALL_POSITION]);
  EXPECT_EQ(6, (*numberOfAlleles)[ALLELE_TWO_ALL_POSITION]);

  //Check data
  EXPECT_EQ(0, (*originalSNPData)[0]);
  EXPECT_EQ(2, (*originalSNPData)[1]);
  EXPECT_EQ(1, (*originalSNPData)[2]);
  EXPECT_EQ(1, (*originalSNPData)[3]);
  EXPECT_EQ(1, (*originalSNPData)[4]);
  EXPECT_EQ(1, (*originalSNPData)[5]);

  delete numberOfAlleles;
  delete originalSNPData;
}
TEST_F(BedReaderTest, ReadSnp1) {
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

  CuEira::FileIO::BedReader bedReader(configMock, *snpVectorFactoryMock, personHandlerMock, numberOfSNPs);

  Id id1("SNP1");
  unsigned int pos1 = 1; //Second SNP
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  Container::SNPVectorMock* snpVectorMock = constructorHelpers.constructSNPVectorMock();
  std::vector<int>* originalSNPData = nullptr;
  std::vector<int>* numberOfAlleles = nullptr;
  bool missingData = true;

  EXPECT_CALL(*snpVectorFactoryMock, constructSNPVector(_,_,_,_)).Times(1).WillRepeatedly(
      DoAll(SaveArg<3>(&missingData), SaveArg<1>(&originalSNPData), SaveArg<2>(&numberOfAlleles),
          Return(snpVectorMock)));

  Container::SNPVector* snpVector = bedReader.readSNP(snp1);
  ASSERT_EQ(snpVectorMock, snpVector);
  delete snpVectorMock;

  ASSERT_FALSE(missingData);

  ASSERT_EQ(numberOfIndividualsToInclude, originalSNPData->size());
  ASSERT_EQ(6, numberOfAlleles->size());

  EXPECT_EQ(1, (*numberOfAlleles)[ALLELE_ONE_CASE_POSITION]);
  EXPECT_EQ(5, (*numberOfAlleles)[ALLELE_TWO_CASE_POSITION]);
  EXPECT_EQ(2, (*numberOfAlleles)[ALLELE_ONE_CONTROL_POSITION]);
  EXPECT_EQ(4, (*numberOfAlleles)[ALLELE_TWO_CONTROL_POSITION]);
  EXPECT_EQ(3, (*numberOfAlleles)[ALLELE_ONE_ALL_POSITION]);
  EXPECT_EQ(9, (*numberOfAlleles)[ALLELE_TWO_ALL_POSITION]);

  //Check data
  EXPECT_EQ(2, (*originalSNPData)[0]);
  EXPECT_EQ(2, (*originalSNPData)[1]);
  EXPECT_EQ(1, (*originalSNPData)[2]);
  EXPECT_EQ(2, (*originalSNPData)[3]);
  EXPECT_EQ(2, (*originalSNPData)[4]);
  EXPECT_EQ(0, (*originalSNPData)[5]);

  delete numberOfAlleles;
  delete originalSNPData;
}

}
/* namespace FileIO */
} /* namespace CuEira */

