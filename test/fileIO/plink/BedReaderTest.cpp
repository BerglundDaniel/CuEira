#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>
#include <utility>

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
#include <AlleleStatisticsMock.h>

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
        4), snpVectorFactoryMock(constructorHelpers.constructSNPVectorFactoryMock()), alleleStatisticsFactoryMock(
        new AlleleStatisticsFactoryMock) {
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
  EXPECT_CALL(configMock, getMinorAlleleFrequencyThreshold()).Times(1).WillRepeatedly(Return(0));

  //Expect PersonHandler
  EXPECT_CALL(personHandlerMock, getNumberOfIndividualsTotal()).Times(1).WillRepeatedly(
      Return(numberOfIndividualsTotal));
  EXPECT_CALL(personHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillRepeatedly(
      Return(numberOfIndividualsToInclude));
}

void BedReaderTest::TearDown() {

}

TEST_F(BedReaderTest, ConstructorCheckMode) {
  CuEira::FileIO::BedReader bedReader(configMock, *snpVectorFactoryMock, *alleleStatisticsFactoryMock,
      personHandlerMock, numberOfSNPs);

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

  CuEira::FileIO::BedReader bedReader(configMock, *snpVectorFactoryMock, *alleleStatisticsFactoryMock,
      personHandlerMock, numberOfSNPs);

  Id id1("SNP1");
  unsigned int pos1 = 0; //First SNP
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  Container::SNPVectorMock* snpVectorMock = constructorHelpers.constructSNPVectorMock();
  AlleleStatisticsMock* alleleStatisticsMock = new AlleleStatisticsMock();
  const std::vector<int>* originalSNPData = nullptr;
  const std::vector<int>* numberOfAlleles = nullptr;

  std::vector<double> alleleFrequencies(6);
  alleleFrequencies[ALLELE_ONE_CASE_POSITION] = 1;
  alleleFrequencies[ALLELE_TWO_CASE_POSITION] = 1;
  alleleFrequencies[ALLELE_ONE_CONTROL_POSITION] = 1;
  alleleFrequencies[ALLELE_TWO_CONTROL_POSITION] = 1;
  alleleFrequencies[ALLELE_ONE_ALL_POSITION] = 1;
  alleleFrequencies[ALLELE_TWO_ALL_POSITION] = 1;

  EXPECT_CALL(*snpVectorFactoryMock, constructSNPVector(_,_)).Times(1).WillRepeatedly(
      DoAll(SaveArg<1>(&originalSNPData), Return(snpVectorMock)));

  EXPECT_CALL(*alleleStatisticsFactoryMock, constructAlleleStatistics(_)).Times(1).WillRepeatedly(
      DoAll(SaveArg<0>(&numberOfAlleles), Return(alleleStatisticsMock)));

  EXPECT_CALL(*alleleStatisticsMock, getAlleleFrequencies()).Times(2).WillRepeatedly(ReturnRef(alleleFrequencies));

  std::pair<const AlleleStatistics*, Container::SNPVector*>* pair = bedReader.readSNP(snp1);
  ASSERT_EQ(snpVectorMock, pair->second);
  ASSERT_EQ(alleleStatisticsMock, pair->first);

  delete alleleStatisticsMock;
  delete snpVectorMock;
  delete pair;

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

  CuEira::FileIO::BedReader bedReader(configMock, *snpVectorFactoryMock, *alleleStatisticsFactoryMock,
      personHandlerMock, numberOfSNPs);

  Id id1("SNP1");
  unsigned int pos1 = 1; //Second SNP
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  Container::SNPVectorMock* snpVectorMock = constructorHelpers.constructSNPVectorMock();
  AlleleStatisticsMock* alleleStatisticsMock = new AlleleStatisticsMock();
  const std::vector<int>* originalSNPData = nullptr;
  const std::vector<int>* numberOfAlleles = nullptr;

  std::vector<double> alleleFrequencies(6);
  alleleFrequencies[ALLELE_ONE_CASE_POSITION] = 1;
  alleleFrequencies[ALLELE_TWO_CASE_POSITION] = 1;
  alleleFrequencies[ALLELE_ONE_CONTROL_POSITION] = 1;
  alleleFrequencies[ALLELE_TWO_CONTROL_POSITION] = 1;
  alleleFrequencies[ALLELE_ONE_ALL_POSITION] = 1;
  alleleFrequencies[ALLELE_TWO_ALL_POSITION] = 1;

  EXPECT_CALL(*snpVectorFactoryMock, constructSNPVector(_,_)).Times(1).WillRepeatedly(
      DoAll(SaveArg<1>(&originalSNPData), Return(snpVectorMock)));

  EXPECT_CALL(*alleleStatisticsFactoryMock, constructAlleleStatistics(_)).Times(1).WillRepeatedly(
      DoAll(SaveArg<0>(&numberOfAlleles), Return(alleleStatisticsMock)));

  EXPECT_CALL(*alleleStatisticsMock, getAlleleFrequencies()).Times(2).WillRepeatedly(ReturnRef(alleleFrequencies));

  std::pair<const AlleleStatistics*, Container::SNPVector*>* pair = bedReader.readSNP(snp1);
  ASSERT_EQ(snpVectorMock, pair->second);
  ASSERT_EQ(alleleStatisticsMock, pair->first);
  delete snpVectorMock;
  delete alleleStatisticsMock;
  delete pair;

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

/*


 TEST_F(SNPVectorFactoryTest, RiskAllele_Equal_Case_Control) {
 SNPVectorFactory snpVectorFactory(configMock);

 SNP snp(Id("snp1"), "a1", "a2", 1);
 std::vector<double> alleleFreqs(6);
 alleleFreqs[ALLELE_ONE_CASE_POSITION] = 1;
 alleleFreqs[ALLELE_TWO_CASE_POSITION] = 1;
 alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 1;
 alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 1;
 alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
 alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;
 snpVectorFactory.setSNPRiskAllele(snp, alleleFreqs);

 EXPECT_EQ(ALLELE_ONE, snp.getRiskAllele());
 }

 TEST_F(SNPVectorFactoryTest, RiskAllele_Equal_Case_Control1_larger_Control2) {
 SNPVectorFactory snpVectorFactory(configMock);

 SNP snp(Id("snp1"), "a1", "a2", 1);
 std::vector<double> alleleFreqs(6);
 alleleFreqs[ALLELE_ONE_CASE_POSITION] = 1;
 alleleFreqs[ALLELE_TWO_CASE_POSITION] = 1;
 alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 2;
 alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 1;
 alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
 alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;
 snpVectorFactory.setSNPRiskAllele(snp, alleleFreqs);

 EXPECT_EQ(ALLELE_TWO, snp.getRiskAllele());
 }

 TEST_F(SNPVectorFactoryTest, RiskAllele_Equal_Case_Control2_larger_Control1) {
 SNPVectorFactory snpVectorFactory(configMock);

 SNP snp(Id("snp1"), "a1", "a2", 1);
 std::vector<double> alleleFreqs(6);
 alleleFreqs[ALLELE_ONE_CASE_POSITION] = 1;
 alleleFreqs[ALLELE_TWO_CASE_POSITION] = 1;
 alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 1;
 alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 2;
 alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
 alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;
 snpVectorFactory.setSNPRiskAllele(snp, alleleFreqs);

 EXPECT_EQ(ALLELE_ONE, snp.getRiskAllele());
 }

 TEST_F(SNPVectorFactoryTest, RiskAllele_Case1_Larger_Case2_Case1_Larger_Control1) {
 SNPVectorFactory snpVectorFactory(configMock);

 SNP snp(Id("snp1"), "a1", "a2", 1);
 std::vector<double> alleleFreqs(6);
 alleleFreqs[ALLELE_ONE_CASE_POSITION] = 2;
 alleleFreqs[ALLELE_TWO_CASE_POSITION] = 1;
 alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 1;
 alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 1;
 alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
 alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;
 snpVectorFactory.setSNPRiskAllele(snp, alleleFreqs);

 EXPECT_EQ(ALLELE_ONE, snp.getRiskAllele());
 }

 TEST_F(SNPVectorFactoryTest, RiskAllele_Case1_Larger_Case2_Case1_Smaller_Control1) {
 SNPVectorFactory snpVectorFactory(configMock);

 SNP snp(Id("snp1"), "a1", "a2", 1);
 std::vector<double> alleleFreqs(6);
 alleleFreqs[ALLELE_ONE_CASE_POSITION] = 2;
 alleleFreqs[ALLELE_TWO_CASE_POSITION] = 1;
 alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 3;
 alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 1;
 alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
 alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;
 snpVectorFactory.setSNPRiskAllele(snp, alleleFreqs);

 EXPECT_EQ(ALLELE_TWO, snp.getRiskAllele());
 }

 TEST_F(SNPVectorFactoryTest, RiskAllele_Case2_Larger_Case1_Case2_Larger_Control2) {
 SNPVectorFactory snpVectorFactory(configMock);

 SNP snp(Id("snp1"), "a1", "a2", 1);
 std::vector<double> alleleFreqs(6);
 alleleFreqs[ALLELE_ONE_CASE_POSITION] = 1;
 alleleFreqs[ALLELE_TWO_CASE_POSITION] = 2;
 alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 1;
 alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 1;
 alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
 alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;
 snpVectorFactory.setSNPRiskAllele(snp, alleleFreqs);

 EXPECT_EQ(ALLELE_TWO, snp.getRiskAllele());
 }

 TEST_F(SNPVectorFactoryTest, RiskAllele_Case2_Larger_Case1_Case2_Smaller_Control2) {
 SNPVectorFactory snpVectorFactory(configMock);

 SNP snp(Id("snp1"), "a1", "a2", 1);
 std::vector<double> alleleFreqs(6);
 alleleFreqs[ALLELE_ONE_CASE_POSITION] = 1;
 alleleFreqs[ALLELE_TWO_CASE_POSITION] = 2;
 alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 1;
 alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 3;
 alleleFreqs[ALLELE_ONE_ALL_POSITION] = 1;
 alleleFreqs[ALLELE_TWO_ALL_POSITION] = 1;
 snpVectorFactory.setSNPRiskAllele(snp, alleleFreqs);

 EXPECT_EQ(ALLELE_ONE, snp.getRiskAllele());
 }

 TEST_F(SNPVectorFactoryTest, SNPInclude_MissingData) {
 SNPVectorFactory snpVectorFactory(configMock);

 SNP snp(Id("snp1"), "a1", "a2", 1);
 std::vector<double> alleleFreqs(6);
 std::vector<int> alleleNumbers(6);
 bool missingData = true;

 snpVectorFactory.setSNPInclude(snp, alleleFreqs, alleleNumbers, missingData);
 EXPECT_FALSE(snp.getInclude());
 }

 TEST_F(SNPVectorFactoryTest, SNPInclude_True) {
 SNPVectorFactory snpVectorFactory(configMock);

 SNP snp(Id("snp1"), "a1", "a2", 1);
 std::vector<double> alleleFreqs(6);
 std::vector<int> alleleNumbers(6);
 bool missingData = false;

 alleleFreqs[ALLELE_ONE_CASE_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_CASE_POSITION] = 0.5;
 alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 0.5;
 alleleFreqs[ALLELE_ONE_ALL_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_ALL_POSITION] = 0.5;

 alleleNumbers[ALLELE_ONE_CASE_POSITION] = 100;
 alleleNumbers[ALLELE_TWO_CASE_POSITION] = 100;
 alleleNumbers[ALLELE_ONE_CONTROL_POSITION] = 100;
 alleleNumbers[ALLELE_TWO_CONTROL_POSITION] = 100;
 alleleNumbers[ALLELE_ONE_ALL_POSITION] = 200;
 alleleNumbers[ALLELE_TWO_ALL_POSITION] = 200;

 snpVectorFactory.setSNPInclude(snp, alleleFreqs, alleleNumbers, missingData);
 EXPECT_TRUE(snp.getInclude());
 }

 TEST_F(SNPVectorFactoryTest, SNPInclude_ToLowAbsFreq_1) {
 SNPVectorFactory snpVectorFactory(configMock);

 SNP snp(Id("snp1"), "a1", "a2", 1);
 std::vector<double> alleleFreqs(6);
 std::vector<int> alleleNumbers(6);
 bool missingData = false;

 alleleFreqs[ALLELE_ONE_CASE_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_CASE_POSITION] = 0.5;
 alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 0.5;
 alleleFreqs[ALLELE_ONE_ALL_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_ALL_POSITION] = 0.5;

 alleleNumbers[ALLELE_ONE_CASE_POSITION] = 1;
 alleleNumbers[ALLELE_TWO_CASE_POSITION] = 100;
 alleleNumbers[ALLELE_ONE_CONTROL_POSITION] = 100;
 alleleNumbers[ALLELE_TWO_CONTROL_POSITION] = 100;
 alleleNumbers[ALLELE_ONE_ALL_POSITION] = 200;
 alleleNumbers[ALLELE_TWO_ALL_POSITION] = 200;

 snpVectorFactory.setSNPInclude(snp, alleleFreqs, alleleNumbers, missingData);
 EXPECT_FALSE(snp.getInclude());
 }

 TEST_F(SNPVectorFactoryTest, SNPInclude_ToLowAbsFreq_2) {
 SNPVectorFactory snpVectorFactory(configMock);

 SNP snp(Id("snp1"), "a1", "a2", 1);
 std::vector<double> alleleFreqs(6);
 std::vector<int> alleleNumbers(6);
 bool missingData = false;

 alleleFreqs[ALLELE_ONE_CASE_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_CASE_POSITION] = 0.5;
 alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 0.5;
 alleleFreqs[ALLELE_ONE_ALL_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_ALL_POSITION] = 0.5;

 alleleNumbers[ALLELE_ONE_CASE_POSITION] = 100;
 alleleNumbers[ALLELE_TWO_CASE_POSITION] = 1;
 alleleNumbers[ALLELE_ONE_CONTROL_POSITION] = 100;
 alleleNumbers[ALLELE_TWO_CONTROL_POSITION] = 100;
 alleleNumbers[ALLELE_ONE_ALL_POSITION] = 200;
 alleleNumbers[ALLELE_TWO_ALL_POSITION] = 200;

 snpVectorFactory.setSNPInclude(snp, alleleFreqs, alleleNumbers, missingData);
 EXPECT_FALSE(snp.getInclude());
 }

 TEST_F(SNPVectorFactoryTest, SNPInclude_ToLowAbsFreq_3) {
 SNPVectorFactory snpVectorFactory(configMock);

 SNP snp(Id("snp1"), "a1", "a2", 1);
 std::vector<double> alleleFreqs(6);
 std::vector<int> alleleNumbers(6);
 bool missingData = false;

 alleleFreqs[ALLELE_ONE_CASE_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_CASE_POSITION] = 0.5;
 alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 0.5;
 alleleFreqs[ALLELE_ONE_ALL_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_ALL_POSITION] = 0.5;

 alleleNumbers[ALLELE_ONE_CASE_POSITION] = 100;
 alleleNumbers[ALLELE_TWO_CASE_POSITION] = 100;
 alleleNumbers[ALLELE_ONE_CONTROL_POSITION] = 1;
 alleleNumbers[ALLELE_TWO_CONTROL_POSITION] = 100;
 alleleNumbers[ALLELE_ONE_ALL_POSITION] = 200;
 alleleNumbers[ALLELE_TWO_ALL_POSITION] = 200;

 snpVectorFactory.setSNPInclude(snp, alleleFreqs, alleleNumbers, missingData);
 EXPECT_FALSE(snp.getInclude());
 }

 TEST_F(SNPVectorFactoryTest, SNPInclude_ToLowAbsFreq_4) {
 SNPVectorFactory snpVectorFactory(configMock);

 SNP snp(Id("snp1"), "a1", "a2", 1);
 std::vector<double> alleleFreqs(6);
 std::vector<int> alleleNumbers(6);
 bool missingData = false;

 alleleFreqs[ALLELE_ONE_CASE_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_CASE_POSITION] = 0.5;
 alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 0.5;
 alleleFreqs[ALLELE_ONE_ALL_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_ALL_POSITION] = 0.5;

 alleleNumbers[ALLELE_ONE_CASE_POSITION] = 100;
 alleleNumbers[ALLELE_TWO_CASE_POSITION] = 100;
 alleleNumbers[ALLELE_ONE_CONTROL_POSITION] = 100;
 alleleNumbers[ALLELE_TWO_CONTROL_POSITION] = 1;
 alleleNumbers[ALLELE_ONE_ALL_POSITION] = 200;
 alleleNumbers[ALLELE_TWO_ALL_POSITION] = 200;

 snpVectorFactory.setSNPInclude(snp, alleleFreqs, alleleNumbers, missingData);
 EXPECT_FALSE(snp.getInclude());
 }

 TEST_F(SNPVectorFactoryTest, SNPInclude_ToLowMAF_Equal) {
 SNPVectorFactory snpVectorFactory(configMock);

 SNP snp(Id("snp1"), "a1", "a2", 1);
 std::vector<double> alleleFreqs(6);
 std::vector<int> alleleNumbers(6);
 bool missingData = false;

 alleleFreqs[ALLELE_ONE_CASE_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_CASE_POSITION] = 0.5;
 alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 0.5;
 alleleFreqs[ALLELE_ONE_ALL_POSITION] = 0.01;
 alleleFreqs[ALLELE_TWO_ALL_POSITION] = 0.01;

 alleleNumbers[ALLELE_ONE_CASE_POSITION] = 100;
 alleleNumbers[ALLELE_TWO_CASE_POSITION] = 100;
 alleleNumbers[ALLELE_ONE_CONTROL_POSITION] = 100;
 alleleNumbers[ALLELE_TWO_CONTROL_POSITION] = 100;
 alleleNumbers[ALLELE_ONE_ALL_POSITION] = 200;
 alleleNumbers[ALLELE_TWO_ALL_POSITION] = 200;

 snpVectorFactory.setSNPInclude(snp, alleleFreqs, alleleNumbers, missingData);
 EXPECT_FALSE(snp.getInclude());
 }

 TEST_F(SNPVectorFactoryTest, SNPInclude_ToLowMAF_1Larger2) {
 SNPVectorFactory snpVectorFactory(configMock);

 SNP snp(Id("snp1"), "a1", "a2", 1);
 std::vector<double> alleleFreqs(6);
 std::vector<int> alleleNumbers(6);
 bool missingData = false;

 alleleFreqs[ALLELE_ONE_CASE_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_CASE_POSITION] = 0.5;
 alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 0.5;
 alleleFreqs[ALLELE_ONE_ALL_POSITION] = 0.02;
 alleleFreqs[ALLELE_TWO_ALL_POSITION] = 0.01;

 alleleNumbers[ALLELE_ONE_CASE_POSITION] = 100;
 alleleNumbers[ALLELE_TWO_CASE_POSITION] = 100;
 alleleNumbers[ALLELE_ONE_CONTROL_POSITION] = 100;
 alleleNumbers[ALLELE_TWO_CONTROL_POSITION] = 100;
 alleleNumbers[ALLELE_ONE_ALL_POSITION] = 200;
 alleleNumbers[ALLELE_TWO_ALL_POSITION] = 200;

 snpVectorFactory.setSNPInclude(snp, alleleFreqs, alleleNumbers, missingData);
 EXPECT_FALSE(snp.getInclude());
 }

 TEST_F(SNPVectorFactoryTest, SNPInclude_ToLowMAF_2Larger1) {
 SNPVectorFactory snpVectorFactory(configMock);

 SNP snp(Id("snp1"), "a1", "a2", 1);
 std::vector<double> alleleFreqs(6);
 std::vector<int> alleleNumbers(6);
 bool missingData = false;

 alleleFreqs[ALLELE_ONE_CASE_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_CASE_POSITION] = 0.5;
 alleleFreqs[ALLELE_ONE_CONTROL_POSITION] = 0.5;
 alleleFreqs[ALLELE_TWO_CONTROL_POSITION] = 0.5;
 alleleFreqs[ALLELE_ONE_ALL_POSITION] = 0.01;
 alleleFreqs[ALLELE_TWO_ALL_POSITION] = 0.02;

 alleleNumbers[ALLELE_ONE_CASE_POSITION] = 100;
 alleleNumbers[ALLELE_TWO_CASE_POSITION] = 100;
 alleleNumbers[ALLELE_ONE_CONTROL_POSITION] = 100;
 alleleNumbers[ALLELE_TWO_CONTROL_POSITION] = 100;
 alleleNumbers[ALLELE_ONE_ALL_POSITION] = 200;
 alleleNumbers[ALLELE_TWO_ALL_POSITION] = 200;

 snpVectorFactory.setSNPInclude(snp, alleleFreqs, alleleNumbers, missingData);
 EXPECT_FALSE(snp.getInclude());
 }









 */
}
/* namespace FileIO */
} /* namespace CuEira */

