#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>
#include <utility>

#include <BedReader.h>
#include <ConfigurationMock.h>
#include <PersonHandlerMock.h>
#include <GeneticModel.h>
#include <Person.h>
#include <Sex.h>
#include <Id.h>
#include <Phenotype.h>
#include <SNP.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <FileReaderException.h>
#include <SNPVectorMock.h>
#include <SNPVectorFactoryMock.h>

using testing::Return;
using testing::_;
using testing::ReturnRef;
using testing::Eq;
using testing::SaveArg;
using testing::DoAll;
using testing::ByRef;

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

  const int numberOfIndividualsTotal;
  const int numberOfIndividualsToInclude;
  const int numberOfSNPs = 10;

  std::string filePath;
  std::vector<int> notInclude; //Index 0 based

  PersonHandlerMock personHandlerMock;
  ConfigurationMock configMock;
  Container::SNPVectorFactoryMock* snpVectorFactoryMock;
};

BedReaderTest::BedReaderTest() :
    filePath(std::string(CuEira_BUILD_DIR) + std::string("/test.bed")), numberOfIndividualsTotal(10), numberOfIndividualsToInclude(
        6), notInclude(4), snpVectorFactoryMock(nullptr) {
  notInclude[0] = 1;
  notInclude[1] = 2;
  notInclude[2] = 5;
  notInclude[3] = 7;
}

BedReaderTest::~BedReaderTest() {

}

void BedReaderTest::SetUp() {
  EXPECT_CALL(configMock, getGeneticModel()).WillRepeatedly(Return(DOMINANT));

  //Expect Configuration
  EXPECT_CALL(configMock, getBedFilePath()).Times(1).WillRepeatedly(Return(filePath));
  EXPECT_CALL(configMock, getMinorAlleleFrequencyThreshold()).Times(1).WillRepeatedly(Return(0));

  //Expect PersonHandler
  EXPECT_CALL(personHandlerMock, getNumberOfIndividualsTotal()).WillRepeatedly(Return(numberOfIndividualsTotal));
  EXPECT_CALL(personHandlerMock, getNumberOfIndividualsToInclude()).WillRepeatedly(
      Return(numberOfIndividualsToInclude));

  snpVectorFactoryMock = new Container::SNPVectorFactoryMock(configMock);
}

void BedReaderTest::TearDown() {

}

TEST_F(BedReaderTest, ConstructorCheckMode) {
  CuEira::FileIO::BedReader bedReader(configMock, snpVectorFactoryMock, personHandlerMock, numberOfSNPs);

  ASSERT_EQ(0, bedReader.mode);
}

TEST_F(BedReaderTest, ReadSnp0) {
  std::vector<Person*> persons(numberOfIndividualsTotal);

  int j = 0;
  int includePos = 0;
  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    std::ostringstream os;
    os << "ind" << i;
    Id id(os.str());
    Sex sex;
    Phenotype phenotype;

    if(rand() % 2 == 0){
      sex = MALE;
    }else{
      sex = FEMALE;
    }

    if(rand() % 2 == 0){
      phenotype = AFFECTED;
    }else{
      phenotype = UNAFFECTED;
    }

    Person* person;
    if(i == notInclude[j]){
      ++j;
      person = new Person(id, sex, MISSING, false);
      includePosArr[i] = -1;
    }else{
      person = new Person(id, sex, phenotype, true);
      includePosArr[i] = includePos;
      ++includePos;
    }
    persons[i] = person;

    EXPECT_CALL(personHandlerMock, getPersonFromRowAll(i)).WillRepeatedly(ReturnRef(*person));
  }

  CuEira::FileIO::BedReader bedReader(configMock, snpVectorFactoryMock, personHandlerMock, numberOfSNPs);

  Id id1("SNP1");
  unsigned int pos1 = 0; //First SNP
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  Container::SNPVectorMock* snpVectorMock = new Container::SNPVectorMock(snp1);
  Container::HostVector* snpDataOriginal;
  std::set<int>* snpMissingData;
  SNP* snpReturn;

  EXPECT_CALL(*snpVectorFactoryMock, constructSNPVector(snp1,_,_)).Times(1).WillRepeatedly(
      DoAll(SaveArg<1>(&originalSNPData), SaveArg<2>(&snpMissingData), Return(snpVectorMock)));

  Container::SNPVector* snpVector = bedReader.readSNP(snp1);
  ASSERT_EQ(snpVectorMock, snpVectorMock);
  ASSERT_EQ(numberOfIndividualsToInclude, snpDataOriginal->getNumberOfRows());
  ASSERT_EQ(0, snpMissingData->size());

  EXPECT_EQ(0, (*snpDataOriginal)[0]);
  EXPECT_EQ(2, (*snpDataOriginal)[1]);
  EXPECT_EQ(1, (*snpDataOriginal)[2]);
  EXPECT_EQ(1, (*snpDataOriginal)[3]);
  EXPECT_EQ(1, (*snpDataOriginal)[4]);
  EXPECT_EQ(1, (*snpDataOriginal)[5]);

  delete snpVectorMock;
  delete originalSNPData;
  delete snpMissingData;

  for(auto person : persons){
    delete person;
  }
}

TEST_F(BedReaderTest, ReadSnp1) {
  std::vector<Person*> persons(numberOfIndividualsTotal);

  int j = 0;
  int includePos = 0;
  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    std::ostringstream os;
    os << "ind" << i;
    Id id(os.str());
    Sex sex;
    Phenotype phenotype;

    if(rand() % 2 == 0){
      sex = MALE;
    }else{
      sex = FEMALE;
    }

    if(rand() % 2 == 0){
      phenotype = AFFECTED;
    }else{
      phenotype = UNAFFECTED;
    }

    Person* person;
    if(i == notInclude[j]){
      ++j;
      person = new Person(id, sex, MISSING, false);
      includePosArr[i] = -1;
    }else{
      person = new Person(id, sex, phenotype, true);
      includePosArr[i] = includePos;
      ++includePos;
    }
    persons[i] = person;

    EXPECT_CALL(personHandlerMock, getPersonFromRowAll(i)).WillRepeatedly(ReturnRef(*person));
  }

  CuEira::FileIO::BedReader bedReader(configMock, snpVectorFactoryMock, personHandlerMock, numberOfSNPs);

  Id id1("SNP1");
  unsigned int pos1 = 1; //Second SNP
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  Container::SNPVectorMock* snpVectorMock = new Container::SNPVectorMock(snp1);
  Container::HostVector* snpDataOriginal;
  std::set<int>* snpMissingData;
  SNP* snpReturn;

  EXPECT_CALL(*snpVectorFactoryMock, constructSNPVector(snp1,_,_)).Times(1).WillRepeatedly(
      DoAll(SaveArg<1>(&originalSNPData), SaveArg<2>(&snpMissingData), Return(snpVectorMock)));

  Container::SNPVector* snpVector = bedReader.readSNP(snp1);
  ASSERT_EQ(snpVectorMock, snpVectorMock);
  ASSERT_EQ(numberOfIndividualsToInclude, snpDataOriginal->getNumberOfRows());
  ASSERT_EQ(0, snpMissingData->size());

  EXPECT_EQ(2, (*snpDataOriginal)[0]);
  EXPECT_EQ(2, (*snpDataOriginal)[1]);
  EXPECT_EQ(1, (*snpDataOriginal)[2]);
  EXPECT_EQ(2, (*snpDataOriginal)[3]);
  EXPECT_EQ(2, (*snpDataOriginal)[4]);
  EXPECT_EQ(0, (*snpDataOriginal)[5]);

  delete snpVectorMock;
  delete originalSNPData;
  delete snpMissingData;

  for(auto person : persons){
    delete person;
  }
}

TEST_F(BedReaderTest, ReadSnp9Missing) {
  std::vector<Person*> persons(numberOfIndividualsTotal);
  //3 are missing 2 4 8, but 2 is skiped

  int j = 0;
  int includePos = 0;
  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    std::ostringstream os;
    os << "ind" << i;
    Id id(os.str());
    Sex sex;
    Phenotype phenotype;

    if(rand() % 2 == 0){
      sex = MALE;
    }else{
      sex = FEMALE;
    }

    if(rand() % 2 == 0){
      phenotype = AFFECTED;
    }else{
      phenotype = UNAFFECTED;
    }

    Person* person;
    if(i == notInclude[j]){
      ++j;
      person = new Person(id, sex, MISSING, false);
      includePosArr[i] = -1;
    }else{
      person = new Person(id, sex, phenotype, true);
      includePosArr[i] = includePos;
      ++includePos;
    }
    persons[i] = person;

    EXPECT_CALL(personHandlerMock, getPersonFromRowAll(i)).WillRepeatedly(ReturnRef(*person));
  }

  CuEira::FileIO::BedReader bedReader(configMock, snpVectorFactoryMock, personHandlerMock, numberOfSNPs);

  Id id1("SNP1");
  unsigned int pos1 = 9; //Last SNP
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  Container::SNPVectorMock* snpVectorMock = new Container::SNPVectorMock(snp1);
  Container::HostVector* snpDataOriginal;
  std::set<int>* snpMissingData;
  SNP* snpReturn;

  EXPECT_CALL(*snpVectorFactoryMock, constructSNPVector(snp1,_,_)).Times(1).WillRepeatedly(
      DoAll(SaveArg<1>(&originalSNPData), SaveArg<2>(&snpMissingData), Return(snpVectorMock)));

  Container::SNPVector* snpVector = bedReader.readSNP(snp1);
  ASSERT_EQ(snpVectorMock, snpVectorMock);
  ASSERT_EQ(numberOfIndividualsToInclude, snpDataOriginal->getNumberOfRows());
  ASSERT_EQ(2, snpMissingData->size());

  ASSERT_EQ(1, snpMissingData->count(4));
  ASSERT_EQ(1, snpMissingData->count(8));

  EXPECT_EQ(1, (*snpDataOriginal)[0]);
  EXPECT_EQ(0, (*snpDataOriginal)[1]);
  EXPECT_EQ(-1, (*snpDataOriginal)[2]);
  EXPECT_EQ(2, (*snpDataOriginal)[3]);
  EXPECT_EQ(-1, (*snpDataOriginal)[4]);
  EXPECT_EQ(2, (*snpDataOriginal)[5]);

  delete snpVectorMock;
  delete originalSNPData;
  delete snpMissingData;

  for(auto person : persons){
    delete person;
  }
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

