#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>

#include <BedReader.h>
#include <BimReader.h>
#include <FamReader.h>
#include <DataFilesReader.h>
#include <DataFilesReaderFactory.h>
#include <ConfigurationMock.h>
#include <GeneticModel.h>
#include <Person.h>
#include <Sex.h>
#include <Id.h>
#include <Phenotype.h>
#include <SNP.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <FileReaderException.h>
#include <EnvironmentFactorHandler.h>
#include <EnvironmentFactor.h>

using testing::Return;
using testing::_;
using testing::ReturnRef;
using testing::Eq;
using testing::AtLeast;

namespace CuEira {
namespace FileIO {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class FileIOIntegrationTest: public ::testing::Test {
protected:
  FileIOIntegrationTest();
  virtual ~FileIOIntegrationTest();
  virtual void SetUp();
  virtual void TearDown();

  static const int numberOfIndividualsTotalStatic = 10;
  static const int numberOfIndividualsToIncludeStatic = 9;
  const int numberOfIndividualsTotal;
  const int numberOfIndividualsToInclude;
  ConfigurationMock configMock;
};

FileIOIntegrationTest::FileIOIntegrationTest() :
    numberOfIndividualsTotal(numberOfIndividualsTotalStatic), numberOfIndividualsToInclude(
        numberOfIndividualsToIncludeStatic) {

}

FileIOIntegrationTest::~FileIOIntegrationTest() {

}

void FileIOIntegrationTest::SetUp() {
  //Expect Configuration
  EXPECT_CALL(configMock, getBimFilePath()).Times(AtLeast(1)).WillRepeatedly(
      Return(std::string(CuEira_BUILD_DIR) + std::string("/test.bim")));
  EXPECT_CALL(configMock, getFamFilePath()).Times(AtLeast(1)).WillRepeatedly(
      Return(std::string(CuEira_BUILD_DIR) + std::string("/test.fam")));
  EXPECT_CALL(configMock, getCSVFilePath()).Times(AtLeast(1)).WillRepeatedly(
      Return(std::string(CuEira_BUILD_DIR) + std::string("/test_env.txt")));

  EXPECT_CALL(configMock, getEnvironmentColumnName()).Times(AtLeast(1)).WillRepeatedly(Return("env1"));
  EXPECT_CALL(configMock, getCSVIdColumnName()).Times(AtLeast(1)).WillRepeatedly(Return("indid"));
  EXPECT_CALL(configMock, getCSVDelimiter()).Times(AtLeast(1)).WillRepeatedly(Return("\t "));
}

void FileIOIntegrationTest::TearDown() {

}

TEST_F(FileIOIntegrationTest, ReadPersonInformation) {
  EXPECT_CALL(configMock, getPhenotypeCoding()).Times(AtLeast(1)).WillRepeatedly(Return(ONE_TWO_CODING));

  DataFilesReaderFactory dataFilesReaderFactory;
  DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configMock);

  const PersonHandler& personHandler = dataFilesReader->getPersonHandler();

  EXPECT_EQ(numberOfIndividualsTotal, personHandler.getNumberOfIndividualsTotal());
  EXPECT_EQ(numberOfIndividualsToInclude, personHandler.getNumberOfIndividualsToInclude());

  const std::vector<Person*>& persons = personHandler.getPersons();

  //TODO check stuff
  for(auto person : persons){

  }

  delete dataFilesReader;
}

TEST_F(FileIOIntegrationTest, ReadSNPInfo) {
  EXPECT_CALL(configMock, excludeSNPsWithNegativePosition()).Times(AtLeast(1)).WillRepeatedly(Return(true));
  EXPECT_CALL(configMock, getPhenotypeCoding()).Times(1).WillRepeatedly(Return(ZERO_ONE_CODING));

  DataFilesReaderFactory dataFilesReaderFactory;
  DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configMock);

  std::vector<SNP*>* snpInformation = dataFilesReader->readSNPInformation();

  int numSNPToInclude = 0;
  int snpSize = snpInformation->size();
  for(int i = 0; i < snpSize; ++i){
    SNP* snp = (*snpInformation)[i];
    if(snp->shouldInclude()){
      ++numSNPToInclude;
    }
  }

  EXPECT_EQ(10, snpSize);
  EXPECT_EQ(8, numSNPToInclude);

  delete dataFilesReader;

  for(int i = 0; i < snpSize; ++i){
    delete (*snpInformation)[i];
  }
  delete snpInformation;
}

TEST_F(FileIOIntegrationTest, ReadCovariates) {
  EXPECT_CALL(configMock, getPhenotypeCoding()).Times(AtLeast(1)).WillRepeatedly(Return(ONE_TWO_CODING));

  DataFilesReaderFactory dataFilesReaderFactory;
  DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configMock);
  Container::HostMatrix* covariatesData = dataFilesReader->readCovariates();

  int covNumberOfRows = covariatesData->getNumberOfRows();
  ASSERT_EQ(numberOfIndividualsToInclude, covNumberOfRows);
  ASSERT_EQ(2, covariatesData->getNumberOfColumns());

  EXPECT_EQ(1, (*covariatesData)(0, 0));
  EXPECT_EQ(2, (*covariatesData)(1, 0));
  EXPECT_EQ(0, (*covariatesData)(2, 0));
  EXPECT_EQ(0, (*covariatesData)(3, 0));
  EXPECT_EQ(1, (*covariatesData)(4, 0));
  EXPECT_EQ(1, (*covariatesData)(5, 0));
  EXPECT_EQ(1, (*covariatesData)(6, 0));
  EXPECT_EQ(1, (*covariatesData)(7, 0));
  EXPECT_EQ(0, (*covariatesData)(8, 0));

  EXPECT_EQ(1, (*covariatesData)(0, 1));
  EXPECT_EQ(0, (*covariatesData)(1, 1));
  EXPECT_EQ(1, (*covariatesData)(2, 1));
  EXPECT_EQ(1, (*covariatesData)(3, 1));
  EXPECT_EQ(2, (*covariatesData)(4, 1));
  EXPECT_EQ(0, (*covariatesData)(5, 1));
  EXPECT_EQ(1, (*covariatesData)(6, 1));
  EXPECT_EQ(0, (*covariatesData)(7, 1));
  EXPECT_EQ(2, (*covariatesData)(8, 1));

  delete covariatesData;
  delete dataFilesReader;
}

TEST_F(FileIOIntegrationTest, ReadEnvironment) {
  EXPECT_CALL(configMock, getPhenotypeCoding()).Times(AtLeast(1)).WillRepeatedly(Return(ONE_TWO_CODING));

  DataFilesReaderFactory dataFilesReaderFactory;
  DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configMock);
  EnvironmentFactorHandler* environmentFactorHandler = dataFilesReader->readEnvironmentFactorInformation();

  Id id("env1");
  EnvironmentFactor envFactor(id);

  const Container::HostVector* envData = environmentFactorHandler->getData(envFactor);
  const std::vector<const EnvironmentFactor*>& envInfo = environmentFactorHandler->getHeaders();

  ASSERT_EQ(numberOfIndividualsToInclude, envData->getNumberOfRows());
  ASSERT_EQ(2, envInfo.size());

  ASSERT_EQ(BINARY, envInfo[0]->getVariableType());
  ASSERT_EQ(OTHER, envInfo[1]->getVariableType());

  EXPECT_EQ(Id("env1"), envInfo[0]->getId());
  EXPECT_EQ(Id("env2"), envInfo[1]->getId());

  EXPECT_EQ(1, (*envData)(0));
  EXPECT_EQ(1, (*envData)(1));
  EXPECT_EQ(0, (*envData)(2));
  EXPECT_EQ(0, (*envData)(3));
  EXPECT_EQ(1, (*envData)(4));
  EXPECT_EQ(1, (*envData)(5));
  EXPECT_EQ(1, (*envData)(6));
  EXPECT_EQ(1, (*envData)(7));
  EXPECT_EQ(0, (*envData)(8));

  delete envData;
  delete dataFilesReader;
  delete environmentFactorHandler;
}

//TODO bed stuff

}
/* namespace FileIO */
} /* namespace CuEira */

