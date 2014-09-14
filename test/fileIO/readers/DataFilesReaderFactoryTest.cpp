#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>

#include <DataFilesReader.h>
#include <DataFilesReaderFactory.h>
#include <ConfigurationMock.h>
#include <Phenotype.h>
#include <FileReaderException.h>

using testing::Return;
using testing::_;
using testing::AtLeast;

namespace CuEira {
namespace FileIO {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DataFilesReaderFactoryTest: public ::testing::Test {
protected:
  DataFilesReaderFactoryTest();
  virtual ~DataFilesReaderFactoryTest();
  virtual void SetUp();
  virtual void TearDown();

  ConfigurationMock configMock;
};

DataFilesReaderFactoryTest::DataFilesReaderFactoryTest() {

}

DataFilesReaderFactoryTest::~DataFilesReaderFactoryTest() {

}

void DataFilesReaderFactoryTest::SetUp() {
  EXPECT_CALL(configMock, getBimFilePath()).Times(AtLeast(1)).WillRepeatedly(
      Return(std::string(CuEira_BUILD_DIR) + std::string("/test.bim")));
  EXPECT_CALL(configMock, getFamFilePath()).Times(AtLeast(1)).WillRepeatedly(
      Return(std::string(CuEira_BUILD_DIR) + std::string("/test.fam")));
  EXPECT_CALL(configMock, getEnvironmentFilePath()).Times(AtLeast(1)).WillRepeatedly(
      Return(std::string(CuEira_BUILD_DIR) + std::string("/test_env.txt")));

  EXPECT_CALL(configMock, getEnvironmentIndividualIdColumnName()).Times(AtLeast(1)).WillRepeatedly(Return("indid"));
  EXPECT_CALL(configMock, getEnvironmentDelimiter()).Times(AtLeast(1)).WillRepeatedly(Return("\t "));
}

void DataFilesReaderFactoryTest::TearDown() {

}

TEST_F(DataFilesReaderFactoryTest, ConstructWithCov) {
  EXPECT_CALL(configMock, covariateFileSpecified()).Times(1).WillRepeatedly(Return(true));

  EXPECT_CALL(configMock, getCovariateFilePath()).Times(1).WillRepeatedly(
      Return(std::string(CuEira_BUILD_DIR) + std::string("/test_cov.txt")));
  EXPECT_CALL(configMock, getCovariateIndividualIdColumnName()).Times(1).WillRepeatedly(Return("indid"));
  EXPECT_CALL(configMock, getCovariateDelimiter()).Times(1).WillRepeatedly(Return("\t "));
  EXPECT_CALL(configMock, getPhenotypeCoding()).Times(1).WillRepeatedly(Return(ZERO_ONE_CODING));

  DataFilesReaderFactory dataFilesReaderFactory;
  DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configMock);

  delete dataFilesReader;
}

TEST_F(DataFilesReaderFactoryTest, ConstructWithoutCov) {
  EXPECT_CALL(configMock, covariateFileSpecified()).Times(1).WillRepeatedly(Return(false));
  EXPECT_CALL(configMock, getPhenotypeCoding()).Times(AtLeast(1)).WillRepeatedly(Return(ONE_TWO_CODING));

  DataFilesReaderFactory dataFilesReaderFactory;
  DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configMock);
  PersonHandler* personHandler = dataFilesReader->readPersonInformation();

  EXPECT_THROW(dataFilesReader->readCovariates(*personHandler), FileReaderException);

  delete personHandler;
  delete dataFilesReader;
}

}
/* namespace FileIO */
} /* namespace CuEira */

