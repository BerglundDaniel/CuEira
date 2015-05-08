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
 * Test for testing DataFilesReaderFactory
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DataFilesReaderFactoryTest: public ::testing::Test {
protected:
  DataFilesReaderFactoryTest();
  virtual ~DataFilesReaderFactoryTest();
  virtual void SetUp();
  virtual void TearDown();

};

DataFilesReaderFactoryTest::DataFilesReaderFactoryTest() {

}

DataFilesReaderFactoryTest::~DataFilesReaderFactoryTest() {

}

void DataFilesReaderFactoryTest::SetUp() {

}

void DataFilesReaderFactoryTest::TearDown() {

}

TEST_F(DataFilesReaderFactoryTest, Construct) {
  ConfigurationMock configMock;

  EXPECT_CALL(configMock, getBimFilePath()).Times(AtLeast(1)).WillRepeatedly(
      Return(std::string(CuEira_BUILD_DIR) + std::string("/test.bim")));
  EXPECT_CALL(configMock, getFamFilePath()).Times(AtLeast(1)).WillRepeatedly(
      Return(std::string(CuEira_BUILD_DIR) + std::string("/test.fam")));
  EXPECT_CALL(configMock, getEnvironmentFilePath()).Times(AtLeast(1)).WillRepeatedly(
      Return(std::string(CuEira_BUILD_DIR) + std::string("/test_env.txt")));

  EXPECT_CALL(configMock, getEnvironmentColumnName()).Times(AtLeast(1)).WillRepeatedly(Return("indid"));
  EXPECT_CALL(configMock, getCSVDelimiter()).Times(AtLeast(1)).WillRepeatedly(Return("\t "));
  EXPECT_CALL(configMock, getCSVFilePath()).Times(1).WillRepeatedly(
      Return(std::string(CuEira_BUILD_DIR) + std::string("/test_cov.txt")));
  EXPECT_CALL(configMock, getCSVIdColumnName()).Times(1).WillRepeatedly(Return("indid"));
  EXPECT_CALL(configMock, getPhenotypeCoding()).Times(1).WillRepeatedly(Return(ZERO_ONE_CODING));

  DataFilesReaderFactory dataFilesReaderFactory;
  DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configMock);

  delete dataFilesReader;
}

}
/* namespace FileIO */
} /* namespace CuEira */

