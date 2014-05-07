#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#include <CSVReader.h>
#include <PersonHandlerMock.h>
#include <Id.h>
#include <Sex.h>
#include <Phenotype.h>
#include <Person.h>
#include <FileReaderException.h>

using testing::Return;
using testing::_;

namespace CuEira {
namespace FileIO {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CSVReaderTest: public ::testing::Test {
protected:
  CSVReaderTest();
  virtual ~CSVReaderTest();
  virtual void SetUp();
  virtual void TearDown();

  int numberOfIndividuals;
  PersonHandlerMock personHandlerMock;
  std::string filePath;
  std::string delimiter;
  std::string idColumnName;
};

CSVReaderTest::CSVReaderTest() :
    numberOfIndividuals(10), filePath("../data/test_cov.txt"), delimiter("\t "), idColumnName("indid") {

}

CSVReaderTest::~CSVReaderTest() {

}

void CSVReaderTest::SetUp() {
  numberOfIndividuals = 10;
}

void CSVReaderTest::TearDown() {

}

TEST_F(CSVReaderTest, ReadFile) {

  CuEira::FileIO::CSVReader CSVReader(filePath, idColumnName, delimiter, personHandlerMock);
}

TEST_F(CSVReaderTest, ReadFileWrongNumber) {
  filePath="../data/test_csv_wrong_number.txt";

  CuEira::FileIO::CSVReader CSVReader(filePath, idColumnName, delimiter, personHandlerMock);
}

}
/* namespace FileIO */
} /* namespace CuEira */

