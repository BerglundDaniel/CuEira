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
#include <HostMatrix.h>
#include <HostVector.h>
#include <ConstructorHelpers.h>

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
class CSVReaderTest: public ::testing::Test {
protected:
  CSVReaderTest();
  virtual ~CSVReaderTest();
  virtual void SetUp();
  virtual void TearDown();

  static const int numberOfIndividualsTotalStatic = 10;
  static const int numberOfIndividualsToIncludeStatic = 6;
  const int numberOfIndividualsTotal;
  const int numberOfIndividualsToInclude;
  PersonHandlerMock personHandlerMock;
  CuEira_Test::ConstructorHelpers constructorHelpers;
  std::string filePath;
  std::string delimiter;
  std::string idColumnName;
  const int notInclude[4] = {1, 2, 5, 7}; //Index 0 based
  const int numberOfColumns = 2;

  Id* ids[numberOfIndividualsTotalStatic];
  Person* persons[numberOfIndividualsTotalStatic];
  int includePosArr[numberOfIndividualsTotalStatic];

};

CSVReaderTest::CSVReaderTest() :
    filePath(std::string(CuEira_BUILD_DIR)+std::string("/test_csv.txt")), delimiter("\t "), idColumnName("indid"), numberOfIndividualsTotal(
        numberOfIndividualsTotalStatic), numberOfIndividualsToInclude(numberOfIndividualsToIncludeStatic) {

}

CSVReaderTest::~CSVReaderTest() {

}

void CSVReaderTest::SetUp() {
  EXPECT_CALL(personHandlerMock, getNumberOfIndividualsTotal()).Times(1).WillRepeatedly(
      Return(numberOfIndividualsTotal));
  EXPECT_CALL(personHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillRepeatedly(
      Return(numberOfIndividualsToInclude));

  int j = 0;
  int includePos = 0;
  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    Person* person;
    if(i == notInclude[j]){
      ++j;
      person = constructorHelpers.constructPersonNotInclude(i);
      includePosArr[i] = -1;
    }else{
      person = constructorHelpers.constructPersonInclude(i);
      includePosArr[i] = includePos;
      ++includePos;
    }
    persons[i] = person;
    ids[i] = new Id(person->getId().getString());
  }

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    Id* id = ids[i];
    Person* person = persons[i];
    EXPECT_CALL(personHandlerMock, getPersonFromId(Eq(*id))).WillRepeatedly(ReturnRef(*person));
    EXPECT_CALL(personHandlerMock, getRowIncludeFromPerson(Eq(*person))).WillRepeatedly(Return(includePosArr[i]));
  }
}

void CSVReaderTest::TearDown() {
  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    delete persons[i];
    delete ids[i];
  }
}

TEST_F(CSVReaderTest, ReadFile) {
  CuEira::FileIO::CSVReader csvReader(filePath, idColumnName, delimiter, personHandlerMock);

  ASSERT_EQ(numberOfIndividualsTotal, csvReader.getNumberOfRows());
  ASSERT_EQ(numberOfColumns, csvReader.getNumberOfColumns());

  const std::vector<std::string>& columnNames = csvReader.getDataColumnHeaders();
  ASSERT_TRUE("cov1" == columnNames[0]);
  ASSERT_TRUE("cov2" == columnNames[1]);
}

TEST_F(CSVReaderTest, ReadFileWrongNumber) {
  filePath = "../data/test_csv_wrong_number.txt";

  ASSERT_THROW(CSVReader(filePath, idColumnName, delimiter, personHandlerMock), FileReaderException);
}

TEST_F(CSVReaderTest, ReadAndGetData) {
  CuEira::FileIO::CSVReader csvReader(filePath, idColumnName, delimiter, personHandlerMock);
  PRECISION column1[numberOfIndividualsToIncludeStatic] = {1, 0, 1, 0, 1, 0};
  PRECISION column2[numberOfIndividualsToIncludeStatic] = {1.1, -3, -10, 3, 2, 2};

  const Container::HostMatrix& dataMatrix = csvReader.getData();
  ASSERT_EQ(numberOfIndividualsToInclude, dataMatrix.getNumberOfRows());
  ASSERT_EQ(numberOfColumns, dataMatrix.getNumberOfColumns());

  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    ASSERT_EQ(column1[i], dataMatrix(i, 0));
    ASSERT_EQ(column2[i], dataMatrix(i, 1));
  }
}

TEST_F(CSVReaderTest, StoreDataException) {
  CuEira::FileIO::CSVReader csvReader(filePath, idColumnName, delimiter, personHandlerMock);
  std::vector<std::string> lineSplit(3);
  lineSplit[0] = "1";
  lineSplit[1] = "NotANumber";
  lineSplit[2] = "ind0";

  ASSERT_THROW(csvReader.storeData(lineSplit), FileReaderException);
}

}
/* namespace FileIO */
} /* namespace CuEira */

