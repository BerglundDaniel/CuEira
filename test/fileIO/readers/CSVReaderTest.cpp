#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <utility>

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
#include <RegularHostMatrix.h>

using testing::Return;
using testing::_;
using testing::ReturnRef;
using testing::Eq;
using testing::Ge;
using testing::Le;
using testing::ByRef;

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
  const int numberOfColumns;
  const int numberOfIndividualsTotal;
  const int numberOfIndividualsToInclude;
  PersonHandlerMock personHandlerMock;
  CuEira_Test::ConstructorHelpers constructorHelpers;
  std::string filePath;
  std::string delimiter;
  std::string idColumnName;
  std::vector<int> notInclude; //Index 0 based

  Id* ids[numberOfIndividualsTotalStatic];
  Person* persons[numberOfIndividualsTotalStatic];
  int includePosArr[numberOfIndividualsTotalStatic];

};

CSVReaderTest::CSVReaderTest() :
    numberOfColumns(2), filePath(std::string(CuEira_BUILD_DIR) + std::string("/test_csv.txt")), delimiter("\t "), idColumnName(
        "indid"), numberOfIndividualsTotal(numberOfIndividualsTotalStatic), numberOfIndividualsToInclude(
        numberOfIndividualsToIncludeStatic), notInclude(4) {
  notInclude[0] = 1;
  notInclude[1] = 2;
  notInclude[2] = 5;
  notInclude[3] = 7;
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
    EXPECT_CALL(personHandlerMock, getRowIncludeFromPerson(Eq(ByRef(*person)))).WillRepeatedly(
        Return(includePosArr[i]));
  }
}

void CSVReaderTest::TearDown() {
  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    delete persons[i];
    delete ids[i];
  }
}

TEST_F(CSVReaderTest, ReadFile) {
  CuEira::FileIO::CSVReader csvReader(filePath, idColumnName, delimiter);

  std::pair<Container::HostMatrix*, std::vector<std::string>*>* csvPair = csvReader.readData(personHandlerMock);
  Container::HostMatrix* dataMatrix = csvPair->first;
  std::vector<std::string>* columnNames = csvPair->second;
  delete csvPair;

  int csvNumberOfIndividualsToInclude = dataMatrix->getNumberOfRows();
  int csvNumberOfColumns = dataMatrix->getNumberOfColumns();

  ASSERT_EQ(numberOfIndividualsToInclude, csvNumberOfIndividualsToInclude);
  ASSERT_EQ(numberOfColumns, csvNumberOfColumns);

  ASSERT_TRUE("cov1" == (*columnNames)[0]);
  ASSERT_TRUE("cov2" == (*columnNames)[1]);

  delete dataMatrix;
  delete columnNames;
}

TEST_F(CSVReaderTest, ReadFileWrongNumber) {
  filePath = "../data/test_csv_wrong_number.txt";
  CuEira::FileIO::CSVReader csvReader(filePath, idColumnName, delimiter);

  ASSERT_THROW(csvReader.readData(personHandlerMock), FileReaderException);
}

TEST_F(CSVReaderTest, ReadAndGetData) {
  CuEira::FileIO::CSVReader csvReader(filePath, idColumnName, delimiter);
  double e = 1e-5;
  double l;
  double h;
  PRECISION column1[numberOfIndividualsToIncludeStatic] = {1, 0, 1, 0, 1, 0};
  PRECISION column2[numberOfIndividualsToIncludeStatic] = {1.1, -3, -10, 3, 2, 2};

  std::pair<Container::HostMatrix*, std::vector<std::string>*>* csvPair = csvReader.readData(personHandlerMock);
  Container::HostMatrix* dataMatrix = csvPair->first;
  std::vector<std::string>* columnNames = csvPair->second;
  delete csvPair;

  ASSERT_EQ(numberOfIndividualsToInclude, dataMatrix->getNumberOfRows());
  ASSERT_EQ(numberOfColumns, dataMatrix->getNumberOfColumns());

  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    l = column1[i] - e;
    h = column1[i] + e;

    EXPECT_THAT((*dataMatrix)(i, 0), Ge(l));
    EXPECT_THAT((*dataMatrix)(i, 0), Le(h));

    l = column2[i] - e;
    h = column2[i] + e;

    EXPECT_THAT((*dataMatrix)(i, 1), Ge(l));
    EXPECT_THAT((*dataMatrix)(i, 1), Le(h));
  }

  delete dataMatrix;
  delete columnNames;
}

TEST_F(CSVReaderTest, StoreDataException) {
  CuEira::FileIO::CSVReader csvReader(filePath, idColumnName, delimiter);

  std::pair<Container::HostMatrix*, std::vector<std::string>*>* csvPair = csvReader.readData(personHandlerMock);
  Container::HostMatrix* dataMatrix1 = csvPair->first;
  std::vector<std::string>* columnNames = csvPair->second;
  delete csvPair;

  std::vector<std::string> lineSplit(3);
  lineSplit[0] = "1";
  lineSplit[1] = "NotANumber";
  lineSplit[2] = "ind0";

  Container::HostMatrix* dataMatrix2 = new Container::RegularHostMatrix(numberOfIndividualsToInclude, numberOfColumns);

  ASSERT_THROW(csvReader.storeData(lineSplit, 0, dataMatrix2, 0), FileReaderException);

  delete dataMatrix1;
  delete columnNames;
}

}
/* namespace FileIO */
} /* namespace CuEira */

