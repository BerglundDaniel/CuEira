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

#ifdef CPU
#include <LapackppHostMatrix.h>
#include <LapackppHostVector.h>
#else
#include <PinnedHostMatrix.h>
#include <PinnedHostVector.h>
#endif

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
  const int notInclude[4] = {1, 2, 5, 7};
  const int numberOfColumns = 2;

  Id* ids[numberOfIndividualsTotalStatic];
  Person* persons[numberOfIndividualsTotalStatic];
  int includePosArr[numberOfIndividualsTotalStatic];

};

CSVReaderTest::CSVReaderTest() :
    filePath("../data/test_csv.txt"), delimiter("\t "), idColumnName("indid"), numberOfIndividualsTotal(
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
}

TEST_F(CSVReaderTest, ReadFileWrongNumber) {
  filePath = "../data/test_csv_wrong_number.txt";

  ASSERT_THROW(CSVReader(filePath, idColumnName, delimiter, personHandlerMock), FileReaderException);
}

//read and get data, both all and by column

//store and get data
//store data exception
//get data exception
}
/* namespace FileIO */
} /* namespace CuEira */

