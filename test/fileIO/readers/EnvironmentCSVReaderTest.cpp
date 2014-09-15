#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#include <EnvironmentCSVReader.h>
#include <PersonHandlerMock.h>
#include <Id.h>
#include <Sex.h>
#include <Phenotype.h>
#include <Person.h>
#include <FileReaderException.h>
#include <EnvironmentFactor.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <ConstructorHelpers.h>
#include <VariableType.h>
#include <EnvironmentFactorHandler.h>
#include <EnvironmentFactorHandlerException.h>

using testing::Return;
using testing::_;
using testing::ReturnRef;
using testing::Eq;
using testing::ByRef;

namespace CuEira {
namespace FileIO {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class EnvironmentCSVReaderTest: public ::testing::Test {
protected:
  EnvironmentCSVReaderTest();
  virtual ~EnvironmentCSVReaderTest();
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

EnvironmentCSVReaderTest::EnvironmentCSVReaderTest() :
    numberOfColumns(2), filePath(std::string(CuEira_BUILD_DIR) + std::string("/test_env.txt")), delimiter("\t "), idColumnName(
        "indid"), numberOfIndividualsTotal(numberOfIndividualsTotalStatic), numberOfIndividualsToInclude(
        numberOfIndividualsToIncludeStatic), notInclude(4) {
  notInclude[0] = 1;
  notInclude[1] = 2;
  notInclude[2] = 5;
  notInclude[3] = 7;
}

EnvironmentCSVReaderTest::~EnvironmentCSVReaderTest() {

}

void EnvironmentCSVReaderTest::SetUp() {
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
    EXPECT_CALL(personHandlerMock, getRowIncludeFromPerson(Eq(ByRef(*person)))).WillRepeatedly(Return(includePosArr[i]));
  }
}

void EnvironmentCSVReaderTest::TearDown() {
  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    delete persons[i];
    delete ids[i];
  }
}

TEST_F(EnvironmentCSVReaderTest, ReadAndGetData) {
  CuEira::FileIO::EnvironmentCSVReader envCSVReader(filePath, idColumnName, delimiter);
  PRECISION column1[numberOfIndividualsToIncludeStatic] = {1, 0, 1, 0, 1, 0};
  PRECISION column2[numberOfIndividualsToIncludeStatic] = {3, 1, 1, 3, 0, 3};

  EnvironmentFactorHandler* envFactorHandler = envCSVReader.readEnvironmentFactorInformation(personHandlerMock);
  const std::vector<const EnvironmentFactor*>& envInfo = envFactorHandler->getHeaders();

  const Container::HostVector* dataVector1 = envFactorHandler->getData(*envInfo[0]);

  ASSERT_EQ(numberOfIndividualsToInclude, dataVector1->getNumberOfRows());
  ASSERT_EQ(BINARY, envInfo[0]->getVariableType());

  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    ASSERT_EQ(column1[i], (*dataVector1)(i));
  }

  const Container::HostVector* dataVector2 = envFactorHandler->getData(*envInfo[1]);

  ASSERT_EQ(numberOfIndividualsToInclude, dataVector2->getNumberOfRows());
  ASSERT_EQ(OTHER, envInfo[1]->getVariableType());

  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    ASSERT_EQ(column2[i], (*dataVector2)(i));
  }

  delete dataVector1;
  delete dataVector2;
  delete envFactorHandler;
}

TEST_F(EnvironmentCSVReaderTest, GetDataException) {
  CuEira::FileIO::EnvironmentCSVReader envCSVReader(filePath, idColumnName, delimiter);
  Id id("NotAColumn");
  EnvironmentFactor environmentFactor(id);

  EnvironmentFactorHandler* envFactorHandler = envCSVReader.readEnvironmentFactorInformation(personHandlerMock);

  ASSERT_THROW(envFactorHandler->getData(environmentFactor), EnvironmentFactorHandlerException);
  delete envFactorHandler;
}

TEST_F(EnvironmentCSVReaderTest, GetEnvironmentFactorsInfo) {
  CuEira::FileIO::EnvironmentCSVReader envCSVReader(filePath, idColumnName, delimiter);

  EnvironmentFactorHandler* envFactorHandler = envCSVReader.readEnvironmentFactorInformation(personHandlerMock);
  const std::vector<const EnvironmentFactor*>& envInfo = envFactorHandler->getHeaders();

  ASSERT_EQ(numberOfColumns, envInfo.size());

  EXPECT_EQ("env1", envInfo[0]->getId().getString());
  EXPECT_EQ("env2", envInfo[1]->getId().getString());

  delete envFactorHandler;
}

}
/* namespace FileIO */
} /* namespace CuEira */

