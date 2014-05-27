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
class EnvironmentCSVReaderTest: public ::testing::Test {
protected:
  EnvironmentCSVReaderTest();
  virtual ~EnvironmentCSVReaderTest();
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

EnvironmentCSVReaderTest::EnvironmentCSVReaderTest() :
    filePath(std::string(CuEira_BUILD_DIR)+std::string("/test_env.txt")), delimiter("\t "), idColumnName("indid"), numberOfIndividualsTotal(
        numberOfIndividualsTotalStatic), numberOfIndividualsToInclude(numberOfIndividualsToIncludeStatic) {

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
    EXPECT_CALL(personHandlerMock, getRowIncludeFromPerson(Eq(*person))).WillRepeatedly(Return(includePosArr[i]));
  }
}

void EnvironmentCSVReaderTest::TearDown() {
  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    delete persons[i];
    delete ids[i];
  }
}

TEST_F(EnvironmentCSVReaderTest, ReadAndGetData) {
  CuEira::FileIO::EnvironmentCSVReader envCSVReader(filePath, idColumnName, delimiter, personHandlerMock);
  PRECISION column1[numberOfIndividualsToIncludeStatic] = {1, 0, 1, 0, 1, 0};
  PRECISION column2[numberOfIndividualsToIncludeStatic] = {3, 1, 1, 3, 0, 3};
  Id id1("env1");
  Id id2("env2");
  EnvironmentFactor environmentFactor1(id1);
  EnvironmentFactor environmentFactor2(id2);

  const Container::HostVector& dataVector1 = envCSVReader.getData(environmentFactor1);
  ASSERT_EQ(numberOfIndividualsToInclude, dataVector1.getNumberOfRows());
  ASSERT_EQ(BINARY, environmentFactor1.getVariableType());

  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    ASSERT_EQ(column1[i], dataVector1(i));
  }

  const Container::HostVector& dataVector2 = envCSVReader.getData(environmentFactor2);
  ASSERT_EQ(numberOfIndividualsToInclude, dataVector2.getNumberOfRows());
  ASSERT_EQ(OTHER, environmentFactor2.getVariableType());

  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    ASSERT_EQ(column2[i], dataVector2(i));
  }
}

TEST_F(EnvironmentCSVReaderTest, GetDataException) {
  CuEira::FileIO::EnvironmentCSVReader envCSVReader(filePath, idColumnName, delimiter, personHandlerMock);
  Id id("NotAColumn");
  EnvironmentFactor environmentFactor(id);

  ASSERT_THROW(envCSVReader.getData(environmentFactor), FileReaderException);
}

TEST_F(EnvironmentCSVReaderTest, GetEnvironmentFactorsInfo) {
  CuEira::FileIO::EnvironmentCSVReader envCSVReader(filePath, idColumnName, delimiter, personHandlerMock);

  const std::vector<EnvironmentFactor*>& envInfo = envCSVReader.getEnvironmentFactorInformation();

  ASSERT_EQ(numberOfColumns, envInfo.size());

  EXPECT_EQ("env1", envInfo[0]->getId().getString());
  EXPECT_EQ("env2", envInfo[1]->getId().getString());
}

}
/* namespace FileIO */
} /* namespace CuEira */

