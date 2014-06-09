#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#include <FamReader.h>
#include <ConfigurationMock.h>
#include <PhenotypeCoding.h>
#include <PersonHandlerMock.h>
#include <Id.h>
#include <Sex.h>
#include <Phenotype.h>
#include <Person.h>
#include <FileReaderException.h>
#include <ConstructorHelpers.h>

using testing::Return;
using testing::_;
using testing::ReturnRef;

namespace CuEira {
namespace FileIO {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class FamReaderTest: public ::testing::Test {
protected:
  FamReaderTest();
  virtual ~FamReaderTest();
  virtual void SetUp();
  virtual void TearDown();

  int numberOfIndividuals;
  ConfigurationMock configMock;
  CuEira_Test::ConstructorHelpers constructorHelpers;
  Person* person;
};

FamReaderTest::FamReaderTest() :
    numberOfIndividuals(10), person(constructorHelpers.constructPersonInclude(0)) {

  EXPECT_CALL(configMock, getFamFilePath()).Times(1).WillRepeatedly(
      Return(std::string(CuEira_BUILD_DIR) + std::string("/test.fam")));
  EXPECT_CALL(configMock, getPhenotypeCoding()).WillRepeatedly(Return(ONE_TWO_CODING));
}

FamReaderTest::~FamReaderTest() {
  delete person;
}

void FamReaderTest::SetUp() {

}

void FamReaderTest::TearDown() {

}

TEST_F(FamReaderTest, ReadFile) {
  CuEira::FileIO::FamReader famReader(configMock);
  PersonHandler* personHandler = famReader.readPersonInformation();

  ASSERT_EQ(numberOfIndividuals, personHandler->getNumberOfIndividualsTotal());
  ASSERT_EQ(9, personHandler->getNumberOfIndividualsToInclude());

  const Container::HostVector& outcomes = personHandler->getOutcomes();

  const Person& person6 = personHandler->getPersonFromRowAll(6);
  EXPECT_FALSE(person6.getInclude());

  EXPECT_EQ(1, outcomes(0));
  EXPECT_EQ(1, outcomes(1));
  EXPECT_EQ(1, outcomes(2));
  EXPECT_EQ(1, outcomes(3));
  EXPECT_EQ(0, outcomes(4));
  EXPECT_EQ(1, outcomes(5));
  EXPECT_EQ(0, outcomes(6));
  EXPECT_EQ(0, outcomes(7));
  EXPECT_EQ(0, outcomes(8));

  delete personHandler;
}

TEST_F(FamReaderTest, StringToSex) {
  CuEira::FileIO::FamReader famReader(configMock);

  Sex sex = famReader.stringToSex("1");
  ASSERT_EQ(MALE, sex);

  sex = famReader.stringToSex("2");
  ASSERT_EQ(FEMALE, sex);

  sex = famReader.stringToSex("3");
  ASSERT_EQ(UNKNOWN, sex);
}

TEST_F(FamReaderTest, StringToSexException) {
  CuEira::FileIO::FamReader famReader(configMock);

  ASSERT_THROW(famReader.stringToSex("asdf"), FileReaderException);
}

TEST_F(FamReaderTest, StringToPhenotypeOneTwoCoding) {
  CuEira::FileIO::FamReader famReader(configMock);

  Phenotype phenotype = famReader.stringToPhenotype("2");
  ASSERT_EQ(AFFECTED, phenotype);

  phenotype = famReader.stringToPhenotype("1");
  ASSERT_EQ(UNAFFECTED, phenotype);

  phenotype = famReader.stringToPhenotype("9");
  ASSERT_EQ(MISSING, phenotype);
}

TEST_F(FamReaderTest, StringToPhenotypeZeroOneCoding) {
  CuEira::FileIO::FamReader famReader(configMock);
  EXPECT_CALL(configMock, getPhenotypeCoding()).WillRepeatedly(Return(ZERO_ONE_CODING));

  Phenotype phenotype = famReader.stringToPhenotype("1");
  ASSERT_EQ(AFFECTED, phenotype);

  phenotype = famReader.stringToPhenotype("0");
  ASSERT_EQ(UNAFFECTED, phenotype);

  phenotype = famReader.stringToPhenotype("9");
  ASSERT_EQ(MISSING, phenotype);
}

TEST_F(FamReaderTest, StringToPhenotypeOneTwoCodingException) {
  CuEira::FileIO::FamReader famReader(configMock);

  ASSERT_THROW(famReader.stringToPhenotype("-3"), FileReaderException);
  ASSERT_THROW(famReader.stringToPhenotype("notAnInt"), FileReaderException);
}

TEST_F(FamReaderTest, StringToPhenotypeZeroOneCodingException) {
  CuEira::FileIO::FamReader famReader(configMock);
  EXPECT_CALL(configMock, getPhenotypeCoding()).WillRepeatedly(Return(ZERO_ONE_CODING));

  ASSERT_THROW(famReader.stringToPhenotype("-3"), FileReaderException);
  ASSERT_THROW(famReader.stringToPhenotype("notAnInt"), FileReaderException);
}

}
/* namespace FileIO */
} /* namespace CuEira */

