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
class FamReaderTest: public ::testing::Test {
protected:
  FamReaderTest();
  virtual ~FamReaderTest();
  virtual void SetUp();
  virtual void TearDown();

  int numberOfIndividuals;
  ConfigurationMock configMock;
  PersonHandlerMock personHandlerMock;
};

FamReaderTest::FamReaderTest() {
  numberOfIndividuals = 10;
  EXPECT_CALL(configMock, getFamFilePath()).Times(1).WillRepeatedly(Return("../data/test.fam"));
  EXPECT_CALL(configMock, getPhenotypeCoding()).WillRepeatedly(Return(ONE_TWO_CODING));
}

FamReaderTest::~FamReaderTest() {

}

void FamReaderTest::SetUp() {
  numberOfIndividuals = 10;
}

void FamReaderTest::TearDown() {

}

TEST_F(FamReaderTest, ReadFile) {
  EXPECT_CALL(personHandlerMock, addPerson(_,_)).Times(numberOfIndividuals);

  CuEira::FileIO::FamReader famReader(configMock, personHandlerMock);

  //ASSERT_EQ();
}

TEST_F(FamReaderTest, StringToSex) {
  EXPECT_CALL(personHandlerMock, addPerson(_,_)).Times(numberOfIndividuals);
  CuEira::FileIO::FamReader famReader(configMock, personHandlerMock);

  Sex sex = famReader.stringToSex("1");
  ASSERT_EQ(MALE, sex);

  sex = famReader.stringToSex("2");
  ASSERT_EQ(FEMALE, sex);

  sex = famReader.stringToSex("3");
  ASSERT_EQ(UNKNOWN, sex);
}

TEST_F(FamReaderTest, StringToSexException) {
  EXPECT_CALL(personHandlerMock, addPerson(_,_)).Times(numberOfIndividuals);
  CuEira::FileIO::FamReader famReader(configMock, personHandlerMock);

  ASSERT_THROW(famReader.stringToSex("asdf"), FileReaderException);
}

TEST_F(FamReaderTest, StringToPhenotypeOneTwoCoding) {
  EXPECT_CALL(personHandlerMock, addPerson(_,_)).Times(numberOfIndividuals);
  CuEira::FileIO::FamReader famReader(configMock, personHandlerMock);

  Phenotype phenotype = famReader.stringToPhenotype("2");
  ASSERT_EQ(AFFECTED, phenotype);

  phenotype = famReader.stringToPhenotype("1");
  ASSERT_EQ(UNAFFECTED, phenotype);

  phenotype = famReader.stringToPhenotype("9");
  ASSERT_EQ(MISSING, phenotype);
}

TEST_F(FamReaderTest, StringToPhenotypeZeroOneCoding) {
  EXPECT_CALL(personHandlerMock, addPerson(_,_)).Times(numberOfIndividuals);
  CuEira::FileIO::FamReader famReader(configMock, personHandlerMock);
  EXPECT_CALL(configMock, getPhenotypeCoding()).WillRepeatedly(Return(ZERO_ONE_CODING));

  Phenotype phenotype = famReader.stringToPhenotype("1");
  ASSERT_EQ(AFFECTED, phenotype);

  phenotype = famReader.stringToPhenotype("0");
  ASSERT_EQ(UNAFFECTED, phenotype);

  phenotype = famReader.stringToPhenotype("9");
  ASSERT_EQ(MISSING, phenotype);
}

TEST_F(FamReaderTest, StringToPhenotypeOneTwoCodingException) {
  EXPECT_CALL(personHandlerMock, addPerson(_,_)).Times(numberOfIndividuals);
  CuEira::FileIO::FamReader famReader(configMock, personHandlerMock);

  ASSERT_THROW(famReader.stringToPhenotype("-3"), FileReaderException);
  ASSERT_THROW(famReader.stringToPhenotype("notAnInt"), FileReaderException);
}

TEST_F(FamReaderTest, StringToPhenotypeZeroOneCodingException) {
  EXPECT_CALL(personHandlerMock, addPerson(_,_)).Times(numberOfIndividuals);
  CuEira::FileIO::FamReader famReader(configMock, personHandlerMock);
  EXPECT_CALL(configMock, getPhenotypeCoding()).WillRepeatedly(Return(ZERO_ONE_CODING));

  ASSERT_THROW(famReader.stringToPhenotype("-3"), FileReaderException);
  ASSERT_THROW(famReader.stringToPhenotype("notAnInt"), FileReaderException);
}

}
/* namespace FileIO */
} /* namespace CuEira */

