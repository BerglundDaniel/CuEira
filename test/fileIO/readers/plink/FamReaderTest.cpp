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

using testing::Return;
using testing::_;
using testing::SaveArg;
using testing::DoAll;

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
};

FamReaderTest::FamReaderTest() :
    numberOfIndividuals(10) {

  EXPECT_CALL(configMock, getFamFilePath()).Times(1).WillRepeatedly(
      Return(std::string(CuEira_BUILD_DIR) + std::string("/test.fam")));
}

FamReaderTest::~FamReaderTest() {

}

void FamReaderTest::SetUp() {

}

void FamReaderTest::TearDown() {

}

TEST_F(FamReaderTest, ReadFile) {
  const int numberOfIndividuals = 10;
  EXPECT_CALL(configMock, getPhenotypeCoding()).Times(1).WillRepeatedly(Return(ONE_TWO_CODING));

  PersonHandlerFactoryMock* personHandlerFactoryMock = new PersonHandlerFactoryMock();
  PersonHandlerMock* personHandlerMock = new PersonHandlerMock();
  std::vector<Person*>* persons;

  CuEira::FileIO::FamReader famReader(configMock, personHandlerFactoryMock);

  EXPECT_CALL(*personHandlerFactoryMock, constructPersonHandler(_)).Times(1).WillOnce(
      DoAll(SaveArg<0>(persons), Return(personHandlerMock)));

  PersonHandler* personHandler = famReader.readPersonInformation();
  ASSERT_EQ(personHandler, personHandlerMock);

  for(int i = 0; i < numberOfIndividuals; ++i){
    std::ostringstream os;
    os << "ind" << i;
    Id id(os.str());

    Sex sex;
    Phenotype phenotype;
    bool include = true;

    if(i == 0 || i == 1 || i == 5 || i == 8){
      sex = MALE;
    }else{
      sex = FEMALE;
    }

    if(i == 4 || i == 7 || i == 8){
      phenotype = UNAFFECTED;
    }else if(i == 6 || i == 9){
      phenotype = MISSING;
      include = false;
    }else{
      phenotype = AFFECTED;
    }

    ASSERT_EQ(id, (*persons)[i]->getId());
    ASSERT_EQ(sex, (*persons)[i]->getSex());
    ASSERT_EQ(phenotype, (*persons)[i]->getPhenotype());
    ASSERT_EQ(include, (*persons)[i]->getInclude());
  }

  delete personHandler;
}

TEST_F(FamReaderTest, StringToSex) {
  EXPECT_CALL(configMock, getPhenotypeCoding()).Times(1).WillRepeatedly(Return(ONE_TWO_CODING));
  CuEira::FileIO::FamReader famReader(configMock);

  Sex sex = famReader.stringToSex("1");
  ASSERT_EQ(MALE, sex);

  sex = famReader.stringToSex("2");
  ASSERT_EQ(FEMALE, sex);

  sex = famReader.stringToSex("3");
  ASSERT_EQ(UNKNOWN, sex);
}

TEST_F(FamReaderTest, StringToSexException) {
  EXPECT_CALL(configMock, getPhenotypeCoding()).Times(1).WillRepeatedly(Return(ONE_TWO_CODING));
  CuEira::FileIO::FamReader famReader(configMock);

  ASSERT_THROW(famReader.stringToSex("asdf"), FileReaderException);
}

TEST_F(FamReaderTest, StringToPhenotypeOneTwoCoding) {
  EXPECT_CALL(configMock, getPhenotypeCoding()).Times(1).WillRepeatedly(Return(ONE_TWO_CODING));
  CuEira::FileIO::FamReader famReader(configMock);

  Phenotype phenotype = famReader.stringToPhenotype("2");
  ASSERT_EQ(AFFECTED, phenotype);

  phenotype = famReader.stringToPhenotype("1");
  ASSERT_EQ(UNAFFECTED, phenotype);

  phenotype = famReader.stringToPhenotype("9");
  ASSERT_EQ(MISSING, phenotype);

  phenotype = famReader.stringToPhenotype("0");
  ASSERT_EQ(MISSING, phenotype);
}

TEST_F(FamReaderTest, StringToPhenotypeZeroOneCoding) {
  EXPECT_CALL(configMock, getPhenotypeCoding()).Times(1).WillRepeatedly(Return(ZERO_ONE_CODING));
  CuEira::FileIO::FamReader famReader(configMock);

  Phenotype phenotype = famReader.stringToPhenotype("1");
  ASSERT_EQ(AFFECTED, phenotype);

  phenotype = famReader.stringToPhenotype("0");
  ASSERT_EQ(UNAFFECTED, phenotype);

  phenotype = famReader.stringToPhenotype("9");
  ASSERT_EQ(MISSING, phenotype);
}

TEST_F(FamReaderTest, StringToPhenotypeOneTwoCodingException) {
  EXPECT_CALL(configMock, getPhenotypeCoding()).Times(1).WillRepeatedly(Return(ONE_TWO_CODING));
  CuEira::FileIO::FamReader famReader(configMock);

  ASSERT_THROW(famReader.stringToPhenotype("-3"), FileReaderException);
  ASSERT_THROW(famReader.stringToPhenotype("notAnInt"), FileReaderException);
}

TEST_F(FamReaderTest, StringToPhenotypeZeroOneCodingException) {
  EXPECT_CALL(configMock, getPhenotypeCoding()).Times(1).WillRepeatedly(Return(ZERO_ONE_CODING));
  CuEira::FileIO::FamReader famReader(configMock);

  ASSERT_THROW(famReader.stringToPhenotype("-3"), FileReaderException);
  ASSERT_THROW(famReader.stringToPhenotype("notAnInt"), FileReaderException);
}

}
/* namespace FileIO */
} /* namespace CuEira */

