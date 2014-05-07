#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>

#include <FamReader.h>
#include <Configuration.h>
#include <ConfigurationMock.h>
#include <PhenotypeCoding.h>
#include <PersonHandlerMock.h>

using testing::Return;
using testing::_;

namespace CuEira {
namespace CuEira_Test {

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
};

FamReaderTest::FamReaderTest() {

}

FamReaderTest::~FamReaderTest() {

}

void FamReaderTest::SetUp() {

}

void FamReaderTest::TearDown() {

}

TEST_F(FamReaderTest, ReadFile){
  int numberOfIndividuals=10;
  ConfigurationMock configMock;
  PersonHandlerMock personHandlerMock;

  //Expect Configuration
  EXPECT_CALL(configMock, getFamFilePath()).Times(1).WillRepeatedly(Return("../data/test.fam"));
  EXPECT_CALL(configMock, getPhenotypeCoding()).WillRepeatedly(Return(ONE_TWO_CODING));

  //Expect PersonHandler
  EXPECT_CALL(personHandlerMock, addPerson(_,_)).Times(numberOfIndividuals);

  CuEira::FileIO::FamReader famReader(configMock, personHandlerMock);

  //ASSERT_EQ();
}

TEST_F(FamReaderTest, StringToSex){

}

TEST_F(FamReaderTest, StringToPhenotype){

}

}
/* namespace CuEira_Test */
} /* namespace CuEira */

