#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <sstream>
#include <ostream>

#include <ContingencyTable.h>
#include <ContingencyTableFactory.h>
#include <SNPVectorMock.h>
#include <EnvironmentVectorMock.h>
#include <ConstructorHelpers.h>

#ifdef CPU
#include <lapackpp/lavd.h>
#include <LapackppHostVector.h>
#else
#include <PinnedHostVector.h>
#endif

using testing::Return;
using testing::ReturnRef;
using testing::_;

namespace CuEira {

/**
 * Test for testing ContingencyTableFactory
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ContingencyTableFactoryTest: public ::testing::Test {
protected:
  ContingencyTableFactoryTest();
  virtual ~ContingencyTableFactoryTest();
  virtual void SetUp();
  virtual void TearDown();

  CuEira_Test::ConstructorHelpers constructorHelpers;
};

ContingencyTableFactoryTest::ContingencyTableFactoryTest() {

}

ContingencyTableFactoryTest::~ContingencyTableFactoryTest() {

}

void ContingencyTableFactoryTest::SetUp() {

}

void ContingencyTableFactoryTest::TearDown() {

}

TEST_F(ContingencyTableFactoryTest, Construct) {
  const int numberOfIndividuals = 10;
  std::vector<int> tableCellNumbers(8);
  Container::SNPVectorMock* snpVector = constructorHelpers.constructSNPVectorMock();
  Container::EnvironmentVectorMock* environmentVector = constructorHelpers.constructEnvironmentVectorMock();

#ifdef CPU
  Container::HostVector* envData = new Container::LapackppHostVector(new LaVectorDouble(numberOfIndividuals));
  Container::HostVector* snpData = new Container::LapackppHostVector(new LaVectorDouble(numberOfIndividuals));
  Container::HostVector* outcomes = new Container::LapackppHostVector(new LaVectorDouble(numberOfIndividuals));
#else
  Container::HostVector* envData = new Container::PinnedHostVector(numberOfIndividuals);
  Container::HostVector* snpData = new Container::PinnedHostVector(numberOfIndividuals);
  Container::HostVector* outcomes = new Container::PinnedHostVector(numberOfIndividuals);
#endif

  for(int i = 0; i < numberOfIndividuals; ++i){
    if(i < 5){
      (*outcomes)(i) = 0;
    }else{
      (*outcomes)(i) = 1;
    }

    if(i < 3 || i > 7){
      (*envData)(i) = 0;
    }else{
      (*envData)(i) = 1;
    }

    if(i < 2 || i > 6){
      (*snpData)(i) = 0;
    }else{
      (*snpData)(i) = 1;
    }
  }

  tableCellNumbers[SNP0_ENV0_CASE_POSITION] = 2;
  tableCellNumbers[SNP1_ENV0_CASE_POSITION] = 0;
  tableCellNumbers[SNP0_ENV1_CASE_POSITION] = 1;
  tableCellNumbers[SNP1_ENV1_CASE_POSITION] = 2;

  tableCellNumbers[SNP0_ENV0_CONTROL_POSITION] = 2;
  tableCellNumbers[SNP1_ENV0_CONTROL_POSITION] = 1;
  tableCellNumbers[SNP0_ENV1_CONTROL_POSITION] = 0;
  tableCellNumbers[SNP1_ENV1_CONTROL_POSITION] = 2;

  std::ostringstream resultOs;
  resultOs << tableCellNumbers[SNP0_ENV0_CASE_POSITION] << "," << tableCellNumbers[SNP0_ENV0_CONTROL_POSITION] << ","
      << tableCellNumbers[SNP1_ENV0_CASE_POSITION] << "," << tableCellNumbers[SNP1_ENV0_CONTROL_POSITION] << ","
      << tableCellNumbers[SNP0_ENV1_CASE_POSITION] << "," << tableCellNumbers[SNP0_ENV1_CONTROL_POSITION] << ","
      << tableCellNumbers[SNP1_ENV1_CASE_POSITION] << "," << tableCellNumbers[SNP1_ENV1_CONTROL_POSITION];

  ContingencyTableFactory contingencyTableFactory(*outcomes);

  EXPECT_CALL(*snpVector, getRecodedData()).Times(1).WillRepeatedly(ReturnRef(*snpData));
  EXPECT_CALL(*environmentVector, getRecodedData()).Times(1).WillRepeatedly(ReturnRef(*envData));

  ContingencyTable* contingencyTable = contingencyTableFactory.constructContingencyTable(*snpVector, *environmentVector);

  std::ostringstream osContingencyTable;
  osContingencyTable << *contingencyTable;

  EXPECT_EQ(resultOs.str(), osContingencyTable.str());

  delete contingencyTable;
  delete environmentVector;
  delete snpVector;
  delete envData;
  delete snpData;
  delete outcomes;
}

} /* namespace CuEira */
