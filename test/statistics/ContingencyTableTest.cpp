#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <sstream>
#include <ostream>

#include <ContingencyTable.h>

namespace CuEira {

/**
 * Test for testing ContingencyTable
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ContingencyTableTest: public ::testing::Test {
protected:
  ContingencyTableTest();
  virtual ~ContingencyTableTest();
  virtual void SetUp();
  virtual void TearDown();

};

ContingencyTableTest::ContingencyTableTest() {

}

ContingencyTableTest::~ContingencyTableTest() {

}

void ContingencyTableTest::SetUp() {

}

void ContingencyTableTest::TearDown() {

}

TEST_F(ContingencyTableTest, Getters) {
  std::vector<int>* tableCellNumbers = new std::vector<int>(8);

  for(int i = 0; i < 8; ++i){
    (*tableCellNumbers)[i] = i;
  }

  ContingencyTable contingencyTable(tableCellNumbers);

  ASSERT_EQ(tableCellNumbers, &contingencyTable.getTable());

  std::ostringstream resultOs;

  resultOs << (*tableCellNumbers)[SNP0_ENV0_CASE_POSITION] << "," << (*tableCellNumbers)[SNP0_ENV0_CONTROL_POSITION]
      << "," << (*tableCellNumbers)[SNP1_ENV0_CASE_POSITION] << "," << (*tableCellNumbers)[SNP1_ENV0_CONTROL_POSITION]
      << "," << (*tableCellNumbers)[SNP0_ENV1_CASE_POSITION] << "," << (*tableCellNumbers)[SNP0_ENV1_CONTROL_POSITION]
      << "," << (*tableCellNumbers)[SNP1_ENV1_CASE_POSITION] << "," << (*tableCellNumbers)[SNP1_ENV1_CONTROL_POSITION];

  std::ostringstream osContingencyTable;
  osContingencyTable << contingencyTable;

  EXPECT_EQ(resultOs.str(), osContingencyTable.str());
}

} /* namespace CuEira */
