#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <sstream>
#include <ostream>

#include <ModelInformation.h>
#include <ModelState.h>
#include <ModelInformationFactory.h>
#include <SNP.h>
#include <EnvironmentFactor.h>
#include <AlleleStatisticsMock.h>
#include <ContingencyTableMock.h>

using testing::Return;
using testing::ReturnRef;
using testing::_;
using testing::ByRef;

namespace CuEira {
namespace Model {

/**
 * Test for testing ModelInformation
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ModelInformationFactoryTest: public ::testing::Test {
protected:
  ModelInformationFactoryTest();
  virtual ~ModelInformationFactoryTest();
  virtual void SetUp();
  virtual void TearDown();

};

ModelInformationFactoryTest::ModelInformationFactoryTest() {

}

ModelInformationFactoryTest::~ModelInformationFactoryTest() {

}

void ModelInformationFactoryTest::SetUp() {

}

void ModelInformationFactoryTest::TearDown() {

}

TEST_F(ModelInformationFactoryTest, ConstructModelInformationState) {
  ModelInformationFactory modelInformationFactory;

  ModelInformation* modelInformation = modelInformationFactory.constructModelInformation(DONE);
  EXPECT_EQ(DONE, modelInformation->getModelState());

  delete modelInformation;
}

TEST_F(ModelInformationFactoryTest, ConstructModelInformationNotTable) {
  ModelInformationFactory modelInformationFactory;

  SNP snp(Id("snp"), "allele1", "allele2", 1);
  EnvironmentFactor environmentFactor(Id("Env"));

  AlleleStatisticsMock alleleStatistics;

  EXPECT_CALL(alleleStatistics, toOstream(_)).Times(2);

  std::ostringstream result;
  result << snp << "," << environmentFactor << "," << alleleStatistics;

  ModelInformation* modelInformation = modelInformationFactory.constructModelInformation(DONE, snp, environmentFactor,
      alleleStatistics);
  EXPECT_EQ(DONE, modelInformation->getModelState());

  std::ostringstream osModelInformation;
  osModelInformation << *modelInformation;
  EXPECT_EQ(result.str(), osModelInformation.str());

  delete modelInformation;
}

TEST_F(ModelInformationFactoryTest, ConstructModelInformationAll) {
  ModelInformationFactory modelInformationFactory;

  SNP snp(Id("snp"), "allele1", "allele2", 1);
  EnvironmentFactor environmentFactor(Id("Env"));

  AlleleStatisticsMock alleleStatistics;
  ContingencyTableMock contingencyTable;

  EXPECT_CALL(alleleStatistics, toOstream(_)).Times(2);
  EXPECT_CALL(contingencyTable, toOstream(_)).Times(2);

  std::ostringstream result;
  result << snp << "," << environmentFactor << "," << alleleStatistics << "," << contingencyTable;


  ModelInformation* modelInformation = modelInformationFactory.constructModelInformation(DONE, snp, environmentFactor,
      alleleStatistics, contingencyTable);
  EXPECT_EQ(DONE, modelInformation->getModelState());

  std::ostringstream osModelInformation;
  osModelInformation << *modelInformation;
  EXPECT_EQ(result.str(), osModelInformation.str());

  delete modelInformation;
}

}
/* namespace Model */
} /* namespace CuEira */
