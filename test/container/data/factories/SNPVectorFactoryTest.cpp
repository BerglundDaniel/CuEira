#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>

#include <SNP.h>
#include <Id.h>
#include <Recode.h>
#include <SNPVector.h>
#include <InvalidState.h>
#include <GeneticModel.h>
#include <RiskAllele.h>
#include <StatisticModel.h>
#include <SNPVectorFactory.h>
#include <ConfigurationMock.h>

using testing::Return;
using testing::_;
using testing::Eq;
using testing::Ge;
using testing::Le;

namespace CuEira {
namespace Container {

/**
 * Test for testing SNPVectorFactory
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class SNPVectorFactoryTest: public ::testing::Test {
protected:
  SNPVectorFactoryTest();
  virtual ~SNPVectorFactoryTest();
  virtual void SetUp();
  virtual void TearDown();

  ConfigurationMock configMock;
};

SNPVectorFactoryTest::SNPVectorFactoryTest() {

}

SNPVectorFactoryTest::~SNPVectorFactoryTest() {

}

void SNPVectorFactoryTest::SetUp() {

}

void SNPVectorFactoryTest::TearDown() {

}

TEST_F(SNPVectorFactoryTest, ConstructSNPVector) {
  GeneticModel geneticModel = DOMINANT;
  const int numberOfIndividuals = 6;
  EXPECT_CALL(configMock, getGeneticModel()).Times(1).WillRepeatedly(Return(geneticModel));
  SNPVectorFactory snpVectorFactory(configMock);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  snp.setRiskAllele(ALLELE_ONE);
  ASSERT_TRUE(snp.shouldInclude());

  std::vector<int>* snpData = new std::vector<int>(numberOfIndividuals);
  (*snpData)[0] = 2;
  (*snpData)[1] = 1;
  (*snpData)[2] = 0;
  (*snpData)[3] = 2;
  (*snpData)[4] = 0;
  (*snpData)[5] = 1;

  std::vector<int>*recodedSNPData = new std::vector<int>(numberOfIndividuals);
  (*recodedSNPData)[0] = 0;
  (*recodedSNPData)[1] = 1;
  (*recodedSNPData)[2] = 1;
  (*recodedSNPData)[3] = 0;
  (*recodedSNPData)[4] = 1;
  (*recodedSNPData)[5] = 1;

  SNPVector* snpVector = snpVectorFactory.constructSNPVector(snp, snpData);
  EXPECT_EQ(geneticModel, snpVector->originalGeneticModel);
  EXPECT_EQ(ALLELE_ONE, snpVector->originalRiskAllele);
  EXPECT_EQ(ALLELE_ONE, snp.getRiskAllele());
  EXPECT_EQ(snp, snpVector->getAssociatedSNP());

  const std::vector<int>& orgData = snpVector->getOrginalData();
  const HostVector& recodedData = snpVector->getRecodedData();

  ASSERT_EQ(numberOfIndividuals, orgData.size());
  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());

  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*snpData)[i], (orgData)[i]);
    EXPECT_EQ((*recodedSNPData)[i], (recodedData)(i));
  }

  delete recodedSNPData;
  delete snpVector;
}

} /* namespace Container */
} /* namespace CuEira */

