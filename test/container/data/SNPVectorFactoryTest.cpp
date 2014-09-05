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
  EXPECT_CALL(configMock, getGeneticModel()).Times(1).WillRepeatedly(Return(geneticModel));
  SNPVectorFactory snpVectorFactory(configMock);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  RiskAllele riskAllele = ALLELE_ONE;
  snp.setRiskAllele(riskAllele);
  ASSERT_TRUE(snp.shouldInclude());

  std::vector<int>* snpData = new std::vector<int>();

  SNPVector* snpVector = snpVectorFactory.constructSNPVector(snp, snpData);
  ASSERT_EQ(geneticModel, snpVector->originalGeneticModel);
  ASSERT_EQ(riskAllele, snpVector->originalRiskAllele);

  delete snpVector;
}

} /* namespace Container */
} /* namespace CuEira */

