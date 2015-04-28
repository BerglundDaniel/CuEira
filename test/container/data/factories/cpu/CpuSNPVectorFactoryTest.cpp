#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>
#include <set>

#include <SNP.h>
#include <Id.h>
#include <Recode.h>
#include <SNPVector.h>
#include <CpuSNPVector.h>
#include <InvalidState.h>
#include <GeneticModel.h>
#include <RiskAllele.h>
#include <CpuSNPVectorFactory.h>
#include <ConfigurationMock.h>

using testing::Return;

namespace CuEira {
namespace Container {
namespace CPU {

/**
 * Test for testing CpuSNPVectorFactory
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuSNPVectorFactoryTest: public ::testing::Test {
protected:
  CpuSNPVectorFactoryTest();
  virtual ~CpuSNPVectorFactoryTest();
  virtual void SetUp();
  virtual void TearDown();

  ConfigurationMock configMock;
};

CpuSNPVectorFactoryTest::CpuSNPVectorFactoryTest() {

}

CpuSNPVectorFactoryTest::~CpuSNPVectorFactoryTest() {

}

void CpuSNPVectorFactoryTest::SetUp() {

}

void CpuSNPVectorFactoryTest::TearDown() {

}

TEST_F(CpuSNPVectorFactoryTest, ConstructSNPVector) {
  GeneticModel geneticModel = DOMINANT;
  const int numberOfIndividuals = 6;
  EXPECT_CALL(configMock, getGeneticModel()).Times(1).WillRepeatedly(Return(geneticModel));
  CpuSNPVectorFactory cpuSNPVectorFactory(configMock);

  SNP snp(Id("snp1"), "a1", "a2", 1);

  RegularHostVector* snpData = new RegularHostVector(numberOfIndividuals);
  for(int i = 0; i < numberOfIndividuals; ++i){
    (*snpData)(i) = i % 2;
  }

  std::set<int>* snpMissingData = new std::set<int>();
  snpMissingData->insert(3);
  snpMissingData->insert(5);

  CpuSNPVector* snpVector = cpuSNPVectorFactory.constructSNPVector(snp, snpData, snpMissingData);
  EXPECT_EQ(snp, snpVector->getAssociatedSNP());

  const RegularHostVector& snpOrgData = snpVector->getOriginalSNPData();
  EXPECT_EQ(&(*snpData)(0), &snpOrgData(0));

  EXPECT_TRUE(snpVector->hasMissing());
  const std::set<int>& missingData = snpVector->getMissing();
  EXPECT_EQ(snpMissingData, &missingData);

  delete snpVector;
}

} /* namespace CPU */
} /* namespace Container */
} /* namespace CuEira */
