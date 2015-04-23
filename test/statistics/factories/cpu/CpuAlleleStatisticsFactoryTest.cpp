#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <sstream>
#include <ostream>

#include <AlleleStatistics.h>
#include <CpuAlleleStatisticsFactory.h>
#include <SNPVectorMock.h>
#include <PhenotypeVectorMock.h>
#include <RegularHostVector.h>

using testing::Return;
using testing::ReturnRef;
using testing::AtLeast;

namespace CuEira {
namespace CPU {

/**
 * Test for testing CpuAlleleStatisticsFactory
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuAlleleStatisticsFactoryTest: public ::testing::Test {
protected:
  CpuAlleleStatisticsFactoryTest();
  virtual ~CpuAlleleStatisticsFactoryTest();
  virtual void SetUp();
  virtual void TearDown();

};

CpuAlleleStatisticsFactoryTest::CpuAlleleStatisticsFactoryTest() {

}

CpuAlleleStatisticsFactoryTest::~CpuAlleleStatisticsFactoryTest() {

}

void CpuAlleleStatisticsFactoryTest::SetUp() {

}

void CpuAlleleStatisticsFactoryTest::TearDown() {

}

TEST_F(CpuAlleleStatisticsFactoryTest, Construct) {
  const int numberOfIndividuals = 10;
  CpuAlleleStatisticsFactory cpuAlleleStatisticsFactory;

  Container::RegularHostVector snpData(numberOfIndividuals);
  Container::RegularHostVector phenotypeData(numberOfIndividuals);

  for(int i = 0; i < numberOfIndividuals; ++i){
    snpData(i) = i % 3;
  }

  for(int i = 0; i < numberOfIndividuals; ++i){
    if(i < 5){
      phenotypeData(i) = 0;
    }else{
      phenotypeData(i) = 1;
    }
  }

  Container::SNPVectorMock<Container::RegularHostVector> snpVectorMock;
  EXPECT_CALL(snpVectorMock, getNumberOfIndividualsToInclude()).WillRepeatedly(Return(numberOfIndividuals));
  EXPECT_CALL(snpVectorMock, getOriginalSNPData()).Times(1).WillRepeatedly(ReturnRef(snpData));

  Container::PhenotypeVectorMock<Container::RegularHostVector> phenotypeVectorMock;
  EXPECT_CALL(phenotypeVectorMock, getNumberOfIndividualsToInclude()).WillRepeatedly(Return(numberOfIndividuals));
  EXPECT_CALL(phenotypeVectorMock, getPhenotypeData()).Times(1).WillRepeatedly(ReturnRef(phenotypeData));

  AlleleStatistics* alleleStatistics = cpuAlleleStatisticsFactory.constructAlleleStatistics(snpVectorMock,
      phenotypeVectorMock);

  const std::vector<int>& numberOfAlleles = &alleleStatistics->getAlleleNumbers();
  const std::vector<double>& alleleFrequencies = alleleStatistics->getAlleleFrequencies();
  const double maf = alleleStatistics->getMinorAlleleFrequecy();

  ASSERT_EQ(6, numberOfAlleles.size());
  ASSERT_EQ(6, alleleFrequencies.size());

  EXPECT_EQ(5, numberOfAlleles[ALLELE_ONE_CASE_POSITION]);
  EXPECT_EQ(5, numberOfAlleles[ALLELE_TWO_CASE_POSITION]);

  EXPECT_EQ(6, numberOfAlleles[ALLELE_ONE_CONTROL_POSITION]);
  EXPECT_EQ(4, numberOfAlleles[ALLELE_TWO_CONTROL_POSITION]);

  EXPECT_EQ(11, numberOfAlleles[ALLELE_ONE_ALL_POSITION]);
  EXPECT_EQ(9, numberOfAlleles[ALLELE_TWO_ALL_POSITION]);

  EXPECT_EQ((double )9 / 20, maf);

  EXPECT_EQ((double )5 / 10, alleleFrequencies[ALLELE_ONE_CASE_POSITION]);
  EXPECT_EQ((double )5 / 10, alleleFrequencies[ALLELE_TWO_CASE_POSITION]);

  EXPECT_EQ((double )6 / 10, alleleFrequencies[ALLELE_ONE_CONTROL_POSITION]);
  EXPECT_EQ((double )4 / 10, alleleFrequencies[ALLELE_TWO_CONTROL_POSITION]);

  EXPECT_EQ((double )11 / 20, alleleFrequencies[ALLELE_ONE_ALL_POSITION]);
  EXPECT_EQ((double )9 / 20, alleleFrequencies[ALLELE_TWO_ALL_POSITION]);

  delete alleleStatistics;
}

} /* namespace CPU */
} /* namespace CuEira */
