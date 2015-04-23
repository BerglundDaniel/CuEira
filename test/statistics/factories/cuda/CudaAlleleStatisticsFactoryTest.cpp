#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <sstream>
#include <ostream>

#include <AlleleStatistics.h>
#include <CudaAlleleStatisticsFactory.h>
#include <SNPVectorMock.h>
#include <PhenotypeVectorMock.h>
#include <PinnedHostMatrix.h>
#include <PinnedHostVector.h>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <DeviceToHost.h>
#include <HostToDevice.h>
#include <Device.h>
#include <Stream.h>
#include <StreamFactory.h>

using testing::Return;
using testing::ReturnRef;
using testing::AtLeast;

namespace CuEira {
namespace CUDA {

/**
 * Test for testing CudaAlleleStatisticsFactory
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaAlleleStatisticsFactoryTest: public ::testing::Test {
protected:
  CudaAlleleStatisticsFactoryTest();
  virtual ~CudaAlleleStatisticsFactoryTest();
  virtual void SetUp();
  virtual void TearDown();

  Device device;
  StreamFactory streamFactory;
  Stream* stream;
  HostToDevice hostToDeviceStream1;
  DeviceToHost deviceToHostStream1;
};

CudaAlleleStatisticsFactoryTest::CudaAlleleStatisticsFactoryTest() :
    device(0), streamFactory(), stream(streamFactory.constructStream(device)), hostToDeviceStream1(*stream), deviceToHostStream1(
        *stream) {

}

CudaAlleleStatisticsFactoryTest::~CudaAlleleStatisticsFactoryTest() {
  delete stream;
}

void CudaAlleleStatisticsFactoryTest::SetUp() {

}

void CudaAlleleStatisticsFactoryTest::TearDown() {

}

TEST_F(CudaAlleleStatisticsFactoryTest, Construct) {
  const int numberOfIndividuals = 10;
  CudaAlleleStatisticsFactory cudaAlleleStatisticsFactory;

  Container::PinnedHostVector snpData(numberOfIndividuals);
  Container::PinnedHostVector phenotypeData(numberOfIndividuals);

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

  Container::DeviceVector* snpDataDevice = hostToDeviceStream1.transferVector(snpData);
  Container::DeviceVector* phenotypeDataDevice = hostToDeviceStream1.transferVector(phenotypeData);

  Container::SNPVectorMock<Container::DeviceVector> snpVectorMock(snp);
  EXPECT_CALL(snpVectorMock, getNumberOfIndividualsToInclude()).WillRepeatedly(Return(numberOfIndividuals));
  EXPECT_CALL(snpVectorMock, getOriginalSNPData()).Times(1).WillRepeatedly(ReturnRef(*snpDataDevice));

  Container::PhenotypeVectorMock<Container::DeviceVector> phenotypeVectorMock;
  EXPECT_CALL(phenotypeVectorMock, getNumberOfIndividualsToInclude()).WillRepeatedly(Return(numberOfIndividuals));
  EXPECT_CALL(phenotypeVectorMock, getPhenotypeData()).Times(1).WillRepeatedly(ReturnRef(*phenotypeDataDevice));

  AlleleStatistics* alleleStatistics = cudaAlleleStatisticsFactory.constructAlleleStatistics(snpVectorMock,
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
  delete snpDataDevice;
  delete phenotypeDataDevice;
}

} /* namespace CUDA */
} /* namespace CuEira */
