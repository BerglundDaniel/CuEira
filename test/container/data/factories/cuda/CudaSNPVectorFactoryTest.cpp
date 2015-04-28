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
#include <CudaSNPVector.h>
#include <InvalidState.h>
#include <RegularHostVector.h>
#include <GeneticModel.h>
#include <RiskAllele.h>
#include <CudaSNPVectorFactory.h>
#include <ConfigurationMock.h>
#include <DeviceVector.h>
#include <PinnedHostVector.h>
#include <DeviceToHost.h>
#include <HostToDevice.h>
#include <Device.h>
#include <Stream.h>
#include <StreamFactory.h>
#include <KernelWrapper.h>
#include <CublasWrapper.h>

using testing::Return;

namespace CuEira {
namespace Container {
namespace CUDA {

/**
 * Test for testing CudaSNPVectorFactory
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaSNPVectorFactoryTest: public ::testing::Test {
protected:
  CudaSNPVectorFactoryTest();
  virtual ~CudaSNPVectorFactoryTest();
  virtual void SetUp();
  virtual void TearDown();

  Device device;
  StreamFactory streamFactory;
  Stream* stream;
  HostToDevice hostToDevice;
  DeviceToHost deviceToHost;
  KernelWrapper kernelWrapper;
  CublasWrapper cublasWrapper;

  ConfigurationMock configMock;
};

CudaSNPVectorFactoryTest::CudaSNPVectorFactoryTest() :
    numberOfIndividuals(6), device(0), streamFactory(), stream(streamFactory.constructStream(device)), hostToDevice(
        *stream), deviceToHost(*stream), kernelWrapper(*stream), cublasWrapper(*stream) {

}

CudaSNPVectorFactoryTest::~CudaSNPVectorFactoryTest() {

}

void CudaSNPVectorFactoryTest::SetUp() {

}

void CudaSNPVectorFactoryTest::TearDown() {

}

TEST_F(CudaSNPVectorFactoryTest, ConstructSNPVectorRegularHostVector) {
  GeneticModel geneticModel = DOMINANT;
  const int numberOfIndividuals = 6;
  EXPECT_CALL(configMock, getGeneticModel()).Times(1).WillRepeatedly(Return(geneticModel));
  CudaSNPVectorFactory cudaSNPVectorFactory(configMock, hostToDevice, kernelWrapper);

  SNP snp(Id("snp1"), "a1", "a2", 1);

  PinnedHostVector* snpData = new RegularHostVector(numberOfIndividuals);
  for(int i = 0; i < numberOfIndividuals; ++i){
    (*snpData)(i) = i % 2;
  }

  std::set<int>* snpMissingData = new std::set<int>();
  snpMissingData->insert(3);
  snpMissingData->insert(5);

  CudaSNPVector* snpVector = cudaSNPVectorFactory.constructSNPVector(snp, snpData, snpMissingData);
  EXPECT_EQ(snp, snpVector->getAssociatedSNP());

  const DeviceVector& snpOrgData = snpVector->getOriginalSNPData();
  const PinnedHostVector snpOrgDataHost = deviceToHost.transferVector(snpOrgData);
  ASSERT_EQ(numberOfIndividuals, snpOrgDataHost.getNumberOfRows());

  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*snpData)(i), (*snpOrgDataHost)(i));
  }

  EXPECT_TRUE(snpVector->hasMissing());
  const std::set<int>& missingData = snpVector->getMissing();
  EXPECT_EQ(snpMissingData, &missingData);

  delete snpOrgDataHost;
  delete snpVector;
}

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */
