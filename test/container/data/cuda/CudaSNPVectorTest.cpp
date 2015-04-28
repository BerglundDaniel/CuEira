#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <set>

#include <SNP.h>
#include <Id.h>
#include <Recode.h>
#include <CudaSNPVector.h>
#include <HostVector.h>
#include <InvalidState.h>
#include <GeneticModel.h>
#include <RiskAllele.h>
#include <DeviceVector.h>
#include <PinnedHostVector.h>
#include <DeviceToHost.h>
#include <HostToDevice.h>
#include <Device.h>
#include <Stream.h>
#include <StreamFactory.h>
#include <KernelWrapper.h>
#include <CublasWrapper.h>

namespace CuEira {
namespace Container {
namespace CUDA {

/**
 * Test for testing CudaSNPVector
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaSNPVectorTest: public ::testing::Test {
protected:
  CudaSNPVectorTest();
  virtual ~CudaSNPVectorTest();
  virtual void SetUp();
  virtual void TearDown();

  Device device;
  StreamFactory streamFactory;
  Stream* stream;
  HostToDevice hostToDevice;
  DeviceToHost deviceToHost;
  KernelWrapper kernelWrapper;
  CublasWrapper cublasWrapper;

  const int numberOfIndividuals;
  DeviceVector* originalSNPData;
  std::set<int>* snpMissingData;
  PinnedHostVector originalSNPDataHost;
};

CudaSNPVectorTest::CudaSNPVectorTest() :
    numberOfIndividuals(6), device(0), streamFactory(), stream(streamFactory.constructStream(device)), hostToDevice(
        *stream), deviceToHost(*stream), kernelWrapper(*stream), cublasWrapper(*stream), originalSNPData(nullptr), snpMissingData(
        nullptr), originalSNPDataHost(numberOfIndividuals) {
  for(int i = 0; i < numberOfIndividuals; ++i){
    originalSNPDataHost(i) = i % 2;
  }
}

CudaSNPVectorTest::~CudaSNPVectorTest() {

}

void CudaSNPVectorTest::SetUp() {
  originalSNPData = hostToDevice.transferVector(originalSNPDataHost);
  stream->syncStream();

  snpMissingData = new std::set<int>();
  snpMissingData->insert(3);
  snpMissingData->insert(8);
}

void CudaSNPVectorTest::TearDown() {

}

#ifdef DEBUG
TEST_F(CudaSNPVectorTest, Exceptions){
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  CudaSNPVector cudaSNPVector(snp1, DOMINANT, originalSNPData, snpMissingData);
  EXPECT_THROW(cudaSNPVector.getSNPData(), InvalidState);
}
#endif

TEST_F(CudaSNPVectorTest, ConstructAndGetWithMissing) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  CudaSNPVector cudaSNPVector(snp1, DOMINANT, originalSNPData, snpMissingData);
  ASSERT_EQ(snp1, cudaSNPVector.getAssociatedSNP());

  const DeviceVector orgData = cudaSNPVector.getOriginalSNPData();
  EXCPECT_EQ(numberOfIndividuals, cudaSNPVector.getNumberOfIndividualsToInclude());
  EXCPECT_FALSE(cudaSNPVector.hasMissing());
  const std::set<int> missingSet = cudaSNPVector.getMissing();

  ASSERT_EQ(numberOfIndividuals, orgData.getNumberOfRows());
  ASSERT_EQ(originalSNPData->getMemoryPointer(), orgData.getMemoryPointer());

  EXPECT_EQ(snpMissingData, &missingSet);
}

TEST_F(CudaSNPVectorTest, ConstructAndGetWithoutMissing) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);

  delete snpMissingData;
  snpMissingData = new std::set<int>();

  CudaSNPVector cudaSNPVector(snp1, DOMINANT, originalSNPData, snpMissingData);
  ASSERT_EQ(snp1, cudaSNPVector.getAssociatedSNP());

  const DeviceVector orgData = cudaSNPVector.getOriginalSNPData();
  EXCPECT_EQ(numberOfIndividuals, cudaSNPVector.getNumberOfIndividualsToInclude());
  EXCPECT_TRUE(cudaSNPVector.hasMissing());
  ASSERT_EQ(numberOfIndividuals, orgData.getNumberOfRows());
  ASSERT_EQ(originalSNPData->getMemoryPointer(), orgData.getMemoryPointer());

  const std::set<int> missingSet = cudaSNPVector.getMissing();
  EXPECT_TRUE(missingSet.empty());
}

TEST_F(CudaSNPVectorTest, ReCodeAllRiskDominant) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_ONE);

  CudaSNPVector cudaSNPVector(snp1, DOMINANT, originalSNPData, snpMissingData);
  cudaSNPVector.recode(ALL_RISK);

  const DeviceVector& recodedData = cudaSNPVector.getSNPData();
  const PinnedHostVector* recodedDataHost = deviceToHost.transferVector(recodedData);
  stream->syncStream();

  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());
  ASSERT_EQ(ALLELE_ONE, snp1.getRiskAllele());

  EXCPECT_EQ(1, (*recodedDataHost)(0));
  EXCPECT_EQ(1, (*recodedDataHost)(1));
  EXCPECT_EQ(0, (*recodedDataHost)(2));
  EXCPECT_EQ(1, (*recodedDataHost)(3));
  EXCPECT_EQ(1, (*recodedDataHost)(4));
  EXCPECT_EQ(0, (*recodedDataHost)(5));

  delete recodedDataHost;
}

TEST_F(CudaSNPVectorTest, ReCodeEnvironmentDominant) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_TWO);

  CudaSNPVector cudaSNPVector(snp1, DOMINANT, originalSNPData, snpMissingData);
  cudaSNPVector.recode(ENVIRONMENT_PROTECT);

  const DeviceVector& recodedData = cudaSNPVector.getSNPData();
  const PinnedHostVector* recodedDataHost = deviceToHost.transferVector(recodedData);
  stream->syncStream();

  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());
  ASSERT_EQ(ALLELE_TWO, snp1.getRiskAllele());

  EXCPECT_EQ(0, (*recodedDataHost)(0));
  EXCPECT_EQ(1, (*recodedDataHost)(1));
  EXCPECT_EQ(1, (*recodedDataHost)(2));
  EXCPECT_EQ(0, (*recodedDataHost)(3));
  EXCPECT_EQ(1, (*recodedDataHost)(4));
  EXCPECT_EQ(1, (*recodedDataHost)(5));

  delete recodedDataHost;
}

TEST_F(CudaSNPVectorTest, ReCodeSNPDominant) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_ONE);

  CudaSNPVector cudaSNPVector(snp1, DOMINANT, originalSNPData, snpMissingData);
  cudaSNPVector.recode(SNP_PROTECT);

  const DeviceVector& recodedData = cudaSNPVector.getSNPData();
  const PinnedHostVector* recodedDataHost = deviceToHost.transferVector(recodedData);
  stream->syncStream();

  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());
  ASSERT_EQ(ALLELE_TWO, snp1.getRiskAllele());

  EXCPECT_EQ(0, (*recodedDataHost)(0));
  EXCPECT_EQ(1, (*recodedDataHost)(1));
  EXCPECT_EQ(1, (*recodedDataHost)(2));
  EXCPECT_EQ(0, (*recodedDataHost)(3));
  EXCPECT_EQ(1, (*recodedDataHost)(4));
  EXCPECT_EQ(1, (*recodedDataHost)(5));

  delete recodedDataHost;
}

TEST_F(CudaSNPVectorTest, ReCodeInteractionDominant) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_TWO);

  CudaSNPVector cudaSNPVector(snp1, DOMINANT, originalSNPData, snpMissingData);
  cudaSNPVector.recode(INTERACTION_PROTECT);

  const DeviceVector& recodedData = cudaSNPVector.getSNPData();
  const PinnedHostVector* recodedDataHost = deviceToHost.transferVector(recodedData);
  stream->syncStream();

  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());
  ASSERT_EQ(ALLELE_ONE, snp1.getRiskAllele());

  EXCPECT_EQ(1, (*recodedDataHost)(0));
  EXCPECT_EQ(1, (*recodedDataHost)(1));
  EXCPECT_EQ(0, (*recodedDataHost)(2));
  EXCPECT_EQ(1, (*recodedDataHost)(3));
  EXCPECT_EQ(1, (*recodedDataHost)(4));
  EXCPECT_EQ(0, (*recodedDataHost)(5));

  delete recodedDataHost;
}

TEST_F(CudaSNPVectorTest, ReCodeAllRiskRecessive) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_ONE);

  CudaSNPVector cudaSNPVector(snp1, RECESSIVE, originalSNPData, snpMissingData);
  cudaSNPVector.recode(ALL_RISK);

  const DeviceVector& recodedData = cudaSNPVector.getSNPData();
  const PinnedHostVector* recodedDataHost = deviceToHost.transferVector(recodedData);
  stream->syncStream();

  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());
  ASSERT_EQ(ALLELE_ONE, snp1.getRiskAllele());

  EXCPECT_EQ(1, (*recodedDataHost)(0));
  EXCPECT_EQ(0, (*recodedDataHost)(1));
  EXCPECT_EQ(0, (*recodedDataHost)(2));
  EXCPECT_EQ(1, (*recodedDataHost)(3));
  EXCPECT_EQ(0, (*recodedDataHost)(4));
  EXCPECT_EQ(0, (*recodedDataHost)(5));

  delete recodedDataHost;
}

TEST_F(CudaSNPVectorTest, ReCodeEnvironmentRecessive) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_TWO);

  CudaSNPVector cudaSNPVector(snp1, RECESSIVE, originalSNPData, snpMissingData);
  cudaSNPVector.recode(ENVIRONMENT_PROTECT);

  const DeviceVector& recodedData = cudaSNPVector.getSNPData();
  const PinnedHostVector* recodedDataHost = deviceToHost.transferVector(recodedData);
  stream->syncStream();

  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());
  ASSERT_EQ(ALLELE_TWO, snp1.getRiskAllele());

  EXCPECT_EQ(0, (*recodedDataHost)(0));
  EXCPECT_EQ(0, (*recodedDataHost)(1));
  EXCPECT_EQ(1, (*recodedDataHost)(2));
  EXCPECT_EQ(0, (*recodedDataHost)(3));
  EXCPECT_EQ(0, (*recodedDataHost)(4));
  EXCPECT_EQ(1, (*recodedDataHost)(5));

  delete recodedDataHost;
}

TEST_F(CudaSNPVectorTest, ReCodeSNPRecessive) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_ONE);

  CudaSNPVector cudaSNPVector(snp1, RECESSIVE, originalSNPData, snpMissingData);
  cudaSNPVector.recode(SNP_PROTECT);

  const DeviceVector& recodedData = cudaSNPVector.getSNPData();
  const PinnedHostVector* recodedDataHost = deviceToHost.transferVector(recodedData);
  stream->syncStream();

  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());
  ASSERT_EQ(ALLELE_TWO, snp1.getRiskAllele());

  EXCPECT_EQ(0, (*recodedDataHost)(0));
  EXCPECT_EQ(0, (*recodedDataHost)(1));
  EXCPECT_EQ(1, (*recodedDataHost)(2));
  EXCPECT_EQ(0, (*recodedDataHost)(3));
  EXCPECT_EQ(0, (*recodedDataHost)(4));
  EXCPECT_EQ(1, (*recodedDataHost)(5));

  delete recodedDataHost;
}

TEST_F(CudaSNPVectorTest, ReCodeInteractionRecessive) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_TWO);

  CudaSNPVector cudaSNPVector(snp1, RECESSIVE, originalSNPData, snpMissingData);
  cudaSNPVector.recode(INTERACTION_PROTECT);

  const DeviceVector& recodedData = cudaSNPVector.getSNPData();
  const PinnedHostVector* recodedDataHost = deviceToHost.transferVector(recodedData);
  stream->syncStream();

  ASSERT_EQ(numberOfIndividuals, recodedData.getNumberOfRows());
  ASSERT_EQ(ALLELE_ONE, snp1.getRiskAllele());

  EXCPECT_EQ(1, (*recodedDataHost)(0));
  EXCPECT_EQ(0, (*recodedDataHost)(1));
  EXCPECT_EQ(0, (*recodedDataHost)(2));
  EXCPECT_EQ(1, (*recodedDataHost)(3));
  EXCPECT_EQ(0, (*recodedDataHost)(4));
  EXCPECT_EQ(0, (*recodedDataHost)(5));

  delete recodedDataHost;
}

TEST_F(CudaSNPVectorTest, ReCodeDominantStack) {
  Id id1("SNP1");
  unsigned int pos1 = 1;
  std::string alleOneString1("a1_1");
  std::string alleTwoString1("a1_2");
  SNP snp1(id1, alleOneString1, alleTwoString1, pos1);
  snp1.setRiskAllele(ALLELE_ONE);

  CudaSNPVector cudaSNPVector(snp1, DOMINANT, originalSNPData, snpMissingData);

  cudaSNPVector.recode(ALL_RISK);
  const DeviceVector& recodedData1 = cudaSNPVector.getSNPData();
  const PinnedHostVector* recodedDataHost1 = deviceToHost.transferVector(recodedData);
  stream->syncStream();
  ASSERT_EQ(ALLELE_ONE, snp1.getRiskAllele());

  EXCPECT_EQ(1, (*recodedDataHost1)(0));
  EXCPECT_EQ(1, (*recodedDataHost1)(1));
  EXCPECT_EQ(0, (*recodedDataHost1)(2));
  EXCPECT_EQ(1, (*recodedDataHost1)(3));
  EXCPECT_EQ(1, (*recodedDataHost1)(4));
  EXCPECT_EQ(0, (*recodedDataHost1)(5));

  cudaSNPVector.recode(ALL_RISK);
  const DeviceVector& recodedData2 = cudaSNPVector.getSNPData();
  const PinnedHostVector* recodedDataHost2 = deviceToHost.transferVector(recodedData);
  stream->syncStream();
  ASSERT_EQ(ALLELE_ONE, snp1.getRiskAllele());

  EXCPECT_EQ(1, (*recodedDataHost2)(0));
  EXCPECT_EQ(1, (*recodedDataHost2)(1));
  EXCPECT_EQ(0, (*recodedDataHost2)(2));
  EXCPECT_EQ(1, (*recodedDataHost2)(3));
  EXCPECT_EQ(1, (*recodedDataHost2)(4));
  EXCPECT_EQ(0, (*recodedDataHost2)(5));

  cudaSNPVector.recode(SNP_PROTECT);
  const DeviceVector& recodedData3 = cudaSNPVector.getSNPData();
  const PinnedHostVector* recodedDataHost3 = deviceToHost.transferVector(recodedData);
  stream->syncStream();
  ASSERT_EQ(ALLELE_TWO, snp1.getRiskAllele());

  EXCPECT_EQ(0, (*recodedDataHost3)(0));
  EXCPECT_EQ(1, (*recodedDataHost3)(1));
  EXCPECT_EQ(1, (*recodedDataHost3)(2));
  EXCPECT_EQ(0, (*recodedDataHost3)(3));
  EXCPECT_EQ(1, (*recodedDataHost3)(4));
  EXCPECT_EQ(1, (*recodedDataHost3)(5));

  cudaSNPVector.recode(INTERACTION_PROTECT);
  const DeviceVector& recodedData4 = cudaSNPVector.getSNPData();
  const PinnedHostVector* recodedDataHost4 = deviceToHost.transferVector(recodedData);
  stream->syncStream();
  ASSERT_EQ(ALLELE_TWO, snp1.getRiskAllele());

  EXCPECT_EQ(0, (*recodedDataHost4)(0));
  EXCPECT_EQ(1, (*recodedDataHost4)(1));
  EXCPECT_EQ(1, (*recodedDataHost4)(2));
  EXCPECT_EQ(0, (*recodedDataHost4)(3));
  EXCPECT_EQ(1, (*recodedDataHost4)(4));
  EXCPECT_EQ(1, (*recodedDataHost4)(5));

  delete recodedDataHost1;
  delete recodedDataHost2;
  delete recodedDataHost3;
  delete recodedDataHost4;
}

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */
