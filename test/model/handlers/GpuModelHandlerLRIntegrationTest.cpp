#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <sstream>
#include <ostream>

#include <HostVector.h>
#include <HostMatrix.h>
#include <DataHandler.h>
#include <LogisticRegression.h>
#include <LogisticRegressionConfiguration.h>
#include <LogisticRegressionResult.h>
#include <SNP.h>
#include <Id.h>
#include <EnvironmentFactor.h>
#include <GpuModelHandler.h>
#include <ModelHandler.h>
#include <Statistics.h>
#include <StatisticsFactory.h>
#include <PinnedHostMatrix.h>
#include <PinnedHostVector.h>
#include <DeviceMatrix.h>
#include <DeviceVector.h>
#include <HostToDevice.h>
#include <DeviceToHost.h>

using testing::Return;
using testing::ReturnRef;
using testing::_;
using testing::Ge;
using testing::Eq;
using testing::Le;
using testing::ByRef;
using testing::InSequence;

namespace CuEira {
namespace Model {

/**
 * Integration test for GpuModelHandler, LogisticRegressionConfiguration, LogisticRegression, LogisticRegressionResult, Statistics
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class GpuModelHandlerLRIntegrationTest: public ::testing::Test {
protected:
  GpuModelHandlerLRIntegrationTest();
  virtual ~GpuModelHandlerLRIntegrationTest();
  virtual void SetUp();
  virtual void TearDown();

};

GpuModelHandlerLRIntegrationTest::GpuModelHandlerLRIntegrationTest() {

}

GpuModelHandlerLRIntegrationTest::~GpuModelHandlerLRIntegrationTest() {

}

void GpuModelHandlerLRIntegrationTest::SetUp() {

}

void GpuModelHandlerLRIntegrationTest::TearDown() {

}

TEST_F(GpuModelHandlerLRIntegrationTest, 2Env_2SNP) {
  const int numberOfIndividuals = 10;

  cudaStream_t cudaStream;
  cublasHandle_t cublasHandle;
  CUDA::handleCublasStatus(cublasCreate(&cublasHandle), "Failed to create cublas handle:");
  CUDA::handleCudaStatus(cudaStreamCreate(&cudaStream), "Failed to create cudaStream:");

  CUDA::HostToDevice hostToDevice(cudaStream);
  CUDA::DeviceToHost deviceToHost(cudaStream);
  ModelStatisticsFactory StatisticsFactory;
  CUDA::KernelWrapper kernelWrapper(cudaStream, cublasHandle);

  Container::PinnedHostVector hostOutcomes(numberOfIndividuals);
  Container::PinnedHostVector hostSnp1(numberOfIndividuals);
  Container::PinnedHostVector hostSnp2(numberOfIndividuals);
  Container::PinnedHostVector hostEnv1(numberOfIndividuals);
  Container::PinnedHostVector hostEnv2(numberOfIndividuals);

  //Outcomes
  hostOutcomes(0) = 1;
  hostOutcomes(1) = 1;
  hostOutcomes(2) = 0;
  hostOutcomes(3) = 0;
  hostOutcomes(4) = 1;
  hostOutcomes(5) = 0;
  hostOutcomes(6) = 1;
  hostOutcomes(7) = 0;
  hostOutcomes(8) = 1;
  hostOutcomes(9) = 1;

  //SNP1
  hostSnp1(0) = 1;
  hostSnp1(1) = 0;
  hostSnp1(2) = 0;
  hostSnp1(3) = 1;
  hostSnp1(4) = 0;
  hostSnp1(5) = 0;
  hostSnp1(6) = 1;
  hostSnp1(7) = 1;
  hostSnp1(8) = 0;
  hostSnp1(9) = 1;

  //SNP2
  hostSnp2(0) = 0;
  hostSnp2(1) = 0;
  hostSnp2(2) = 0;
  hostSnp2(3) = 1;
  hostSnp2(4) = 1;
  hostSnp2(5) = 0;
  hostSnp2(6) = 1;
  hostSnp2(7) = 1;
  hostSnp2(8) = 1;
  hostSnp2(9) = 0;

  //ENV1
  hostEnv1(0) = 0;
  hostEnv1(1) = 1;
  hostEnv1(2) = 0;
  hostEnv1(3) = 1;
  hostEnv1(4) = 1;
  hostEnv1(5) = 0;
  hostEnv1(6) = 1;
  hostEnv1(7) = 0;
  hostEnv1(8) = 1;
  hostEnv1(9) = 0;

  //ENV2
  hostEnv2(0) = 0.2;
  hostEnv2(1) = -0.1;
  hostEnv2(2) = 10;
  hostEnv2(3) = 5.6;
  hostEnv2(4) = 1.3;
  hostEnv2(5) = 0;
  hostEnv2(6) = -2;
  hostEnv2(7) = -1.1;
  hostEnv2(8) = 1;
  hostEnv2(9) = 7.2;

  Container::DeviceVector* deviceOutcomes = hostToDevice.transferVector(&hostOutcomes);

  DataHandler dataHandler(ADDITIVE, bedReaderMock, environmentInformation, dataQueue, environmentVector,
      interactionVector);

  Model::LogisticRegression::LogisticRegressionConfiguration =
      new Model::LogisticRegression::LogisticRegressionConfiguration(configuration, hostToDevice, *deviceOutcomes,
          kernelWrapper);

  GpuModelHandler gpuModelHandler(statisticsFactory, dataHandler, logisticRegressionConfiguration,
      logisticRegressionMock);

  while(gpuModelHandler.next()){
    InteractionStatistics* statistics = gpuModelHandler.calculateModel();
    const Container::SNPVector& snpVector = gpuModelHandler.getSNPVector();
    const SNP& snp = gpuModelHandler.getCurrentSNP();
    const EnvironmentFactor& envFactor = gpuModelHandler.getCurrentEnvironmentFactor();

    //TODO check results

    delete statistics;
  }

  CUDA::handleCudaStatus(cudaStreamDestroy(cudaStream), "Failed to destroy cudaStream:");
  CUDA::handleCublasStatus(cublasDestroy(cublasHandle), "Failed to destroy cublas handle:");

  delete deviceOutcomes;
}

}
/* namespace Model */
} /* namespace CuEira */
