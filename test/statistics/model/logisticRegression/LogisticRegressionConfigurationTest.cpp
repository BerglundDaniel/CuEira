#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <CudaAdapter.cu>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <HostVector.h>
#include <HostMatrix.h>
#include <PinnedHostMatrix.h>
#include <PinnedHostVector.h>
#include <DeviceToHost.h>
#include <HostToDevice.h>
#include <HostToDeviceMock.h>
#include <LogisticRegressionConfiguration.h>
#include <Configuration.h>
#include <ConfigurationMock.h>
#include <KernelWrapper.h>

using testing::Return;
using testing::_;
using testing::AtLeast;

namespace CuEira {
namespace Model {
namespace LogisticRegression {

using namespace CuEira::Container;
using namespace CuEira::CUDA;

/**
 * Test for testing transfers between host and device in both directions.
 * Assumes that the container classes are working.
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LogisticRegressionConfigurationTest: public ::testing::Test {
protected:
  LogisticRegressionConfigurationTest();
  virtual ~LogisticRegressionConfigurationTest();
  virtual void SetUp();
  virtual void TearDown();

  ConfigurationMock configMock;
  const double convergenceThreshold;
  const int numberOfMaxLRIterations;
  const int numberOfRows;
  const int numberOfCov;
  const int numberOfPredictorsNoCov;
  const int numberOfPredictorsWithCov;

  cublasStatus_t cublasStatus;
  cudaStream_t stream1;
  cublasHandle_t cublasHandle;
  HostToDevice hostToDeviceStream1;
  DeviceToHost deviceToHostStream1;
  HostToDeviceMock hostToDeviceMock;
  KernelWrapper kernelWrapper;

  PinnedHostVector outcomes;
  PinnedHostMatrix covariates;
  PinnedHostVector snpData;
  PinnedHostVector environmentData;
  PinnedHostVector interactionVector;
};

LogisticRegressionConfigurationTest::LogisticRegressionConfigurationTest() :
    convergenceThreshold(10e-5), numberOfMaxLRIterations(500), numberOfRows(10), numberOfCov(3), numberOfPredictorsNoCov(
        4), numberOfPredictorsWithCov(numberOfPredictorsNoCov + numberOfCov), outcomes(numberOfRows), covariates(
        numberOfRows, numberOfCov), cublasStatus(cublasCreate(&cublasHandle)), hostToDeviceStream1(
        HostToDevice(stream1)), deviceToHostStream1(DeviceToHost(stream1)), snpData(numberOfRows), environmentData(
        numberOfRows), interactionVector(numberOfRows), kernelWrapper(stream1, cublasHandle) {

  handleCublasStatus(cublasStatus, "Failed to create cublas handle:");
  handleCudaStatus(cudaGetLastError(), "Error with LR config test setup: ");
  handleCudaStatus(cudaStreamCreate(&stream1), "Failed to create cuda stream 1:");
  handleCublasStatus(cublasSetStream(cublasHandle, stream1), "Failed to set cuda stream:");

  EXPECT_CALL(configMock, getLRConvergenceThreshold()).Times(AtLeast(0)).WillRepeatedly(Return(convergenceThreshold));
  EXPECT_CALL(configMock, getNumberOfMaxLRIterations()).Times(AtLeast(0)).WillRepeatedly(
      Return(numberOfMaxLRIterations));

  for(int i = 0; i < numberOfRows; ++i){
    outcomes(i) = i;

    snpData(i) = i * 10;
    environmentData(i) = i * 100;
    interactionVector(i) = i * 1000;
  }

  for(int j = 0; j < numberOfCov; ++j){
    for(int i = 0; i < numberOfRows; ++i){
      covariates(i, j) = (j * numberOfRows) + i;
    }
  }
}

LogisticRegressionConfigurationTest::~LogisticRegressionConfigurationTest() {
  handleCudaStatus(cudaStreamDestroy(stream1), "Failed to destroy cuda stream 1:");
  handleCublasStatus(cublasDestroy(cublasHandle), "Failed to destroy cublas handle:");
}

void LogisticRegressionConfigurationTest::SetUp() {

}

void LogisticRegressionConfigurationTest::TearDown() {

}

TEST_F(LogisticRegressionConfigurationTest, ConstructorWithMock) {
  DeviceVector* outcomeDeviceVector = new DeviceVector(numberOfRows);

  EXPECT_CALL(hostToDeviceMock, transferVector(_,_)).Times(1);

  LogisticRegressionConfiguration lrConfig(configMock, hostToDeviceMock, *outcomeDeviceVector, kernelWrapper);
  cudaStreamSynchronize(stream1);
  handleCudaStatus(cudaGetLastError(), "Error with LR config in test: ");

  EXPECT_EQ(numberOfMaxLRIterations, lrConfig.getNumberOfMaxIterations());
  EXPECT_EQ(convergenceThreshold, lrConfig.getConvergenceThreshold());
  EXPECT_EQ(numberOfRows, lrConfig.getNumberOfRows());
  EXPECT_EQ(numberOfPredictorsNoCov, lrConfig.getNumberOfPredictors());

  const DeviceVector& outcomeDeviceLR = lrConfig.getOutcomes();
  ASSERT_EQ(numberOfRows, outcomeDeviceLR.getNumberOfRows());
  ASSERT_EQ(outcomeDeviceVector->getMemoryPointer(), outcomeDeviceLR.getMemoryPointer());

  delete outcomeDeviceVector;
}

TEST_F(LogisticRegressionConfigurationTest, ConstructorCovWithMock) {
  DeviceVector* outcomeDeviceVector = new DeviceVector(numberOfRows);

  EXPECT_CALL(hostToDeviceMock, transferVector(_,_)).Times(1);
  EXPECT_CALL(hostToDeviceMock, transferMatrix(_,_)).Times(1);

  LogisticRegressionConfiguration lrConfig(configMock, hostToDeviceMock, *outcomeDeviceVector, kernelWrapper,
      covariates);
  cudaStreamSynchronize(stream1);
  handleCudaStatus(cudaGetLastError(), "Error with LR config in test: ");

  EXPECT_EQ(numberOfMaxLRIterations, lrConfig.getNumberOfMaxIterations());
  EXPECT_EQ(convergenceThreshold, lrConfig.getConvergenceThreshold());
  EXPECT_EQ(numberOfRows, lrConfig.getNumberOfRows());
  EXPECT_EQ(numberOfPredictorsWithCov, lrConfig.getNumberOfPredictors());

  const DeviceVector& outcomeDeviceLR = lrConfig.getOutcomes();
  ASSERT_EQ(numberOfRows, outcomeDeviceLR.getNumberOfRows());
  ASSERT_EQ(outcomeDeviceVector->getMemoryPointer(), outcomeDeviceLR.getMemoryPointer());

  delete outcomeDeviceVector;
}

TEST_F(LogisticRegressionConfigurationTest, ConstructorNoMock) {
  DeviceVector* outcomeDeviceVector = hostToDeviceStream1.transferVector(&outcomes);
  cudaStreamSynchronize(stream1);

  LogisticRegressionConfiguration lrConfig(configMock, hostToDeviceStream1, *outcomeDeviceVector, kernelWrapper);
  cudaStreamSynchronize(stream1);
  handleCudaStatus(cudaGetLastError(), "Error with LR config in test: ");

  EXPECT_EQ(numberOfMaxLRIterations, lrConfig.getNumberOfMaxIterations());
  EXPECT_EQ(convergenceThreshold, lrConfig.getConvergenceThreshold());
  EXPECT_EQ(numberOfRows, lrConfig.getNumberOfRows());
  EXPECT_EQ(numberOfPredictorsNoCov, lrConfig.getNumberOfPredictors());

  const DeviceVector& outcomeDeviceVectorOut = lrConfig.getOutcomes();
  HostVector* outcomeHostVector = deviceToHostStream1.transferVector(&outcomeDeviceVectorOut);
  cudaStreamSynchronize(stream1);

  ASSERT_EQ(numberOfRows, outcomeHostVector->getNumberOfRows());
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i, (*outcomeHostVector)(i));
  }

  const DeviceMatrix& predictorsDeviceMatrix = lrConfig.getPredictors();
  HostMatrix* predictorsHostMatrix = deviceToHostStream1.transferMatrix(&predictorsDeviceMatrix);
  cudaStreamSynchronize(stream1);
  ASSERT_EQ(numberOfRows, predictorsHostMatrix->getNumberOfRows());
  ASSERT_EQ(numberOfPredictorsNoCov, predictorsHostMatrix->getNumberOfColumns());

  //Intercept
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(1, (*predictorsHostMatrix)(i, 0));
  }

  //Beta
  const DeviceVector& betaDevice = lrConfig.getBetaCoefficents();
  HostVector* betaHost = deviceToHostStream1.transferVector(&betaDevice);
  cudaStreamSynchronize(stream1);

  ASSERT_EQ(numberOfPredictorsNoCov, betaHost->getNumberOfRows());
  for(int i = 0; i < numberOfPredictorsNoCov; ++i){
    EXPECT_EQ(0, (*betaHost)(i));
  }

  delete outcomeDeviceVector;
  delete betaHost;
}

TEST_F(LogisticRegressionConfigurationTest, ConstructorCovNoMock) {
  DeviceVector* outcomeDeviceVector = hostToDeviceStream1.transferVector(&outcomes);
  cudaStreamSynchronize(stream1);

  LogisticRegressionConfiguration lrConfig(configMock, hostToDeviceStream1, *outcomeDeviceVector, kernelWrapper,
      covariates);
  cudaStreamSynchronize(stream1);
  handleCudaStatus(cudaGetLastError(), "Error with LR config in test: ");

  EXPECT_EQ(numberOfMaxLRIterations, lrConfig.getNumberOfMaxIterations());
  EXPECT_EQ(convergenceThreshold, lrConfig.getConvergenceThreshold());
  EXPECT_EQ(numberOfRows, lrConfig.getNumberOfRows());
  EXPECT_EQ(numberOfPredictorsWithCov, lrConfig.getNumberOfPredictors());

  const DeviceVector& outcomeDeviceVectorOut = lrConfig.getOutcomes();
  HostVector* outcomeHostVector = deviceToHostStream1.transferVector(&outcomeDeviceVectorOut);
  cudaStreamSynchronize(stream1);

  ASSERT_EQ(numberOfRows, outcomeHostVector->getNumberOfRows());
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i, (*outcomeHostVector)(i));
  }

  const DeviceMatrix& predictorsDeviceMatrix = lrConfig.getPredictors();
  HostMatrix* predictorsHostMatrix = deviceToHostStream1.transferMatrix(&predictorsDeviceMatrix);
  cudaStreamSynchronize(stream1);
  ASSERT_EQ(numberOfRows, predictorsHostMatrix->getNumberOfRows());
  ASSERT_EQ(numberOfPredictorsWithCov, predictorsHostMatrix->getNumberOfColumns());

  //Intercept
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(1, (*predictorsHostMatrix)(i, 0));
  }

  //Covariates
  for(int j = 0; j < numberOfCov; ++j){
    for(int i = 0; i < numberOfRows; ++i){
      EXPECT_EQ((j * numberOfRows + i), (*predictorsHostMatrix)(i, j + 4));
    }
  }

  //Beta
  const DeviceVector& betaDevice = lrConfig.getBetaCoefficents();
  HostVector* betaHost = deviceToHostStream1.transferVector(&betaDevice);
  cudaStreamSynchronize(stream1);

  ASSERT_EQ(numberOfPredictorsWithCov, betaHost->getNumberOfRows());
  for(int i = 0; i < numberOfPredictorsWithCov; ++i){
    EXPECT_EQ(0, (*betaHost)(i));
  }

  delete outcomeDeviceVector;
  delete betaHost;
}

TEST_F(LogisticRegressionConfigurationTest, GetSNPEnvInteract) {
  DeviceVector* outcomeDeviceVector = hostToDeviceStream1.transferVector(&outcomes);
  cudaStreamSynchronize(stream1);

  LogisticRegressionConfiguration lrConfig(configMock, hostToDeviceStream1, *outcomeDeviceVector, kernelWrapper);
  cudaStreamSynchronize(stream1);
  handleCudaStatus(cudaGetLastError(), "Error with LR config in test: ");

  EXPECT_EQ(numberOfMaxLRIterations, lrConfig.getNumberOfMaxIterations());
  EXPECT_EQ(convergenceThreshold, lrConfig.getConvergenceThreshold());
  EXPECT_EQ(numberOfRows, lrConfig.getNumberOfRows());
  EXPECT_EQ(numberOfPredictorsNoCov, lrConfig.getNumberOfPredictors());

  //Set snp
  lrConfig.setSNP(snpData);

  //Set env
  lrConfig.setEnvironmentFactor(environmentData);

  //Set interact
  lrConfig.setInteraction(interactionVector);
  cudaStreamSynchronize(stream1);

  const DeviceVector& outcomeDeviceVectorOut = lrConfig.getOutcomes();
  HostVector* outcomeHostVector = deviceToHostStream1.transferVector(&outcomeDeviceVectorOut);
  cudaStreamSynchronize(stream1);

  ASSERT_EQ(numberOfRows, outcomeHostVector->getNumberOfRows());
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i, (*outcomeHostVector)(i));
  }

  const DeviceMatrix& predictorsDeviceMatrix = lrConfig.getPredictors();
  HostMatrix* predictorsHostMatrix = deviceToHostStream1.transferMatrix(&predictorsDeviceMatrix);
  cudaStreamSynchronize(stream1);
  ASSERT_EQ(numberOfRows, predictorsHostMatrix->getNumberOfRows());
  ASSERT_EQ(numberOfPredictorsNoCov, predictorsHostMatrix->getNumberOfColumns());

  //Intercept
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(1, (*predictorsHostMatrix)(i, 0));
  }

  //SNP
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i * 10, (*predictorsHostMatrix)(i, 1));
  }

  //Env
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i * 100, (*predictorsHostMatrix)(i, 2));
  }

  //Interact
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i * 1000, (*predictorsHostMatrix)(i, 3));
  }

  delete outcomeDeviceVector;
}

TEST_F(LogisticRegressionConfigurationTest, CovGetSNPEnvInteract) {
  DeviceVector* outcomeDeviceVector = hostToDeviceStream1.transferVector(&outcomes);
  cudaStreamSynchronize(stream1);

  LogisticRegressionConfiguration lrConfig(configMock, hostToDeviceStream1, *outcomeDeviceVector, kernelWrapper,
      covariates);
  cudaStreamSynchronize(stream1);
  handleCudaStatus(cudaGetLastError(), "Error with LR config in test: ");

  EXPECT_EQ(numberOfMaxLRIterations, lrConfig.getNumberOfMaxIterations());
  EXPECT_EQ(convergenceThreshold, lrConfig.getConvergenceThreshold());
  EXPECT_EQ(numberOfRows, lrConfig.getNumberOfRows());
  EXPECT_EQ(numberOfPredictorsWithCov, lrConfig.getNumberOfPredictors());

  //Set snp
  lrConfig.setSNP(snpData);

  //Set env
  lrConfig.setEnvironmentFactor(environmentData);

  //Set interact
  lrConfig.setInteraction(interactionVector);
  cudaStreamSynchronize(stream1);

  const DeviceVector& outcomeDeviceVectorOut = lrConfig.getOutcomes();
  HostVector* outcomeHostVector = deviceToHostStream1.transferVector(&outcomeDeviceVectorOut);
  cudaStreamSynchronize(stream1);

  ASSERT_EQ(numberOfRows, outcomeHostVector->getNumberOfRows());
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i, (*outcomeHostVector)(i));
  }

  const DeviceMatrix& predictorsDeviceMatrix = lrConfig.getPredictors();
  HostMatrix* predictorsHostMatrix = deviceToHostStream1.transferMatrix(&predictorsDeviceMatrix);
  cudaStreamSynchronize(stream1);
  ASSERT_EQ(numberOfRows, predictorsHostMatrix->getNumberOfRows());
  ASSERT_EQ(numberOfPredictorsWithCov, predictorsHostMatrix->getNumberOfColumns());

  //Intercept
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(1, (*predictorsHostMatrix)(i, 0));
  }

  //SNP
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i * 10, (*predictorsHostMatrix)(i, 1));
  }

  //Env
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i * 100, (*predictorsHostMatrix)(i, 2));
  }

  //Interact
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i * 1000, (*predictorsHostMatrix)(i, 3));
  }

  //Covariates
  for(int j = 0; j < numberOfCov; ++j){
    for(int i = 0; i < numberOfRows; ++i){
      EXPECT_EQ((j * numberOfRows + i), (*predictorsHostMatrix)(i, j + 4));
    }
  }

  delete outcomeDeviceVector;
}

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */
