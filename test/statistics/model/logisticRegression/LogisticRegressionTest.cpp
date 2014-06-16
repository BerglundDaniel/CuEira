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
#include <LogisticRegressionConfiguration.h>
#include <Configuration.h>
#include <ConfigurationMock.h>
#include <KernelWrapper.h>
#include <LogisticRegression.h>

using testing::Return;
using testing::_;
using testing::AtLeast;
using ::testing::Ge;
using ::testing::Le;

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
class LogisticRegressionTest: public ::testing::Test {
protected:
  LogisticRegressionTest();
  virtual ~LogisticRegressionTest();
  virtual void SetUp();
  virtual void TearDown();

  ConfigurationMock configMock;
  const double convergenceThreshold;
  const int numberOfMaxLRIterations;
  const int numberOfCov;
  const int numberOfPredictorsNoCov;
  const int numberOfPredictorsWithCov;

  cublasStatus_t cublasStatus;
  cudaStream_t stream1;
  cublasHandle_t cublasHandle;
  HostToDevice hostToDeviceStream1;
  DeviceToHost deviceToHostStream1;
  KernelWrapper kernelWrapper;
  double e;
};

LogisticRegressionTest::LogisticRegressionTest() :
    convergenceThreshold(10e-5), numberOfMaxLRIterations(500), numberOfCov(2), numberOfPredictorsNoCov(4), numberOfPredictorsWithCov(
        numberOfPredictorsNoCov + numberOfCov), cublasStatus(cublasCreate(&cublasHandle)), hostToDeviceStream1(
        HostToDevice(stream1)), deviceToHostStream1(DeviceToHost(stream1)), kernelWrapper(stream1, cublasHandle), e(
        1e-4) {

  handleCublasStatus(cublasStatus, "Failed to create cublas handle:");
  handleCudaStatus(cudaGetLastError(), "Error with LR config test setup: ");
  handleCudaStatus(cudaStreamCreate(&stream1), "Failed to create cuda stream 1:");
  handleCublasStatus(cublasSetStream(cublasHandle, stream1), "Failed to set cuda stream:");

  EXPECT_CALL(configMock, getLRConvergenceThreshold()).Times(AtLeast(0)).WillRepeatedly(Return(convergenceThreshold));
  EXPECT_CALL(configMock, getNumberOfMaxLRIterations()).Times(AtLeast(0)).WillRepeatedly(
      Return(numberOfMaxLRIterations));

}

LogisticRegressionTest::~LogisticRegressionTest() {
  handleCudaStatus(cudaStreamDestroy(stream1), "Failed to destroy cuda stream 1:");
  handleCublasStatus(cublasDestroy(cublasHandle), "Failed to destroy cublas handle:");
}

void LogisticRegressionTest::SetUp() {

}

void LogisticRegressionTest::TearDown() {

}

TEST_F(LogisticRegressionTest, SmallTestNoCov) {
  const int numberOfRows = 10;
  PinnedHostVector outcomes(numberOfRows);
  PinnedHostVector snpData(numberOfRows);
  PinnedHostVector environmentData(numberOfRows);
  PinnedHostVector interactionVector(numberOfRows);

  kernelWrapper.setSymbolNumberOfRows(numberOfRows);
  kernelWrapper.setSymbolNumberOfPredictors(numberOfPredictorsNoCov);

  //SNP
  snpData(0) = 1.33582291;
  snpData(1) = -0.21913482;
  snpData(2) = 0.29749252;
  snpData(3) = 0.49347861;
  snpData(4) = -0.57089565;
  snpData(5) = -1.03339458;
  snpData(6) = 0.11693107;
  snpData(7) = -0.38543587;
  snpData(8) = 0.25468775;
  snpData(9) = -0.69603999;

  //Env
  environmentData(0) = 1.4961856;
  environmentData(1) = -0.99393901;
  environmentData(2) = 1.16772209;
  environmentData(3) = 0.49370225;
  environmentData(4) = 0.76115578;
  environmentData(5) = -0.38176981;
  environmentData(6) = -0.92562295;
  environmentData(7) = -0.60920825;
  environmentData(8) = -0.62394504;
  environmentData(9) = 0.32976581;

  //Interaction
  interactionVector(0) = 1;
  interactionVector(1) = 1;
  interactionVector(2) = 1;
  interactionVector(3) = 1;
  interactionVector(4) = 1;
  interactionVector(5) = 0;
  interactionVector(6) = 0;
  interactionVector(7) = 0;
  interactionVector(8) = 0;
  interactionVector(9) = 0;

  outcomes(0) = 1;
  outcomes(1) = 1;
  outcomes(2) = 0;
  outcomes(3) = 0;
  outcomes(4) = 1;
  outcomes(5) = 0;
  outcomes(6) = 1;
  outcomes(7) = 0;
  outcomes(8) = 1;
  outcomes(9) = 1;

  std::vector<PRECISION> correctBeta(numberOfPredictorsNoCov);
  correctBeta[0] = -0.0065;
  correctBeta[1] = 0.5096;
  correctBeta[2] = 1.9418;
  correctBeta[3] = 1.8898;

  DeviceVector* outcomeDeviceVector = hostToDeviceStream1.transferVector(&outcomes);
  cudaStreamSynchronize(stream1);

  LogisticRegressionConfiguration lrConfig(configMock, hostToDeviceStream1, *outcomeDeviceVector, kernelWrapper);
  lrConfig.setSNP(snpData);
  lrConfig.setEnvironmentFactor(environmentData);
  lrConfig.setInteraction(interactionVector);
  cudaStreamSynchronize(stream1);
  handleCudaStatus(cudaGetLastError(), "Error with LR config in test: ");

  std::cerr << "t1" << std::endl;
  LogisticRegression logisticRegression(lrConfig, hostToDeviceStream1, deviceToHostStream1);
  std::cerr << "t2" << std::endl;
  HostVector* beta = logisticRegression.stealBeta();
  std::cerr << "t3" << std::endl;

  for(int i = 0; i < numberOfPredictorsNoCov; ++i){
    std::cerr << "t4" << std::endl;
    PRECISION l = correctBeta[i] - e;
    PRECISION h = correctBeta[i] + e;

    EXPECT_THAT((*beta)(i), Ge(l));
    EXPECT_THAT((*beta)(i), Le(h));
  }
  std::cerr << "t5" << std::endl;

  delete beta;
  delete outcomeDeviceVector;
}

TEST_F(LogisticRegressionTest, ConstructorCovNoMock) {
  /*PinnedHostVector outcomes;
   PinnedHostMatrix covariates;
   PinnedHostVector snpData;
   PinnedHostVector environmentData;
   PinnedHostVector interactionVector;

   //TODO set stuff

   DeviceVector* outcomeDeviceVector = hostToDeviceStream1.transferVector(&outcomes);
   cudaStreamSynchronize(stream1);

   LogisticRegressionConfiguration lrConfig(configMock, hostToDeviceStream1, *outcomeDeviceVector, kernelWrapper,
   covariates);
   lrConfig.setSNP(snpData);
   lrConfig.setEnvironmentFactor(environmentData);
   lrConfig.setInteraction(interactionVector);
   cudaStreamSynchronize(stream1);
   handleCudaStatus(cudaGetLastError(), "Error with LR config in test: ");

   LogisticRegression logisticRegression(lrConfig, hostToDeviceStream1, deviceToHostStream1);
   const HostVector& beta = logisticRegression.getBeta();

   delete betaHost;*/
}

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

