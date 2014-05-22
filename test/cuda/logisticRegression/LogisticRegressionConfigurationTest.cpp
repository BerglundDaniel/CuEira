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

using testing::Return;
using testing::_;
using testing::AtLeast;

namespace CuEira {
namespace CUDA {
namespace LogisticRegression {

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

  Container::PinnedHostVector outcomes;
  Container::PinnedHostMatrix covariates;
};

LogisticRegressionConfigurationTest::LogisticRegressionConfigurationTest() :
    convergenceThreshold(10e-5), numberOfMaxLRIterations(500), numberOfRows(10), numberOfCov(3), numberOfPredictorsNoCov(
        4), numberOfPredictorsWithCov(numberOfPredictorsNoCov + numberOfCov), outcomes(numberOfRows), covariates(
        numberOfRows, numberOfCov), cublasStatus(cublasCreate(&cublasHandle)), hostToDeviceStream1(
        HostToDevice(stream1)), deviceToHostStream1(DeviceToHost(stream1)) {

  EXPECT_CALL(configMock, getLRConvergenceThreshold()).Times(AtLeast(0)).WillRepeatedly(Return(convergenceThreshold));
  EXPECT_CALL(configMock, getNumberOfMaxLRIterations()).Times(AtLeast(0)).WillRepeatedly(
      Return(numberOfMaxLRIterations));

  for(int i = 0; i < numberOfRows; ++i){
    outcomes(i) = i;
  }

  for(int j = 0; j < numberOfCov; ++j){
    for(int i = 0; i < numberOfRows; ++i){
      covariates(i, j) = j * numberOfRows + i;
    }
  }
}

LogisticRegressionConfigurationTest::~LogisticRegressionConfigurationTest() {

}

void LogisticRegressionConfigurationTest::SetUp() {

}

void LogisticRegressionConfigurationTest::TearDown() {

}

TEST_F(LogisticRegressionConfigurationTest, Constructor) {
  LogisticRegressionConfiguration lrConfig(configMock, hostToDeviceMock, outcomes);

  EXPECT_EQ(numberOfMaxLRIterations, lrConfig.getNumberOfMaxIterations());
  EXPECT_EQ(convergenceThreshold, lrConfig.getConvergenceThreshold());
}

//test constructor covariates

//test constructor without mock

//test constructor without mock cov

//test snp without mock

//test snp cov without mock

//test env without mock

//test env cov without mock

//test inter without mock

//test inter cov without mock

} /* namespace LogisticRegression */
} /* namespace CUDA */
} /* namespace CUDA */

