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
#include <DeviceToHostMock.h>
#include <HostToDeviceMock.h>
#include <CudaLogisticRegressionConfiguration.h>
#include <Configuration.h>
#include <ConfigurationMock.h>
#include <KernelWrapper.h>
#include <MKLWrapper.h>
#include <Device.h>
#include <Stream.h>
#include <StreamFactory.h>

using testing::Return;
using testing::_;
using testing::AtLeast;

namespace CuEira {
namespace Model {
namespace LogisticRegression {
namespace CUDA {

using namespace CuEira::Container;
using namespace CuEira::CUDA;

/**
 * Test for ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaLogisticRegressionConfigurationTest: public ::testing::Test {
protected:
  CudaLogisticRegressionConfigurationTest();
  virtual ~CudaLogisticRegressionConfigurationTest();
  virtual void SetUp();
  virtual void TearDown();

  ConfigurationMock configMock;
  const double convergenceThreshold;
  const int numberOfMaxLRIterations;
  const int numberOfRows;
  const int numberOfCov;
  const int numberOfPredictorsNoCov;
  const int numberOfPredictorsWithCov;

  Device device;
  StreamFactory streamFactory;
  Stream* stream;
  HostToDevice hostToDeviceStream1;
  DeviceToHost deviceToHostStream1;
  HostToDeviceMock hostToDeviceMock;
  DeviceToHostMock deviceToHostMock;
  KernelWrapper kernelWrapper;
  MKLWrapper blasWrapper;

  PinnedHostVector outcomes;
  PinnedHostMatrix covariates;
  PinnedHostVector snpData;
  PinnedHostVector environmentData;
  PinnedHostVector interactionVector;
};

CudaLogisticRegressionConfigurationTest::CudaLogisticRegressionConfigurationTest() :
    device(0), streamFactory(), stream(streamFactory.constructStream(device)), hostToDeviceStream1(*stream), deviceToHostStream1(
        *stream), convergenceThreshold(10e-5), numberOfMaxLRIterations(500), numberOfRows(10), numberOfCov(3), numberOfPredictorsNoCov(
        4), numberOfPredictorsWithCov(numberOfPredictorsNoCov + numberOfCov), outcomes(numberOfRows), covariates(
        numberOfRows, numberOfCov), snpData(numberOfRows), environmentData(numberOfRows), interactionVector(
        numberOfRows), kernelWrapper(*stream), blasWrapper() {

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

CudaLogisticRegressionConfigurationTest::~CudaLogisticRegressionConfigurationTest() {
  delete stream;
}

void CudaLogisticRegressionConfigurationTest::SetUp() {

}

void CudaLogisticRegressionConfigurationTest::TearDown() {

}

TEST_F(CudaLogisticRegressionConfigurationTest, ConstructorWithMock) {
  DeviceVector* outcomeDeviceVector = new DeviceVector(numberOfRows);

  EXPECT_CALL(hostToDeviceMock, transferVector(_,_)).Times(1);

  CudaLogisticRegressionConfiguration lrConfig(configMock, hostToDeviceMock, deviceToHostMock, *outcomeDeviceVector,
      kernelWrapper, blasWrapper);
  stream->syncStream();
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

TEST_F(CudaLogisticRegressionConfigurationTest, ConstructorCovWithMock) {
  DeviceVector* outcomeDeviceVector = new DeviceVector(numberOfRows);

  EXPECT_CALL(hostToDeviceMock, transferVector(_,_)).Times(1);
  EXPECT_CALL(hostToDeviceMock, transferMatrix(_,_)).Times(1);

  CudaLogisticRegressionConfiguration lrConfig(configMock, hostToDeviceMock, deviceToHostMock, *outcomeDeviceVector,
      kernelWrapper, blasWrapper, covariates);
  stream->syncStream();
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

TEST_F(CudaLogisticRegressionConfigurationTest, ConstructorNoMock) {
  DeviceVector* outcomeDeviceVector = hostToDeviceStream1.transferVector(outcomes);
  stream->syncStream();

  CudaLogisticRegressionConfiguration lrConfig(configMock, hostToDeviceStream1, deviceToHostStream1,
      *outcomeDeviceVector, kernelWrapper, blasWrapper);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error with LR config in test: ");

  EXPECT_EQ(numberOfMaxLRIterations, lrConfig.getNumberOfMaxIterations());
  EXPECT_EQ(convergenceThreshold, lrConfig.getConvergenceThreshold());
  EXPECT_EQ(numberOfRows, lrConfig.getNumberOfRows());
  EXPECT_EQ(numberOfPredictorsNoCov, lrConfig.getNumberOfPredictors());

  const DeviceVector& outcomeDeviceVectorOut = lrConfig.getOutcomes();
  HostVector* outcomeHostVector = deviceToHostStream1.transferVector(outcomeDeviceVectorOut);
  stream->syncStream();

  ASSERT_EQ(numberOfRows, outcomeHostVector->getNumberOfRows());
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i, (*outcomeHostVector)(i));
  }

  const DeviceMatrix& predictorsDeviceMatrix = lrConfig.getPredictors();
  HostMatrix* predictorsHostMatrix = deviceToHostStream1.transferMatrix(predictorsDeviceMatrix);
  stream->syncStream();
  ASSERT_EQ(numberOfRows, predictorsHostMatrix->getNumberOfRows());
  ASSERT_EQ(numberOfPredictorsNoCov, predictorsHostMatrix->getNumberOfColumns());

  //Intercept
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(1, (*predictorsHostMatrix)(i, 0));
  }

  delete outcomeDeviceVector;
}

TEST_F(CudaLogisticRegressionConfigurationTest, ConstructorCovNoMock) {
  DeviceVector* outcomeDeviceVector = hostToDeviceStream1.transferVector(outcomes);
  stream->syncStream();

  CudaLogisticRegressionConfiguration lrConfig(configMock, hostToDeviceStream1, deviceToHostStream1,
      *outcomeDeviceVector, kernelWrapper, blasWrapper, covariates);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error with LR config in test: ");

  EXPECT_EQ(numberOfMaxLRIterations, lrConfig.getNumberOfMaxIterations());
  EXPECT_EQ(convergenceThreshold, lrConfig.getConvergenceThreshold());
  EXPECT_EQ(numberOfRows, lrConfig.getNumberOfRows());
  EXPECT_EQ(numberOfPredictorsWithCov, lrConfig.getNumberOfPredictors());

  const DeviceVector& outcomeDeviceVectorOut = lrConfig.getOutcomes();
  HostVector* outcomeHostVector = deviceToHostStream1.transferVector(outcomeDeviceVectorOut);
  stream->syncStream();

  ASSERT_EQ(numberOfRows, outcomeHostVector->getNumberOfRows());
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i, (*outcomeHostVector)(i));
  }

  const DeviceMatrix& predictorsDeviceMatrix = lrConfig.getPredictors();
  HostMatrix* predictorsHostMatrix = deviceToHostStream1.transferMatrix(predictorsDeviceMatrix);
  stream->syncStream();
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

  delete outcomeDeviceVector;
}

TEST_F(CudaLogisticRegressionConfigurationTest, GetSNPEnvInteract) {
  DeviceVector* outcomeDeviceVector = hostToDeviceStream1.transferVector(outcomes);
  stream->syncStream();

  CudaLogisticRegressionConfiguration lrConfig(configMock, hostToDeviceStream1, deviceToHostStream1,
      *outcomeDeviceVector, kernelWrapper, blasWrapper);
  stream->syncStream();
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
  stream->syncStream();

  const DeviceVector& outcomeDeviceVectorOut = lrConfig.getOutcomes();
  HostVector* outcomeHostVector = deviceToHostStream1.transferVector(outcomeDeviceVectorOut);
  stream->syncStream();

  ASSERT_EQ(numberOfRows, outcomeHostVector->getNumberOfRows());
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i, (*outcomeHostVector)(i));
  }

  const DeviceMatrix& predictorsDeviceMatrix = lrConfig.getPredictors();
  HostMatrix* predictorsHostMatrix = deviceToHostStream1.transferMatrix(predictorsDeviceMatrix);
  stream->syncStream();
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

TEST_F(CudaLogisticRegressionConfigurationTest, CovGetSNPEnvInteract) {
  DeviceVector* outcomeDeviceVector = hostToDeviceStream1.transferVector(outcomes);
  stream->syncStream();

  CudaLogisticRegressionConfiguration lrConfig(configMock, hostToDeviceStream1, deviceToHostStream1,
      *outcomeDeviceVector, kernelWrapper, blasWrapper, covariates);
  stream->syncStream();
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
  stream->syncStream();

  const DeviceVector& outcomeDeviceVectorOut = lrConfig.getOutcomes();
  HostVector* outcomeHostVector = deviceToHostStream1.transferVector(outcomeDeviceVectorOut);
  stream->syncStream();

  ASSERT_EQ(numberOfRows, outcomeHostVector->getNumberOfRows());
  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i, (*outcomeHostVector)(i));
  }

  const DeviceMatrix& predictorsDeviceMatrix = lrConfig.getPredictors();
  HostMatrix* predictorsHostMatrix = deviceToHostStream1.transferMatrix(predictorsDeviceMatrix);
  stream->syncStream();
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

} /* namespace CUDA */
} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

