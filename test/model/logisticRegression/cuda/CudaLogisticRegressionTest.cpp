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
#include <CudaLogisticRegressionConfiguration.h>
#include <Configuration.h>
#include <ConfigurationMock.h>
#include <KernelWrapper.h>
#include <MKLWrapper.h>
#include <CudaLogisticRegression.h>
#include <LogisticRegressionResult.h>
#include <Device.h>
#include <Stream.h>
#include <StreamFactory.h>

using testing::Return;
using testing::_;
using testing::AtLeast;
using testing::Ge;
using testing::Le;

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
class CudaLogisticRegressionTest: public ::testing::Test {
protected:
  CudaLogisticRegressionTest();
  virtual ~CudaLogisticRegressionTest();
  virtual void SetUp();
  virtual void TearDown();

  ConfigurationMock configMock;
  const double convergenceThreshold;
  const int numberOfMaxLRIterations;
  const int numberOfCov;
  const int numberOfPredictorsNoCov;
  const int numberOfPredictorsWithCov;

  Device device;
  StreamFactory streamFactory;
  Stream* stream;
  HostToDevice hostToDeviceStream1;
  DeviceToHost deviceToHostStream1;
  KernelWrapper kernelWrapper;
  MKLWrapper blasWrapper;
};

CudaLogisticRegressionTest::CudaLogisticRegressionTest() :
    convergenceThreshold(1e-3), numberOfMaxLRIterations(500), numberOfCov(2), numberOfPredictorsNoCov(4), numberOfPredictorsWithCov(
        numberOfPredictorsNoCov + numberOfCov), device(0), streamFactory(), stream(
        streamFactory.constructStream(device)), hostToDeviceStream1(*stream), deviceToHostStream1(*stream), kernelWrapper(
        *stream), blasWrapper() {

  EXPECT_CALL(configMock, getLRConvergenceThreshold()).Times(AtLeast(0)).WillRepeatedly(Return(convergenceThreshold));
  EXPECT_CALL(configMock, getNumberOfMaxLRIterations()).Times(AtLeast(0)).WillRepeatedly(
      Return(numberOfMaxLRIterations));

}

CudaLogisticRegressionTest::~CudaLogisticRegressionTest() {
  delete stream;
}

void CudaLogisticRegressionTest::SetUp() {

}

void CudaLogisticRegressionTest::TearDown() {

}

TEST_F(CudaLogisticRegressionTest, calcuateProbabilites) {
  double e = 1e-5;
  double x, h, l;
  const int numberOfRows = 3;
  const int numberOfPredictors = 3;

  PinnedHostVector outcomesLRConfig(numberOfRows);

  DeviceVector* outcomeDeviceLRConfig = hostToDeviceStream1.transferVector(outcomesLRConfig);
  stream->syncStream();

  CudaLogisticRegressionConfiguration* lrConfig = new CudaLogisticRegressionConfiguration(configMock,
      hostToDeviceStream1, deviceToHostStream1, *outcomeDeviceLRConfig, kernelWrapper, blasWrapper);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error with LR config in calcuateProbabilites: ");

  CudaLogisticRegression logisticRegression(lrConfig);

  PinnedHostMatrix predHost(numberOfRows, numberOfPredictors);
  PinnedHostVector betaHost(numberOfPredictors);

  predHost(0, 0) = 1;
  predHost(1, 0) = 1;
  predHost(2, 0) = 1;

  predHost(0, 1) = 1;
  predHost(1, 1) = 2;
  predHost(2, 1) = 0.3;

  predHost(0, 2) = 0.1;
  predHost(1, 2) = 0.2;
  predHost(2, 2) = 0.5;

  betaHost(0) = 1;
  betaHost(1) = 2;
  betaHost(2) = 3;

  DeviceMatrix* predictorsDevice = hostToDeviceStream1.transferMatrix(predHost);
  DeviceVector* betaCoefficentsDevice = hostToDeviceStream1.transferVector(betaHost);
  DeviceVector* probabilitesDevice = new DeviceVector(numberOfRows);
  DeviceVector* workVectorNx1Device = new DeviceVector(numberOfRows);

  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error with setup in calcuateProbabilites: ");

  logisticRegression.calcuateProbabilites(*predictorsDevice, *betaCoefficentsDevice, *probabilitesDevice,
      *workVectorNx1Device);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error with calcuateProbabilites in calcuateProbabilites: ");

  HostVector* probHost = deviceToHostStream1.transferVector(*probabilitesDevice);

  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error with transfer back in calcuateProbabilites: ");
  ASSERT_EQ(numberOfRows, probHost->getNumberOfRows());

  x = 0.9644288;
  l = x - e;
  h = x + e;
  EXPECT_THAT((*probHost)(0), Ge(l));
  EXPECT_THAT((*probHost)(0), Le(h));

  x = 0.9963157;
  l = x - e;
  h = x + e;
  EXPECT_THAT((*probHost)(1), Ge(l));
  EXPECT_THAT((*probHost)(1), Le(h));

  x = 0.9568927;
  l = x - e;
  h = x + e;
  EXPECT_THAT((*probHost)(2), Ge(l));
  EXPECT_THAT((*probHost)(2), Le(h));

  delete outcomeDeviceLRConfig;
  delete predictorsDevice;
  delete betaCoefficentsDevice;
  delete probabilitesDevice;
  delete workVectorNx1Device;
  delete probHost;
}

TEST_F(CudaLogisticRegressionTest, calculateScores) {
  double e = 1e-5;
  double x, h, l;
  const int numberOfRows = 3;
  const int numberOfPredictors = 3;

  PinnedHostVector outcomesLRConfig(numberOfRows);

  DeviceVector* outcomeDeviceLRConfig = hostToDeviceStream1.transferVector(outcomesLRConfig);
  stream->syncStream();

  CudaLogisticRegressionConfiguration* lrConfig = new CudaLogisticRegressionConfiguration(configMock,
      hostToDeviceStream1, deviceToHostStream1, *outcomeDeviceLRConfig, kernelWrapper, blasWrapper);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error with LR config in calculateScores: ");

  CudaLogisticRegression logisticRegression(lrConfig);

  PinnedHostMatrix predHost(numberOfRows, numberOfPredictors);
  PinnedHostVector betaHost(numberOfPredictors);
  PinnedHostVector probHost(numberOfRows);
  PinnedHostVector outcomeHost(numberOfRows);

  predHost(0, 0) = 1;
  predHost(1, 0) = 1;
  predHost(2, 0) = 1;

  predHost(0, 1) = 1;
  predHost(1, 1) = 2;
  predHost(2, 1) = 0.3;

  predHost(0, 2) = 0.1;
  predHost(1, 2) = 0.2;
  predHost(2, 2) = 0.5;

  betaHost(0) = 1;
  betaHost(1) = 2;
  betaHost(2) = 3;

  probHost(0) = 0.9644288;
  probHost(1) = 0.9963157;
  probHost(2) = 0.9568927;

  outcomeHost(0) = 1;
  outcomeHost(1) = 1;
  outcomeHost(2) = 0;

  DeviceMatrix* predictorsDevice = hostToDeviceStream1.transferMatrix(predHost);
  DeviceVector* betaCoefficentsDevice = hostToDeviceStream1.transferVector(betaHost);
  DeviceVector* workVectorNx1Device = new DeviceVector(numberOfRows);
  DeviceVector* outcomesDevice = hostToDeviceStream1.transferVector(outcomeHost);
  DeviceVector* probabilitesDevice = hostToDeviceStream1.transferVector(probHost);
  DeviceVector* scoresDevice = new DeviceVector(numberOfPredictors);

  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error with setup in calculateScores: ");

  logisticRegression.calculateScores(*predictorsDevice, *outcomesDevice, *probabilitesDevice, *scoresDevice,
      *workVectorNx1Device);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error with calcuateProbabilites in calculateScores: ");

  HostVector* scoreHost = deviceToHostStream1.transferVector(*scoresDevice);

  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error with transfer back in calculateScores: ");
  ASSERT_EQ(numberOfPredictors, scoreHost->getNumberOfRows());

  x = -0.9176373;
  l = x - e;
  h = x + e;
  EXPECT_THAT((*scoreHost)(0), Ge(l));
  EXPECT_THAT((*scoreHost)(0), Le(h));

  x = -0.2441281;
  l = x - e;
  h = x + e;
  EXPECT_THAT((*scoreHost)(1), Ge(l));
  EXPECT_THAT((*scoreHost)(1), Le(h));

  x = -0.4741524;
  l = x - e;
  h = x + e;
  EXPECT_THAT((*scoreHost)(2), Ge(l));
  EXPECT_THAT((*scoreHost)(2), Le(h));

  delete outcomeDeviceLRConfig;
  delete predictorsDevice;
  delete betaCoefficentsDevice;
  delete probabilitesDevice;
  delete workVectorNx1Device;
  delete scoreHost;
  delete outcomesDevice;
  delete scoresDevice;
}

TEST_F(CudaLogisticRegressionTest, calculateInformationMatrix) {
  double e = 1e-5;
  double x, h, l;
  const int numberOfRows = 3;
  const int numberOfPredictors = 3;

  PinnedHostVector outcomesLRConfig(numberOfRows);

  DeviceVector* outcomeDeviceLRConfig = hostToDeviceStream1.transferVector(outcomesLRConfig);
  stream->syncStream();

  CudaLogisticRegressionConfiguration* lrConfig = new CudaLogisticRegressionConfiguration(configMock,
      hostToDeviceStream1, deviceToHostStream1, *outcomeDeviceLRConfig, kernelWrapper, blasWrapper);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error with LR config in calculateInformationMatrix: ");

  CudaLogisticRegression logisticRegression(lrConfig);

  PinnedHostMatrix predHost(numberOfRows, numberOfPredictors);
  PinnedHostVector probHost(numberOfRows);
  PinnedHostVector outcomeHost(numberOfRows);

  predHost(0, 0) = 1;
  predHost(1, 0) = 1;
  predHost(2, 0) = 1;

  predHost(0, 1) = 1;
  predHost(1, 1) = 2;
  predHost(2, 1) = 0.3;

  predHost(0, 2) = 0.1;
  predHost(1, 2) = 0.2;
  predHost(2, 2) = 0.5;

  probHost(0) = 0.9;
  probHost(1) = 0.3;
  probHost(2) = 0.5;

  DeviceMatrix* predictorsDevice = hostToDeviceStream1.transferMatrix(predHost);
  DeviceVector* workVectorNx1Device = new DeviceVector(numberOfRows);
  DeviceVector* probabilitesDevice = hostToDeviceStream1.transferVector(probHost);
  DeviceMatrix* informationMatrixDevice = new DeviceMatrix(numberOfPredictors, numberOfPredictors);
  DeviceMatrix* workMatrixNxMDevice = new DeviceMatrix(numberOfRows, numberOfPredictors);

  logisticRegression.calculateInformationMatrix(*predictorsDevice, *probabilitesDevice, *workVectorNx1Device,
      *informationMatrixDevice, *workMatrixNxMDevice);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error with calcuateProbabilites in calculateInformationMatrix: ");

  HostMatrix* infoMatHost = deviceToHostStream1.transferMatrix(*informationMatrixDevice);

  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error with transfer back in calculateInformationMatrix: ");
  ASSERT_EQ(numberOfPredictors, infoMatHost->getNumberOfRows());
  ASSERT_EQ(numberOfPredictors, infoMatHost->getNumberOfColumns());

  x = 0.55;
  l = x - e;
  h = x + e;
  EXPECT_THAT((*infoMatHost)(0, 0), Ge(l));
  EXPECT_THAT((*infoMatHost)(0, 0), Le(h));

  x = 0.585;
  l = x - e;
  h = x + e;
  EXPECT_THAT((*infoMatHost)(0, 1), Ge(l));
  EXPECT_THAT((*infoMatHost)(0, 1), Le(h));

  x = 0.176;
  l = x - e;
  h = x + e;
  EXPECT_THAT((*infoMatHost)(0, 2), Ge(l));
  EXPECT_THAT((*infoMatHost)(0, 2), Le(h));

  x = 0.9525;
  l = x - e;
  h = x + e;
  EXPECT_THAT((*infoMatHost)(1, 1), Ge(l));
  EXPECT_THAT((*infoMatHost)(1, 1), Le(h));

  x = 0.1305;
  l = x - e;
  h = x + e;
  EXPECT_THAT((*infoMatHost)(1, 2), Ge(l));
  EXPECT_THAT((*infoMatHost)(1, 2), Le(h));

  x = 0.0718;
  l = x - e;
  h = x + e;
  EXPECT_THAT((*infoMatHost)(2, 2), Ge(l));
  EXPECT_THAT((*infoMatHost)(2, 2), Le(h));

  EXPECT_EQ((*infoMatHost)(1, 0), (*infoMatHost)(0, 1));
  EXPECT_EQ((*infoMatHost)(2, 0), (*infoMatHost)(0, 2));
  EXPECT_EQ((*infoMatHost)(2, 1), (*infoMatHost)(1, 2));

  delete outcomeDeviceLRConfig;
  delete predictorsDevice;
  delete workVectorNx1Device;
  delete probabilitesDevice;
  delete informationMatrixDevice;
  delete workMatrixNxMDevice;
  delete infoMatHost;
}

TEST_F(CudaLogisticRegressionTest, calculateLogLikelihood) {
  double e = 1e-5;
  double x, h, l;
  const int numberOfRows = 3;
  const int numberOfPredictors = 3;

  PinnedHostVector outcomesLRConfig(numberOfRows);

  DeviceVector* outcomeDeviceLRConfig = hostToDeviceStream1.transferVector(outcomesLRConfig);
  stream->syncStream();

  CudaLogisticRegressionConfiguration* lrConfig = new CudaLogisticRegressionConfiguration(configMock,
      hostToDeviceStream1, deviceToHostStream1, *outcomeDeviceLRConfig, kernelWrapper, blasWrapper);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error with LR config in calculateLogLikelihood: ");

  CudaLogisticRegression logisticRegression(lrConfig);

  PinnedHostMatrix predHost(numberOfRows, numberOfPredictors);
  PinnedHostVector probHost(numberOfRows);
  PinnedHostVector outcomeHost(numberOfRows);
  PinnedHostVector oneVectorHost(numberOfRows);

  for(int i = 0; i < numberOfRows; ++i){
    oneVectorHost(i) = 1;
  }

  predHost(0, 0) = 1;
  predHost(1, 0) = 1;
  predHost(2, 0) = 1;

  predHost(0, 1) = 1;
  predHost(1, 1) = 2;
  predHost(2, 1) = 0.3;

  predHost(0, 2) = 0.1;
  predHost(1, 2) = 0.2;
  predHost(2, 2) = 0.5;

  probHost(0) = 0.9644288;
  probHost(1) = 0.9963157;
  probHost(2) = 0.9568927;

  outcomeHost(0) = 1;
  outcomeHost(1) = 1;
  outcomeHost(2) = 0;

  DeviceMatrix* predictorsDevice = hostToDeviceStream1.transferMatrix(predHost);
  DeviceVector* workVectorNx1Device = new DeviceVector(numberOfRows);
  DeviceVector* outcomesDevice = hostToDeviceStream1.transferVector(outcomeHost);
  DeviceVector* probabilitesDevice = hostToDeviceStream1.transferVector(probHost);
  DeviceVector* oneVectorDevice = hostToDeviceStream1.transferVector(oneVectorHost);
  PRECISION logLikelihood = 0;

  logisticRegression.calculateLogLikelihood(*outcomesDevice, *oneVectorDevice, *probabilitesDevice,
      *workVectorNx1Device, logLikelihood);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error with calcuateProbabilites in calculateLogLikelihood: ");

  x = -3.18397427;
  l = x - e;
  h = x + e;
  EXPECT_THAT(logLikelihood, Ge(l));
  EXPECT_THAT(logLikelihood, Le(h));

  delete outcomeDeviceLRConfig;
  delete predictorsDevice;
  delete probabilitesDevice;
  delete workVectorNx1Device;
  delete outcomesDevice;
  delete oneVectorDevice;
}

TEST_F(CudaLogisticRegressionTest, SmallTestNoCov) {
  double e = 1e-4;
  const int numberOfRows = 10;
  PinnedHostVector outcomes(numberOfRows);
  PinnedHostVector snpData(numberOfRows);
  PinnedHostVector environmentData(numberOfRows);
  PinnedHostVector interactionVector(numberOfRows);

  //SNP
  snpData(0) = 1.33;
  snpData(1) = -0.2;
  snpData(2) = 0.29;
  snpData(3) = 0.49;
  snpData(4) = -0.57;
  snpData(5) = -1;
  snpData(6) = 0.1;
  snpData(7) = -0.38;
  snpData(8) = 0.25;
  snpData(9) = -0.69;

  //Env
  environmentData(0) = 1.49;
  environmentData(1) = -0.99;
  environmentData(2) = 1.16;
  environmentData(3) = 0.49;
  environmentData(4) = 0.76;
  environmentData(5) = -0.3;
  environmentData(6) = -0.92;
  environmentData(7) = -0.6;
  environmentData(8) = -0.6;
  environmentData(9) = 0.32;

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
  correctBeta[0] = 0.4563;
  correctBeta[1] = 0.7382;
  correctBeta[2] = -0.5478;
  correctBeta[3] = 0.0867;

  DeviceVector* outcomeDeviceVector = hostToDeviceStream1.transferVector(outcomes);
  stream->syncStream();

  CudaLogisticRegressionConfiguration* lrConfig = new CudaLogisticRegressionConfiguration(configMock,
      hostToDeviceStream1, deviceToHostStream1, *outcomeDeviceVector, kernelWrapper, blasWrapper);
  lrConfig->setSNP(snpData);
  lrConfig->setEnvironmentFactor(environmentData);
  lrConfig->setInteraction(interactionVector);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error with LR config in test: ");

  CudaLogisticRegression logisticRegression(lrConfig);
  LogisticRegressionResult* lrResult = logisticRegression.calculate();
  const HostVector& beta = lrResult->getBeta();

  for(int i = 0; i < numberOfPredictorsNoCov; ++i){
    PRECISION l = correctBeta[i] - e;
    PRECISION h = correctBeta[i] + e;

    EXPECT_THAT(beta(i), Ge(l));
    EXPECT_THAT(beta(i), Le(h));
  }

  delete outcomeDeviceVector;
  delete lrResult;
}

TEST_F(CudaLogisticRegressionTest, SmallTestNoCovIntOnly) {
  double e = 1e-4;
  const int numberOfRows = 10;
  PinnedHostVector outcomes(numberOfRows);
  PinnedHostVector snpData(numberOfRows);
  PinnedHostVector environmentData(numberOfRows);
  PinnedHostVector interactionVector(numberOfRows);

  //SNP
  snpData(0) = 1;
  snpData(1) = 0;
  snpData(2) = 2;
  snpData(3) = 4;
  snpData(4) = 7;
  snpData(5) = -1;
  snpData(6) = 1;
  snpData(7) = 3;
  snpData(8) = -2;
  snpData(9) = -6;

  //Env
  environmentData(0) = 1;
  environmentData(1) = 0;
  environmentData(2) = 1;
  environmentData(3) = 4;
  environmentData(4) = 2;
  environmentData(5) = -3;
  environmentData(6) = -1;
  environmentData(7) = 6;
  environmentData(8) = 4;
  environmentData(9) = 3;

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
  correctBeta[0] = 0.2500;
  correctBeta[1] = -0.2557;
  correctBeta[2] = -0.0200;
  correctBeta[3] = 0.9337;

  DeviceVector* outcomeDeviceVector = hostToDeviceStream1.transferVector(outcomes);
  stream->syncStream();

  CudaLogisticRegressionConfiguration* lrConfig = new CudaLogisticRegressionConfiguration(configMock,
      hostToDeviceStream1, deviceToHostStream1, *outcomeDeviceVector, kernelWrapper, blasWrapper);
  lrConfig->setSNP(snpData);
  lrConfig->setEnvironmentFactor(environmentData);
  lrConfig->setInteraction(interactionVector);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error with LR config in test: ");

  CudaLogisticRegression logisticRegression(lrConfig);
  LogisticRegressionResult* lrResult = logisticRegression.calculate();
  const HostVector& beta = lrResult->getBeta();

  for(int i = 0; i < numberOfPredictorsNoCov; ++i){
    PRECISION l = correctBeta[i] - e;
    PRECISION h = correctBeta[i] + e;

    EXPECT_THAT(beta(i), Ge(l));
    EXPECT_THAT(beta(i), Le(h));
  }

  delete outcomeDeviceVector;
  delete lrResult;
}

TEST_F(CudaLogisticRegressionTest, SmallTestCov) {
  //TODO
}

} /* namespace CUDA */
} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

