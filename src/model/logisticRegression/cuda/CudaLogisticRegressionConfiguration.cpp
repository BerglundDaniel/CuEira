#include "CudaLogisticRegressionConfiguration.h"

namespace CuEira {
namespace Model {
namespace LogisticRegression {
namespace CUDA {

#ifdef PROFILE
boost::chrono::duration<long long, boost::nano> CudaLogisticRegressionConfiguration::timeSpentTransferToDevice;
#endif

CudaLogisticRegressionConfiguration::CudaLogisticRegressionConfiguration(const Configuration& configuration,
    const HostToDevice& hostToDevice, const DeviceToHost& deviceToHost, const DeviceVector& deviceOutcomes,
    const KernelWrapper& kernelWrapper, const MKLWrapper& blasWrapper) :
    LogisticRegressionConfiguration(configuration, blasWrapper, false, deviceOutcomes.getNumberOfRows(), 4), hostToDevice(
        &hostToDevice), deviceToHost(&deviceToHost), kernelWrapper(&kernelWrapper), devicePredictors(
        new DeviceMatrix(numberOfRows, numberOfPredictors)), deviceOutcomes(&deviceOutcomes), devicePredictorsMemoryPointer(
        devicePredictors->getMemoryPointer()), betaCoefficentsDevice(new DeviceVector(numberOfPredictors)), probabilitesDevice(
        new DeviceVector(numberOfRows)), scoresDevice(new DeviceVector(numberOfPredictors)), informationMatrixDevice(
        new DeviceMatrix(numberOfPredictors, numberOfPredictors)), workMatrixNxMDevice(
        new DeviceMatrix(numberOfRows, numberOfPredictors)), workVectorNx1Device(new DeviceVector(numberOfRows)), betaCoefficentsDefaultHost(
        new PinnedHostVector(numberOfPredictors)), scoresHost(new PinnedHostVector(numberOfPredictors)) {

#ifdef PROFILE
  boost::chrono::system_clock::time_point beforeTransfer = boost::chrono::system_clock::now();
#endif
  transferIntercept();

  setDefaultBeta(*betaCoefficentsDefaultHost);

  kernelWrapper.syncStream();

#ifdef PROFILE
  boost::chrono::system_clock::time_point afterTransfer = boost::chrono::system_clock::now();
  timeSpentTransferToDevice+=afterTransfer - beforeTransfer;
#endif
}

CudaLogisticRegressionConfiguration::CudaLogisticRegressionConfiguration(const Configuration& configuration,
    const HostToDevice& hostToDevice, const DeviceToHost& deviceToHost, const DeviceVector& deviceOutcomes,
    const KernelWrapper& kernelWrapper, const MKLWrapper& blasWrapper, const PinnedHostMatrix& covariates) :
    LogisticRegressionConfiguration(configuration, blasWrapper, true, deviceOutcomes.getNumberOfRows(),
        4 + covariates.getNumberOfColumns()), hostToDevice(&hostToDevice), deviceToHost(&deviceToHost), kernelWrapper(
        &kernelWrapper), devicePredictors(new DeviceMatrix(numberOfRows, numberOfPredictors)), deviceOutcomes(
        &deviceOutcomes), devicePredictorsMemoryPointer(devicePredictors->getMemoryPointer()), betaCoefficentsDevice(
        new DeviceVector(numberOfPredictors)), probabilitesDevice(new DeviceVector(numberOfRows)), scoresDevice(
        new DeviceVector(numberOfPredictors)), informationMatrixDevice(
        new DeviceMatrix(numberOfPredictors, numberOfPredictors)), workMatrixNxMDevice(
        new DeviceMatrix(numberOfRows, numberOfPredictors)), workVectorNx1Device(new DeviceVector(numberOfRows)), betaCoefficentsDefaultHost(
        new PinnedHostVector(numberOfPredictors)), scoresHost(new PinnedHostVector(numberOfPredictors)) {

#ifdef PROFILE
  boost::chrono::system_clock::time_point beforeTransfer = boost::chrono::system_clock::now();
#endif
  transferIntercept();

  //Transfer covariates
  PRECISION* pos = devicePredictorsMemoryPointer + numberOfRows * 4; //Putting the covariates in the columns after the intercept, snp, environment and interaction columns.
  hostToDevice.transferMatrix(covariates, pos);

  setDefaultBeta(*betaCoefficentsDefaultHost);

  kernelWrapper.syncStream();

#ifdef PROFILE
  boost::chrono::system_clock::time_point afterTransfer = boost::chrono::system_clock::now();
  timeSpentTransferToDevice+=afterTransfer - beforeTransfer;
#endif
}

CudaLogisticRegressionConfiguration::CudaLogisticRegressionConfiguration(const Configuration& configuration,
    const MKLWrapper& blasWrapper) :
    LogisticRegressionConfiguration(configuration, blasWrapper, false, 0, 0), hostToDevice(nullptr), deviceToHost(
        nullptr), kernelWrapper(nullptr), deviceOutcomes(nullptr), devicePredictors(nullptr), devicePredictorsMemoryPointer(
        nullptr), betaCoefficentsDevice(nullptr), probabilitesDevice(nullptr), scoresDevice(nullptr), informationMatrixDevice(
        nullptr), workMatrixNxMDevice(nullptr), workVectorNx1Device(nullptr), betaCoefficentsDefaultHost(nullptr), scoresHost(
        nullptr) {

}

CudaLogisticRegressionConfiguration::~CudaLogisticRegressionConfiguration() {
  delete devicePredictors;
  delete betaCoefficentsDevice;
  delete probabilitesDevice;
  delete scoresDevice;
  delete informationMatrixDevice;
  delete workMatrixNxMDevice;
  delete workVectorNx1Device;
  delete betaCoefficentsDefaultHost;
  delete scoresHost;
}

void CudaLogisticRegressionConfiguration::transferIntercept() {
  PinnedHostVector interceptHostVector(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    interceptHostVector(i) = 1;
  }

  hostToDevice->transferVector(interceptHostVector, devicePredictorsMemoryPointer); //Putting the intercept as first column
}

void CudaLogisticRegressionConfiguration::setEnvironmentFactor(const HostVector& environmentData) {
  PRECISION* pos = devicePredictorsMemoryPointer + numberOfRows * 2; //Putting the environment as the third column

#ifdef PROFILE
      boost::chrono::system_clock::time_point beforeTransfer = boost::chrono::system_clock::now();
#endif

  hostToDevice->transferVector((const PinnedHostVector&) environmentData, pos);
  kernelWrapper->syncStream();

#ifdef PROFILE
  boost::chrono::system_clock::time_point afterTransfer = boost::chrono::system_clock::now();
  timeSpentTransferToDevice+=afterTransfer - beforeTransfer;
#endif
}

void CudaLogisticRegressionConfiguration::setSNP(const HostVector& snpData) {
  PRECISION* pos = devicePredictorsMemoryPointer + numberOfRows * 1; //Putting the snp column as the second column

#ifdef PROFILE
      boost::chrono::system_clock::time_point beforeTransfer = boost::chrono::system_clock::now();
#endif

  hostToDevice->transferVector((const PinnedHostVector&) snpData, pos);
  kernelWrapper->syncStream();

#ifdef PROFILE
  boost::chrono::system_clock::time_point afterTransfer = boost::chrono::system_clock::now();
  timeSpentTransferToDevice+=afterTransfer - beforeTransfer;
#endif
}

void CudaLogisticRegressionConfiguration::setInteraction(const HostVector& interactionVector) {
  PRECISION* pos = devicePredictorsMemoryPointer + numberOfRows * 3; //Putting the interaction column as the fourth column

#ifdef PROFILE
      boost::chrono::system_clock::time_point beforeTransfer = boost::chrono::system_clock::now();
#endif

  hostToDevice->transferVector((const PinnedHostVector&) interactionVector, pos);
  kernelWrapper->syncStream();

#ifdef PROFILE
  boost::chrono::system_clock::time_point afterTransfer = boost::chrono::system_clock::now();
  timeSpentTransferToDevice+=afterTransfer - beforeTransfer;
#endif
}

const KernelWrapper& CudaLogisticRegressionConfiguration::getKernelWrapper() const {
  return *kernelWrapper;
}

const HostToDevice& CudaLogisticRegressionConfiguration::getHostToDevice() const {
  return *hostToDevice;
}

const DeviceToHost& CudaLogisticRegressionConfiguration::getDeviceToHost() const {
  return *deviceToHost;
}

const DeviceMatrix& CudaLogisticRegressionConfiguration::getPredictors() const {
  return *devicePredictors;
}

const DeviceVector& CudaLogisticRegressionConfiguration::getOutcomes() const {
  return *deviceOutcomes;
}

DeviceVector& CudaLogisticRegressionConfiguration::getProbabilites() {
  return *probabilitesDevice;
}

DeviceVector& CudaLogisticRegressionConfiguration::getScores() {
  return *scoresDevice;
}

HostVector& CudaLogisticRegressionConfiguration::getScoresHost() {
  return *scoresHost;
}

DeviceMatrix& CudaLogisticRegressionConfiguration::getInformationMatrix() {
  return *informationMatrixDevice;
}

DeviceVector& CudaLogisticRegressionConfiguration::getBetaCoefficents() {
  return *betaCoefficentsDevice;
}

DeviceMatrix& CudaLogisticRegressionConfiguration::getWorkMatrixNxM() {
  return *workMatrixNxMDevice;
}

DeviceVector& CudaLogisticRegressionConfiguration::getWorkVectorNx1() {
  return *workVectorNx1Device;
}

const PinnedHostVector& CudaLogisticRegressionConfiguration::getDefaultBetaCoefficents() const {
  return *betaCoefficentsDefaultHost;
}

} /* namespace CUDA */
} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */
