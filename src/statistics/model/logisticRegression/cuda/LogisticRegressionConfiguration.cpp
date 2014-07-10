#include "LogisticRegressionConfiguration.h"

namespace CuEira {
namespace Model {
namespace LogisticRegression {

LogisticRegressionConfiguration::LogisticRegressionConfiguration(const Configuration& configuration,
    const HostToDevice& hostToDevice, const DeviceVector& deviceOutcomes, const KernelWrapper& kernelWrapper) :
    usingCovariates(false), hostToDevice(&hostToDevice), kernelWrapper(&kernelWrapper), configuration(&configuration), numberOfRows(
        deviceOutcomes.getNumberOfRows()), numberOfPredictors(4), devicePredictors(
        new DeviceMatrix(numberOfRows, numberOfPredictors)), deviceOutcomes(&deviceOutcomes), maxIterations(
        configuration.getNumberOfMaxLRIterations()), convergenceThreshold(configuration.getLRConvergenceThreshold()), devicePredictorsMemoryPointer(
        devicePredictors->getMemoryPointer()), betaCoefficentsDevice(new DeviceVector(numberOfPredictors)), probabilitesDevice(
        new DeviceVector(numberOfRows)), scoresDevice(new DeviceVector(numberOfPredictors)), informationMatrixDevice(
        new DeviceMatrix(numberOfPredictors, numberOfPredictors)), workMatrixNxMDevice(
        new DeviceMatrix(numberOfRows, numberOfPredictors)), workVectorNx1Device(new DeviceVector(numberOfRows)), betaCoefficentsDefaultHost(
        new PinnedHostVector(numberOfPredictors)) {

  kernelWrapper.syncStream();
  transferIntercept();

  setDefaultBeta();

  kernelWrapper.syncStream();
}

LogisticRegressionConfiguration::LogisticRegressionConfiguration(const Configuration& configuration,
    const HostToDevice& hostToDevice, const DeviceVector& deviceOutcomes, const KernelWrapper& kernelWrapper,
    const HostMatrix& covariates) :
    usingCovariates(true), hostToDevice(&hostToDevice), kernelWrapper(&kernelWrapper), configuration(&configuration), numberOfRows(
        deviceOutcomes.getNumberOfRows()), numberOfPredictors(4 + covariates.getNumberOfColumns()), devicePredictors(
        new DeviceMatrix(numberOfRows, numberOfPredictors)), deviceOutcomes(&deviceOutcomes), maxIterations(
        configuration.getNumberOfMaxLRIterations()), convergenceThreshold(configuration.getLRConvergenceThreshold()), devicePredictorsMemoryPointer(
        devicePredictors->getMemoryPointer()), betaCoefficentsDevice(new DeviceVector(numberOfPredictors)), probabilitesDevice(
        new DeviceVector(numberOfRows)), scoresDevice(new DeviceVector(numberOfRows)), informationMatrixDevice(
        new DeviceMatrix(numberOfPredictors, numberOfPredictors)), workMatrixNxMDevice(
        new DeviceMatrix(numberOfRows, numberOfPredictors)), workVectorNx1Device(new DeviceVector(numberOfRows)), betaCoefficentsDefaultHost(
        new PinnedHostVector(numberOfPredictors)) {

  kernelWrapper.syncStream();
  transferIntercept();

  //Transfer covariates
  PRECISION* pos = devicePredictorsMemoryPointer + numberOfRows * 4; //Putting the covariates in the columns after the intercept, snp, environment and interaction columns.
  hostToDevice.transferMatrix(&covariates, pos);

  setDefaultBeta();

  kernelWrapper.syncStream();
}

LogisticRegressionConfiguration::LogisticRegressionConfiguration() :
    usingCovariates(false), numberOfPredictors(0), numberOfRows(0), maxIterations(0), hostToDevice(nullptr), kernelWrapper(
        nullptr), configuration(nullptr), deviceOutcomes(nullptr), devicePredictors(nullptr), convergenceThreshold(0), devicePredictorsMemoryPointer(
        nullptr), betaCoefficentsDevice(nullptr), probabilitesDevice(nullptr), scoresDevice(nullptr), informationMatrixDevice(
        nullptr), workMatrixNxMDevice(nullptr), workVectorNx1Device(nullptr), betaCoefficentsDefaultHost(nullptr) {

}

LogisticRegressionConfiguration::~LogisticRegressionConfiguration() {
  delete devicePredictors;
  delete betaCoefficentsDevice;
  delete probabilitesDevice;
  delete scoresDevice;
  delete informationMatrixDevice;
  delete workMatrixNxMDevice;
  delete workVectorNx1Device;
  delete betaCoefficentsDefaultHost;
}

void LogisticRegressionConfiguration::transferIntercept() {
  PinnedHostVector interceptHostVector(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    interceptHostVector(i) = 1;
  }

  hostToDevice->transferVector(&interceptHostVector, devicePredictorsMemoryPointer); //Putting the intercept as first column
}

void LogisticRegressionConfiguration::setDefaultBeta() {
  for(int i = 0; i < numberOfPredictors; ++i){
    (*betaCoefficentsDefaultHost)(i) = 0;
  }
}

void LogisticRegressionConfiguration::setEnvironmentFactor(const HostVector& environmentData) {
  PRECISION* pos = devicePredictorsMemoryPointer + numberOfRows * 2; //Putting the environment as the third column
  hostToDevice->transferVector(&environmentData, pos);
}

void LogisticRegressionConfiguration::setSNP(const HostVector& snpData) {
  PRECISION* pos = devicePredictorsMemoryPointer + numberOfRows * 1; //Putting the snp column as the second column
  hostToDevice->transferVector(&snpData, pos);
}

void LogisticRegressionConfiguration::setInteraction(const HostVector& interactionVector) {
  PRECISION* pos = devicePredictorsMemoryPointer + numberOfRows * 3; //Putting the interaction column as the fourth column
  hostToDevice->transferVector(&interactionVector, pos);
}

int LogisticRegressionConfiguration::getNumberOfRows() const {
  return numberOfRows;
}

int LogisticRegressionConfiguration::getNumberOfPredictors() const {
  return numberOfPredictors;
}

int LogisticRegressionConfiguration::getNumberOfMaxIterations() const {
  return maxIterations;
}

double LogisticRegressionConfiguration::getConvergenceThreshold() const {
  return convergenceThreshold;
}

const KernelWrapper& LogisticRegressionConfiguration::getKernelWrapper() const {
  return *kernelWrapper;
}

const DeviceMatrix& LogisticRegressionConfiguration::getPredictors() const {
  return *devicePredictors;
}

const DeviceVector& LogisticRegressionConfiguration::getOutcomes() const {
  return *deviceOutcomes;
}

DeviceVector& LogisticRegressionConfiguration::getProbabilites() {
  return *probabilitesDevice;
}

DeviceVector& LogisticRegressionConfiguration::getScores() {
  return *scoresDevice;
}

DeviceMatrix& LogisticRegressionConfiguration::getInformationMatrix() {
  return *informationMatrixDevice;
}

DeviceVector& LogisticRegressionConfiguration::getBetaCoefficents() {
  hostToDevice->transferVector(betaCoefficentsDefaultHost, betaCoefficentsDevice->getMemoryPointer());
  return *betaCoefficentsDevice;
}

DeviceMatrix& LogisticRegressionConfiguration::getWorkMatrixNxM() {
  return *workMatrixNxMDevice;
}

DeviceVector& LogisticRegressionConfiguration::getWorkVectorNx1() {
  return *workVectorNx1Device;
}

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */
