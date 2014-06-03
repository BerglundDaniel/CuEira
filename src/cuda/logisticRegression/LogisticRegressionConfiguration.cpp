#include "LogisticRegressionConfiguration.h"

namespace CuEira {
namespace CUDA {
namespace LogisticRegression {

LogisticRegressionConfiguration::LogisticRegressionConfiguration(const Configuration& configuration,
    const HostToDevice& hostToDevice, const DeviceVector& deviceOutcomes, const KernelWrapper& kernelWrapper) :
    usingCovariates(false), hostToDevice(hostToDevice), kernelWrapper(kernelWrapper), configuration(configuration), numberOfRows(
        deviceOutcomes.getNumberOfRows()), numberOfPredictors(4), devicePredictors(
        new DeviceMatrix(numberOfRows, numberOfPredictors)), deviceOutcomes(deviceOutcomes), maxIterations(
        configuration.getNumberOfMaxLRIterations()), convergenceThreshold(configuration.getLRConvergenceThreshold()), devicePredictorsMemoryPointer(
        devicePredictors->getMemoryPointer()), betaCoefficentsDevice(new DeviceVector(numberOfPredictors)), betaCoefficentsOldDevice(
        new DeviceVector(numberOfPredictors)), probabilitesDevice(new DeviceVector(numberOfRows)), scoresDevice(
        new DeviceVector(numberOfPredictors)), informationMatrixDevice(
        new DeviceMatrix(numberOfPredictors, numberOfPredictors)), workMatrixNxMDevice(
        new DeviceMatrix(numberOfRows, numberOfPredictors)), workVectorNx1Device(new DeviceVector(numberOfRows)), workVectorMx1Device(
        new DeviceVector(numberOfPredictors)), uSVD(new DeviceMatrix(numberOfPredictors, numberOfPredictors)), vtSVD(
        new DeviceMatrix(numberOfPredictors, numberOfPredictors)), sigmaSVD(new DeviceVector(numberOfPredictors)), workMatrixMxMDevice(
        new DeviceMatrix(numberOfPredictors, numberOfPredictors)), inverseMatrixDevice(
        new DeviceMatrix(numberOfPredictors, numberOfPredictors)) {

  kernelWrapper.syncStream();
  transferIntercept();
}

LogisticRegressionConfiguration::LogisticRegressionConfiguration(const Configuration& configuration,
    const HostToDevice& hostToDevice, const DeviceVector& deviceOutcomes, const KernelWrapper& kernelWrapper,
    const HostMatrix& covariates) :
    usingCovariates(true), hostToDevice(hostToDevice), kernelWrapper(kernelWrapper), configuration(configuration), numberOfRows(
        deviceOutcomes.getNumberOfRows()), numberOfPredictors(4 + covariates.getNumberOfColumns()), devicePredictors(
        new DeviceMatrix(numberOfRows, numberOfPredictors)), deviceOutcomes(deviceOutcomes), maxIterations(
        configuration.getNumberOfMaxLRIterations()), convergenceThreshold(configuration.getLRConvergenceThreshold()), devicePredictorsMemoryPointer(
        devicePredictors->getMemoryPointer()), betaCoefficentsDevice(new DeviceVector(numberOfPredictors)), betaCoefficentsOldDevice(
        new DeviceVector(numberOfPredictors)), probabilitesDevice(new DeviceVector(numberOfRows)), scoresDevice(
        new DeviceVector(numberOfRows)), informationMatrixDevice(
        new DeviceMatrix(numberOfPredictors, numberOfPredictors)), workMatrixNxMDevice(
        new DeviceMatrix(numberOfRows, numberOfPredictors)), workVectorNx1Device(new DeviceVector(numberOfRows)), workVectorMx1Device(
        new DeviceVector(numberOfPredictors)), uSVD(new DeviceMatrix(numberOfPredictors, numberOfPredictors)), vtSVD(
        new DeviceMatrix(numberOfPredictors, numberOfPredictors)), sigmaSVD(new DeviceVector(numberOfPredictors)), workMatrixMxMDevice(
        new DeviceMatrix(numberOfPredictors, numberOfPredictors)), inverseMatrixDevice(
        new DeviceMatrix(numberOfPredictors, numberOfPredictors)) {

  kernelWrapper.syncStream();
  transferIntercept();

  //Transfer covariates
  PRECISION* pos = devicePredictorsMemoryPointer + numberOfRows * 4; //Putting the covariates in the columns after the intercept, snp, environment and interaction columns.
  hostToDevice.transferMatrix(&covariates, pos);
}

LogisticRegressionConfiguration::~LogisticRegressionConfiguration() {
  delete devicePredictors;
  delete betaCoefficentsDevice;
  delete betaCoefficentsOldDevice;
  delete probabilitesDevice;
  delete scoresDevice;
  delete informationMatrixDevice;
  delete workMatrixNxMDevice;
  delete workVectorNx1Device;
  delete workVectorMx1Device;
  delete workMatrixMxMDevice;
  delete uSVD;
  delete vtSVD;
  delete sigmaSVD;
  delete inverseMatrixDevice;
}

void LogisticRegressionConfiguration::transferIntercept() {
  PinnedHostVector interceptHostVector(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    interceptHostVector(i) = 1;
  }

  hostToDevice.transferVector(&interceptHostVector, devicePredictorsMemoryPointer); //Putting the intercept as first column
}

void LogisticRegressionConfiguration::setEnvironmentFactor(const HostVector& environmentData) {
  PRECISION* pos = devicePredictorsMemoryPointer + numberOfRows * 2; //Putting the environment as the third column
  hostToDevice.transferVector(&environmentData, pos);
}

void LogisticRegressionConfiguration::setSNP(const HostVector& snpData) {
  PRECISION* pos = devicePredictorsMemoryPointer + numberOfRows * 1; //Putting the snp column as the second column
  hostToDevice.transferVector(&snpData, pos);
}

void LogisticRegressionConfiguration::setInteraction(const HostVector& interactionVector) {
  PRECISION* pos = devicePredictorsMemoryPointer + numberOfRows * 3; //Putting the interaction column as the fourth column
  hostToDevice.transferVector(&interactionVector, pos);
}

void LogisticRegressionConfiguration::setBetaCoefficents(const HostVector& betaCoefficents) {
  delete betaCoefficentsDevice;
  betaCoefficentsDevice = hostToDevice.transferVector(&betaCoefficents);
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
  return kernelWrapper;
}

const DeviceMatrix& LogisticRegressionConfiguration::getPredictors() const {
  return *devicePredictors;
}

const DeviceVector& LogisticRegressionConfiguration::getOutcomes() const {
  return deviceOutcomes;
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
  return *betaCoefficentsDevice;
}

DeviceVector& LogisticRegressionConfiguration::getBetaCoefficentsOld() {
  return *betaCoefficentsOldDevice;
}

DeviceMatrix& LogisticRegressionConfiguration::getWorkMatrixNxM() {
  return *workMatrixNxMDevice;
}

DeviceMatrix& LogisticRegressionConfiguration::getWorkMatrixMxM() {
  return *workMatrixMxMDevice;
}

DeviceVector& LogisticRegressionConfiguration::getWorkVectorNx1() {
  return *workVectorNx1Device;
}

DeviceVector& LogisticRegressionConfiguration::getWorkVectorMx1() {
  return *workVectorMx1Device;
}

DeviceMatrix& LogisticRegressionConfiguration::getUSVD() {
  return *uSVD;
}

DeviceMatrix& LogisticRegressionConfiguration::getVtSVD() {
  return *vtSVD;
}

DeviceVector& LogisticRegressionConfiguration::getSigmaSVD() {
  return *sigmaSVD;
}

DeviceMatrix& LogisticRegressionConfiguration::getInverseMatrix() {
  return *inverseMatrixDevice;
}

} /* namespace LogisticRegression */
} /* namespace CUDA */
} /* namespace CuEira */
