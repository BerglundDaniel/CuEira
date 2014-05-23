#include "LogisticRegressionConfiguration.h"

namespace CuEira {
namespace CUDA {
namespace LogisticRegression {

LogisticRegressionConfiguration::LogisticRegressionConfiguration(const Configuration& configuration,
    const HostToDevice& hostToDevice, const HostVector& outcomes, const KernelWrapper& kernelWrapper) :
    usingCovariates(false), hostToDevice(hostToDevice), kernelWrapper(kernelWrapper), configuration(configuration), numberOfRows(
        outcomes.getNumberOfRows()), numberOfPredictors(4), devicePredictors(
        new DeviceMatrix(numberOfRows, numberOfPredictors)), deviceOutcomes(hostToDevice.transferVector(&outcomes)), maxIterations(
        configuration.getNumberOfMaxLRIterations()), convergenceThreshold(configuration.getLRConvergenceThreshold()), devicePredictorsMemoryPointer(
        devicePredictors->getMemoryPointer()), betaCoefficentsDevice(new DeviceVector(numberOfPredictors)), betaCoefficentsOldDevice(
        new DeviceVector(numberOfPredictors)), probabilitesDevice(new DeviceVector(numberOfRows)), scoresDevice(
        new DeviceVector(numberOfRows)), informationMatrixDevice(
        new DeviceMatrix(numberOfPredictors, numberOfPredictors)), workMatrixNxMDevice(
        new DeviceMatrix(numberOfRows, numberOfPredictors)), workVectorNx1Device(new DeviceVector(numberOfRows)), workVectorMx1Device(
        new DeviceVector(numberOfPredictors)), uSVD(new DeviceMatrix(numberOfPredictors, numberOfPredictors)), vtSVD(
        new DeviceMatrix(numberOfPredictors, numberOfPredictors)), sigmaSVD(new DeviceVector(numberOfPredictors)) {

  transferIntercept();
}

LogisticRegressionConfiguration::LogisticRegressionConfiguration(const Configuration& configuration,
    const HostToDevice& hostToDevice, const HostVector& outcomes, const KernelWrapper& kernelWrapper,
    const HostMatrix& covariates) :
    usingCovariates(true), hostToDevice(hostToDevice), kernelWrapper(kernelWrapper), configuration(configuration), numberOfRows(
        outcomes.getNumberOfRows()), numberOfPredictors(4 + covariates.getNumberOfColumns()), devicePredictors(
        new DeviceMatrix(numberOfRows, numberOfPredictors)), deviceOutcomes(hostToDevice.transferVector(&outcomes)), maxIterations(
        configuration.getNumberOfMaxLRIterations()), convergenceThreshold(configuration.getLRConvergenceThreshold()), devicePredictorsMemoryPointer(
        devicePredictors->getMemoryPointer()), betaCoefficentsDevice(new DeviceVector(numberOfPredictors)), betaCoefficentsOldDevice(
        new DeviceVector(numberOfPredictors)), probabilitesDevice(new DeviceVector(numberOfRows)), scoresDevice(
        new DeviceVector(numberOfRows)), informationMatrixDevice(
        new DeviceMatrix(numberOfPredictors, numberOfPredictors)), workMatrixNxMDevice(
        new DeviceMatrix(numberOfRows, numberOfPredictors)), workVectorNx1Device(new DeviceVector(numberOfRows)), workVectorMx1Device(
        new DeviceVector(numberOfPredictors)), uSVD(new DeviceMatrix(numberOfPredictors, numberOfPredictors)), vtSVD(
        new DeviceMatrix(numberOfPredictors, numberOfPredictors)), sigmaSVD(new DeviceVector(numberOfPredictors)) {

  transferIntercept();

  //Transfer covariates
  PRECISION* pos = devicePredictorsMemoryPointer + numberOfRows * 4; //Putting the covariates in the columns after the intercept, snp, environment and interaction columns.
  hostToDevice.transferMatrix(&covariates, pos);
}

LogisticRegressionConfiguration::~LogisticRegressionConfiguration() {
  delete devicePredictors;
  delete deviceOutcomes;
  delete betaCoefficentsDevice;
  delete betaCoefficentsOldDevice;
  delete probabilitesDevice;
  delete scoresDevice;
  delete informationMatrixDevice;
  delete workMatrixNxMDevice;
  delete workVectorNx1Device;
  delete workVectorMx1Device;
  delete uSVD;
  delete vtSVD;
  delete sigmaSVD;
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
  return *deviceOutcomes;
}

const DeviceVector& LogisticRegressionConfiguration::getProbabilites() const {
  return *probabilitesDevice;
}

const DeviceVector& LogisticRegressionConfiguration::getScores() const {
  return *scoresDevice;
}

const DeviceMatrix& LogisticRegressionConfiguration::getInformationMatrix() const {
  return *informationMatrixDevice;
}

const DeviceVector& LogisticRegressionConfiguration::getBetaCoefficents() const {
  return *betaCoefficentsDevice;
}

const DeviceVector& LogisticRegressionConfiguration::getBetaCoefficentsOld() const {
  return *betaCoefficentsOldDevice;
}

const DeviceMatrix& LogisticRegressionConfiguration::getWorkMatrixNxM() const {
  return *workMatrixNxMDevice;
}

const DeviceVector& LogisticRegressionConfiguration::getWorkVectorNx1() const {
  return *workVectorNx1Device;
}

const DeviceVector& LogisticRegressionConfiguration::getWorkVectorMx1() const {
  return *workVectorMx1Device;
}

const DeviceMatrix& LogisticRegressionConfiguration::getUSVD() const {
  return *uSVD;
}

const DeviceMatrix& LogisticRegressionConfiguration::getVtSVD() const {
  return *vtSVD;
}

const DeviceVector& LogisticRegressionConfiguration::getSigmaSVD() const {
  return *sigmaSVD;
}

} /* namespace LogisticRegression */
} /* namespace CUDA */
} /* namespace CuEira */
