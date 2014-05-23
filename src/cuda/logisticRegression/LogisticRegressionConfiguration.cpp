#include "LogisticRegressionConfiguration.h"

namespace CuEira {
namespace CUDA {
namespace LogisticRegression {

LogisticRegressionConfiguration::LogisticRegressionConfiguration(const Configuration& configuration,
    const HostToDevice& hostToDevice, const HostVector& outcomes) :
    usingCovariates(false), hostToDevice(hostToDevice), configuration(configuration), numberOfRows(
        outcomes.getNumberOfRows()), numberOfPredictors(4), devicePredictors(
        new DeviceMatrix(numberOfRows, numberOfPredictors)), deviceOutcomes(hostToDevice.transferVector(&outcomes)), maxIterations(
        configuration.getNumberOfMaxLRIterations()), convergenceThreshold(configuration.getLRConvergenceThreshold()), devicePredictorsMemoryPointer(
        devicePredictors->getMemoryPointer()) {

  transferIntercept();
}

LogisticRegressionConfiguration::LogisticRegressionConfiguration(const Configuration& configuration,
    const HostToDevice& hostToDevice, const HostVector& outcomes, const HostMatrix& covariates) :
    usingCovariates(true), hostToDevice(hostToDevice), configuration(configuration), numberOfRows(
        outcomes.getNumberOfRows()), numberOfPredictors(4 + covariates.getNumberOfColumns()), devicePredictors(
        new DeviceMatrix(numberOfRows, numberOfPredictors)), deviceOutcomes(hostToDevice.transferVector(&outcomes)), maxIterations(
        configuration.getNumberOfMaxLRIterations()), convergenceThreshold(configuration.getLRConvergenceThreshold()), devicePredictorsMemoryPointer(
        devicePredictors->getMemoryPointer()) {

  transferIntercept();

  //Transfer covariates
  PRECISION* pos = devicePredictorsMemoryPointer + numberOfRows * 4; //Putting the covariates in the columns after the intercept, snp, environment and interaction columns.
  hostToDevice.transferMatrix(&covariates, pos);
}

LogisticRegressionConfiguration::~LogisticRegressionConfiguration() {

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

const DeviceMatrix& LogisticRegressionConfiguration::getPredictors() const {
  return *devicePredictors;
}

const DeviceVector& LogisticRegressionConfiguration::getOutcomes() const {
  return *deviceOutcomes;
}

} /* namespace LogisticRegression */
} /* namespace CUDA */
} /* namespace CuEira */
