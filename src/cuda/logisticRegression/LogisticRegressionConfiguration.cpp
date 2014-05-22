#include "LogisticRegressionConfiguration.h"

namespace CuEira {
namespace CUDA {
namespace LogisticRegression {

LogisticRegressionConfiguration::LogisticRegressionConfiguration(const Configuration& configuration,
    const HostToDevice& hostToDevice, const HostVector& outcomes) :
    usingCovariates(false), hostToDevice(hostToDevice), configuration(configuration), numberOfRows(
        outcomes.getNumberOfRows()), numberOfPredictors(4), devicePredictors(
        new DeviceMatrix(numberOfRows, numberOfPredictors)), deviceOutcomes(hostToDevice.transferVector(outcomes)), maxIterations(
        configuration.getNumberOfMaxLRIterations()), convergenceThreshold(configuration.getLRConvergenceThreshold()) {
}

LogisticRegressionConfiguration::LogisticRegressionConfiguration(const Configuration& configuration,
    const HostToDevice& hostToDevice, const HostVector& outcomes, const HostMatrix& covariates) :
    usingCovariates(true), hostToDevice(hostToDevice), configuration(configuration), numberOfRows(
        outcomes.getNumberOfRows()), numberOfPredictors(4 + covariates.getNumberOfColumns()), devicePredictors(
        new DeviceMatrix(numberOfRows, numberOfPredictors)), deviceOutcomes(hostToDevice.transferVector(outcomes)), maxIterations(
        configuration.getNumberOfMaxLRIterations()), convergenceThreshold(configuration.getLRConvergenceThreshold()) {

}

LogisticRegressionConfiguration::~LogisticRegressionConfiguration() {

}

void LogisticRegressionConfiguration::setEnvironmentFactor(const HostVector& environmentData) {

}

void LogisticRegressionConfiguration::setSNP(const HostVector& snpData) {

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
