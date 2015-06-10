#include "LogisticRegressionConfiguration.h"

namespace CuEira {
namespace Model {
namespace LogisticRegression {

LogisticRegressionConfiguration::LogisticRegressionConfiguration(const Configuration& configuration,
    bool usingCovariates, const int numberOfRows, const int numberOfPredictors) :
    ModelConfiguration(configuration), maxIterations(configuration.getNumberOfMaxLRIterations()), convergenceThreshold(
        configuration.getLRConvergenceThreshold()), numberOfPredictors(numberOfPredictors), numberOfRows(numberOfRows), usingCovariates(
        usingCovariates){

}

LogisticRegressionConfiguration::~LogisticRegressionConfiguration(){

}

void LogisticRegressionConfiguration::setDefaultBeta(HostVector& beta){
  for(int i = 0; i < numberOfPredictors; ++i){
    beta(i) = 0;
  }
}

int LogisticRegressionConfiguration::getNumberOfRows() const{
  return numberOfRows;
}

int LogisticRegressionConfiguration::getNumberOfPredictors() const{
  return numberOfPredictors;
}

int LogisticRegressionConfiguration::getNumberOfMaxIterations() const{
  return maxIterations;
}

double LogisticRegressionConfiguration::getConvergenceThreshold() const{
  return convergenceThreshold;
}

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */
