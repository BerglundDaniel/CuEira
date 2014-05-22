#include "LogisticRegression.h"

namespace CuEira {
namespace CUDA {
namespace LogisticRegression {

LogisticRegression::LogisticRegression(const KernelWrapper& kernelWrapper,
    const LogisticRegressionConfiguration& lrConfiguration) :
    kernelWrapper(kernelWrapper), lrConfiguration(lrConfiguration), maxIterations(
        lrConfiguration.getNumberOfMaxIterations()), convergenceThreshold(lrConfiguration.getConvergenceThreshold()), numberOfRows(
        lrConfiguration.getNumberOfRows()), numberOfPredictors(lrConfiguration.getNumberOfPredictors()) {

  //Allocate all the stuff
  //prob
  //score
  //work areas? old beta

  for(iterationNumber = 0; iterationNumber < maxIterations; ++iterationNumber){

    PRECISION diffSum;
    if(diffSum < convergenceThreshold){

      //Calculate loglikelihood

      break;
    }
  }
}

LogisticRegression::~LogisticRegression() {

}

const DeviceVector* LogisticRegression::getBeta() const {

}

const DeviceMatrix* LogisticRegression::getInformationMatrix() const {

}

int LogisticRegression::getNumberOfIterations() const {
  return iterationNumber;
}

PRECISION* LogisticRegression::getLogLikelihood() const {

}

} /* namespace LogisticRegression */
} /* namespace CUDA */
} /* namespace CuEira */
