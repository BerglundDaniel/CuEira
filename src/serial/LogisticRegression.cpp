#include "LogisticRegression.h"

using namespace LogisticRegression;

namespace LogisticRegression {

LogisticRegression::LogisticRegression(LaVectorDouble* betaCoefficients, const int MAXIT,
    const double CONVERGENCETHRESHOLD) :
    betaCoefficients(betaCoefficients), MAXIT(MAXIT), CONVERGENCETHRESHOLD(CONVERGENCETHRESHOLD), currentIteration(0), logLikelihood(
        0), informationMatrix(LaGenMatDouble(betaCoefficients->size(), betaCoefficients->size())) {

}

LogisticRegression::~LogisticRegression() {

}

bool LogisticRegression::checkBreakConditions(const LaVectorDouble* betaCoefficients,
    const LaVectorDouble* betaCoefficientsOld) {
  double convergenceDifference = 0;
  for(int i = 0; i < betaCoefficients->size(); ++i){
    convergenceDifference += fabs((*betaCoefficientsOld)(i) - (*betaCoefficients)(i));
  }

  if(convergenceDifference <= CONVERGENCETHRESHOLD || currentIteration >= MAXIT){
    return true;
  }else{
    return false;
  }
}

const LaVectorDouble& LogisticRegression::getBeta() {
  return *betaCoefficients;
}

const LaGenMatDouble& LogisticRegression::getInformationMatrix() {
  return informationMatrix;
}

double LogisticRegression::getLogLikelihood() {
  return logLikelihood;
}

int LogisticRegression::getNumberOfIterations() {
  return currentIteration;
}

int LogisticRegression::getMaximumNumberOfIterations() {
  return MAXIT;
}

double LogisticRegression::getConvergenceThreshold() {
  return CONVERGENCETHRESHOLD;
}

}
