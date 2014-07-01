#include "LogisticRegressionResult.h"

namespace CuEira {
namespace Model {
namespace LogisticRegression {

LogisticRegressionResult::LogisticRegressionResult(Container::HostVector* beta,
    Container::HostMatrix* informationMatrix, Container::HostMatrix* inverseInformationMatrixHost,
    int numberOfIterations, PRECISION logLikelihood) :
    beta(beta), informationMatrix(informationMatrix), inverseInformationMatrixHost(inverseInformationMatrixHost), numberOfIterations(
        numberOfIterations), logLikelihood(logLikelihood) {

}

LogisticRegressionResult::~LogisticRegressionResult() {
  delete beta;
  delete informationMatrix;
  delete inverseInformationMatrixHost;
}

const Container::HostVector& LogisticRegressionResult::getBeta() const {
  return *beta;
}

const Container::HostMatrix& LogisticRegressionResult::getInformationMatrix() const {
  return *informationMatrix;
}

const Container::HostMatrix& LogisticRegressionResult::getInverseInformationMatrixHost() const {
  return *inverseInformationMatrixHost;
}

int LogisticRegressionResult::getNumberOfIterations() const {
  return numberOfIterations;
}

PRECISION LogisticRegressionResult::getLogLikelihood() const {
  return logLikelihood;
}

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */
