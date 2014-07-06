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
  std::cerr << "LR beta" << std::endl;
  return *beta;
}

const Container::HostMatrix& LogisticRegressionResult::getInformationMatrix() const {
  return *informationMatrix;
}

const Container::HostMatrix& LogisticRegressionResult::getInverseInformationMatrix() const {
  std::cerr << "LR inverse" << std::endl;
  return *inverseInformationMatrixHost;
}

int LogisticRegressionResult::getNumberOfIterations() const {
  return numberOfIterations;
}

PRECISION LogisticRegressionResult::getLogLikelihood() const {
  return logLikelihood;
}

Recode LogisticRegressionResult::calculateRecode() const {
  Recode recode = ALL_RISK;

  double snpBeta = (*beta)(1);
  double envBeta = (*beta)(2);
  double interactionBeta = (*beta)(3);

  if(snpBeta < 0 && snpBeta < envBeta && snpBeta < interactionBeta){
    recode = SNP_PROTECT;
  }else if(envBeta < 0 && envBeta < snpBeta && envBeta < interactionBeta){
    recode = ENVIRONMENT_PROTECT;
  }else if(interactionBeta < 0 && interactionBeta < snpBeta && interactionBeta < envBeta){
    recode = INTERACTION_PROTECT;
  }

  return recode;
}

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */
