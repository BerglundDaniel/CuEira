#include "CpuLogisticRegression.h"

namespace CuEira {
namespace Model {
namespace LogisticRegression {
namespace CPU {

CpuLogisticRegression::CpuLogisticRegression(CpuLogisticRegressionConfiguration* cpuLogisticRegressionConfiguration) :
    LogisticRegression(cpuLogisticRegressionConfiguration), cpuLogisticRegressionConfiguration(
        cpuLogisticRegressionConfiguration), outcomes(&cpuLogisticRegressionConfiguration->getOutcomes()), predictors(
        &cpuLogisticRegressionConfiguration->getPredictors()), probabilites(
        &cpuLogisticRegressionConfiguration->getProbabilites()), workMatrixNxM(
        &cpuLogisticRegressionConfiguration->getWorkMatrixNxM()), workVectorNx1(
        &cpuLogisticRegressionConfiguration->getWorkVectorNx1()), defaultBetaCoefficents(
        &cpuLogisticRegressionConfiguration->getDefaultBetaCoefficents()){

}

CpuLogisticRegression::CpuLogisticRegression() :
    LogisticRegression(), cpuLogisticRegressionConfiguration(nullptr), outcomes(nullptr), predictors(nullptr), probabilites(
        nullptr), workMatrixNxM(nullptr), workVectorNx1(nullptr), defaultBetaCoefficents(nullptr){

}

CpuLogisticRegression::~CpuLogisticRegression(){

}

LogisticRegressionResult* CpuLogisticRegression::calculate(){
  PRECISION diffSum = 0;
  logLikelihood = 0;

  Container::HostVector* betaCoefficents = new Container::RegularHostVector(numberOfPredictors);
  Blas::copyVector(*defaultBetaCoefficents, *betaCoefficents);

  Container::HostMatrix* informationMatrix = new Container::RegularHostMatrix(numberOfPredictors, numberOfPredictors);
  Container::HostMatrix* inverseInformationMatrix = new Container::RegularHostMatrix(numberOfPredictors,
      numberOfPredictors);

  int iterationNumber = 1;
  for(iterationNumber = 1; iterationNumber < maxIterations; ++iterationNumber){
    //Copy beta to old beta
    Blas::copyVector(*betaCoefficents, *betaCoefficentsOldHost);

    calcuateProbabilites(*predictors, *betaCoefficents, *probabilites, *workVectorNx1);

    calculateScores(*predictors, *outcomes, *probabilites, *scoresHost, *workVectorNx1);

    calculateInformationMatrix(*predictors, *probabilites, *workVectorNx1, *informationMatrix, *workMatrixNxM);

    invertInformationMatrix(*informationMatrix, *inverseInformationMatrix, *uSVD, *sigma, *vtSVD, *workMatrixMxMHost);

    calculateNewBeta(*inverseInformationMatrix, *scoresHost, *betaCoefficents);

    calculateDifference(*betaCoefficents, *betaCoefficentsOldHost, diffSum);

    if(diffSum < convergenceThreshold){
      calculateLogLikelihood(*outcomes, *probabilites, logLikelihood);
      break;
    }
  } /* for iterationNumber */

  return new LogisticRegressionResult(betaCoefficents, informationMatrix, inverseInformationMatrix, iterationNumber,
      logLikelihood);
}

void CpuLogisticRegression::calcuateProbabilites(const HostMatrix& predictors, const HostVector& betaCoefficents,
    HostVector& probabilites, HostVector& workVectorNx1){
  Blas::matrixVectorMultiply(predictors, betaCoefficents, workVectorNx1, 1, 0);

  for(int i = 0; i < numberOfRows; ++i){
    probabilites(i) = exp(workVectorNx1(i)) / (1 + exp(workVectorNx1(i)));
  }
}

void CpuLogisticRegression::calculateScores(const HostMatrix& predictors, const HostVector& outcomes,
    const HostVector& probabilites, HostVector& scores, HostVector& workVectorNx1){
  for(int i = 0; i < numberOfRows; ++i){
    workVectorNx1(i) = outcomes(i) - probabilites(i);
  }

  Blas::matrixTransVectorMultiply(predictors, workVectorNx1, scores, 1, 0);
}

void CpuLogisticRegression::calculateInformationMatrix(const HostMatrix& predictors, const HostVector& probabilites,
    HostVector& workVectorNx1, HostMatrix& informationMatrix, HostMatrix& workMatrixNxM){
  for(int i = 0; i < numberOfRows; ++i){
    workVectorNx1(i) = probabilites(i) * (1 - probabilites(i));
  }

  for(int k = 0; k < numberOfPredictors; ++k){
    const HostVector* columnVector = predictors(k);
    HostVector* columnResultVector = workMatrixNxM(k);

    for(int i = 0; i < numberOfRows; ++i){
      (*columnResultVector)(i) = workVectorNx1(i) * (*columnVector)(i);
    }

    delete columnVector;
    delete columnResultVector;
  }

  Blas::matrixTransMatrixMultiply(predictors, workMatrixNxM, informationMatrix, 1, 0);
}

void CpuLogisticRegression::calculateLogLikelihood(const HostVector& outcomes, const HostVector& probabilites,
    PRECISION& logLikelihood){
  logLikelihood = 0;
  for(int i = 0; i < numberOfRows; ++i){
    logLikelihood += outcomes(i) * log(probabilites(i)) + (1 - outcomes(i)) * log(1 - probabilites(i));
  }
}

} /* namespace CPU */
} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */
