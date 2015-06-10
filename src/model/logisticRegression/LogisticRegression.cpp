#include "LogisticRegression.h"

namespace CuEira {
namespace Model {
namespace LogisticRegression {

LogisticRegression::LogisticRegression(LogisticRegressionConfiguration* logisticRegressionConfiguration) :
    Model(logisticRegressionConfiguration), logisticRegressionConfiguration(logisticRegressionConfiguration), numberOfRows(
        logisticRegressionConfiguration->getNumberOfRows()), numberOfPredictors(
        logisticRegressionConfiguration->getNumberOfPredictors()), maxIterations(
        logisticRegressionConfiguration->getNumberOfMaxIterations()), convergenceThreshold(
        logisticRegressionConfiguration->getConvergenceThreshold()), logLikelihood(0), sigma(
        new RegularHostVector(numberOfPredictors)), uSVD(new RegularHostMatrix(numberOfPredictors, numberOfPredictors)), vtSVD(
        new RegularHostMatrix(numberOfPredictors, numberOfPredictors)), workMatrixMxMHost(
        new RegularHostMatrix(numberOfPredictors, numberOfPredictors)), betaCoefficentsOldHost(
        new Container::RegularHostVector(numberOfPredictors)), scoresHost(
        &logisticRegressionConfiguration->getScoresHost()){

}

LogisticRegression::LogisticRegression() :
    Model(), logisticRegressionConfiguration(nullptr), numberOfRows(0), numberOfPredictors(0), maxIterations(0), convergenceThreshold(
        0), logLikelihood(0), sigma(nullptr), uSVD(nullptr), vtSVD(nullptr), workMatrixMxMHost(nullptr), betaCoefficentsOldHost(
        nullptr), scoresHost(nullptr){

}

LogisticRegression::~LogisticRegression(){
  delete betaCoefficentsOldHost;
  delete sigma;
  delete uSVD;
  delete vtSVD;
  delete workMatrixMxMHost;
}

void LogisticRegression::invertInformationMatrix(HostMatrix& informationMatrixHost,
    HostMatrix& inverseInformationMatrixHost, HostMatrix& uSVD, HostVector& sigma, HostMatrix& vtSVD,
    HostMatrix& workMatrixMxMHost){
  int size = informationMatrixHost.getNumberOfRows();

  Blas::svd(informationMatrixHost, uSVD, sigma, vtSVD);

  //diag(sigma)*uSVD'
  for(int i = 0; i < size; ++i){
    PRECISION inverseSigma;
    if(fabs(sigma(i)) > 1e-5){
      inverseSigma = 1 / sigma(i);
    }else{
      inverseSigma = 0;
    }
    //cblas_ FIXME use scal
    for(int k = 0; k < size; ++k){
      workMatrixMxMHost(i, k) = inverseSigma * uSVD(k, i);
    }
  }

  Blas::matrixTransMatrixMultiply(vtSVD, workMatrixMxMHost, inverseInformationMatrixHost, 1, 0);
}

void LogisticRegression::calculateNewBeta(HostMatrix& inverseInformationMatrixHost, HostVector& scoresHost,
    HostVector& betaCoefficentsHost){
  //beta=inv*scores+beta
  Blas::matrixVectorMultiply(inverseInformationMatrixHost, scoresHost, betaCoefficentsHost, 1, 1);
}

void LogisticRegression::calculateDifference(const HostVector& betaCoefficentsHost, HostVector& betaCoefficentsOldHost,
    PRECISION& diffSumHost){
  Blas::differenceElememtWise(betaCoefficentsHost, betaCoefficentsOldHost);
  Blas::absoluteSum(betaCoefficentsOldHost, diffSumHost);
}

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */
