#include "MultipleLogisticRegression.h"

namespace CuEira {
namespace Model {
namespace LogisticRegression {
namespace Serial {

MultipleLogisticRegression::MultipleLogisticRegression(const LaGenMatDouble& predictorsRef,
    const LaVectorDouble& binaryOutcomes, LaVectorDouble* betaCoefficients, const int MAXIT,
    const double CONVERGENCETHRESHOLD) :
    LogisticRegression(betaCoefficients, MAXIT, CONVERGENCETHRESHOLD), predictors(addColumnForIntercept(predictorsRef)), binaryOutcomes(
        binaryOutcomes), numberOfInstances(predictorsRef.rows()), numberOfPredictors(betaCoefficients->size()) {
  if(predictors->rows() != binaryOutcomes.size()){
    throw std::invalid_argument("Predictors and binary outcomes not of same size");
  }
  if(predictors->cols() != numberOfPredictors){
    throw std::invalid_argument("Number of variables in predictor and beta is not matching");
  }

}

MultipleLogisticRegression::~MultipleLogisticRegression() {
  delete predictors;
}

const LaGenMatDouble* MultipleLogisticRegression::addColumnForIntercept(const LaGenMatDouble& predictors) {
  LaGenMatDouble* X = new LaGenMatDouble(numberOfInstances, numberOfPredictors);
  for(int j = 0; j < numberOfInstances; ++j){
    (*X)(j, 0) = 1;
  }
  for(int i = 1; i < numberOfPredictors; ++i){
    for(int j = 0; j < numberOfInstances; ++j){
      (*X)(j, i) = predictors(j, i - 1);
    }
  }
  return X;
}

void MultipleLogisticRegression::calculate() {
  /*
   Uses the Newton-Raphson algorithm to calculate maximum
   likliehood estimates of a simple logistic regression.

   covmat = inverse(info_mat)     --> covariance matrix
   stderr = sqrt(diag(covmat)) --> standard errors for beta
   deviance = -2l              --> scaled deviance statistic
   chi-squared value for -2l is the model chi-squared test.
   */

  LaVectorDouble scores = LaVectorDouble(numberOfPredictors);
  LaVectorDouble* betaCoefficientsOld = new LaVectorDouble(numberOfPredictors);
  LaVectorDouble probabilites = LaVectorDouble(numberOfInstances);
  LaVectorDouble workVectorNx1 = LaVectorDouble(numberOfInstances);
  double convergenceDifference;

  while(true){
    ++currentIteration;
    betaCoefficientsOld->copy(*betaCoefficients);

    calculateProbabilitiesScoreAndLogLikelihood(probabilites, scores, logLikelihood, betaCoefficients, workVectorNx1);

    calculateInformationMatrix(informationMatrix, probabilites, workVectorNx1);

    calculateNewBeta(betaCoefficients, betaCoefficientsOld, informationMatrix, scores); //NOTE Overwrites the informationMatrix

    bool done = checkBreakConditions(betaCoefficients, betaCoefficientsOld);
    if(done){
      calculateInformationMatrix(informationMatrix, probabilites, workVectorNx1);
      break;
    }
  }

  delete betaCoefficientsOld;
  delete predictors;
  predictors = nullptr;

  return;
}

void MultipleLogisticRegression::calculateProbabilitiesScoreAndLogLikelihood(LaVectorDouble& probabilites,
    LaVectorDouble& scores, double& logLikelihood, const LaVectorDouble* betaCoefficients,
    LaVectorDouble workVectorNx1) {
  logLikelihood = 0;

  Blas_Mat_Vec_Mult(*predictors, *betaCoefficients, workVectorNx1, 1, 0); //workAreaVectorNx1=predictors*beta

  for(int i = 0; i < numberOfInstances; ++i){
    probabilites(i) = exp(workVectorNx1(i)) / (1 + exp(workVectorNx1(i)));
    workVectorNx1(i) = binaryOutcomes(i) - probabilites(i);
    logLikelihood += binaryOutcomes(i) * log(probabilites(i)) + (1 - binaryOutcomes(i)) * log(1 - probabilites(i));
  }

  Blas_Mat_Trans_Vec_Mult(*predictors, workVectorNx1, scores, 1, 0); //s=X'*(y-p)
}

void MultipleLogisticRegression::calculateInformationMatrix(LaGenMatDouble& informationMatrix,
    const LaVectorDouble& probabilites, LaVectorDouble workVectorNx1) {
  LaGenMatDouble workMatrixNxM = LaGenMatDouble(numberOfInstances, numberOfPredictors);

  for(int i = 0; i < numberOfInstances; ++i){
    workVectorNx1(i) = probabilites(i) * (1 - probabilites(i));
  }

  for(int i = 0; i < numberOfPredictors; ++i){
    for(int j = 0; j < numberOfInstances; ++j){
      workMatrixNxM(j, i) = (*predictors)(j, i) * workVectorNx1(j);
    }
  }

  Blas_Mat_Trans_Mat_Mult(*predictors, workMatrixNxM, informationMatrix, 1, 0); //info_mat=X'*work_mat size(info_mat)=npreds x npreds
}

void MultipleLogisticRegression::calculateNewBeta(LaVectorDouble* betaCoefficients,
    const LaVectorDouble* betaCoefficientsOld, LaGenMatDouble& informationMatrix, const LaVectorDouble& scores) {
  LaVectorDouble sigmaDiagonal = LaVectorDouble(numberOfPredictors);
  LaGenMatDouble uSVD = LaGenMatDouble(numberOfPredictors, numberOfPredictors);
  LaGenMatDouble vSVDTranspose = LaGenMatDouble(numberOfPredictors, numberOfPredictors);
  LaVectorDouble workAreaVectorMx1 = LaVectorDouble(numberOfPredictors);

  //Calculate new beta, inverse(info_mat)*s+beta
  //Get the pseudo-inverse of the information matrix by using signular value decomposition
  //If the matrix is invertiable then the pseudo inverse is the same as the inverse
  LaSVD_IP(informationMatrix, sigmaDiagonal, uSVD, vSVDTranspose); //NOTE overwrites informationMatrix

  Blas_Mat_Trans_Vec_Mult(uSVD, scores, *betaCoefficients, 1, 0); //beta=u'*s //NOTE overwrites betaCoefficients, uses it for tmp storage

  //Multiply the pseudo-inverse of the MxN diagonal matrix witth u'*s stored in beta
  //sigma is the min(n,m) diagonal vector for the matrix
  //The pseudo-inverse is the sigma with all non-zero elements inverted
  //Diagonal matrix is NxM after pseudo-inverse instead of MxN,
  //this doesn't change anything since N=M=2 for single logistic regression
  for(int i = 0; i < numberOfPredictors; ++i){
    if(fabs(sigmaDiagonal(i)) > 1e-5){
      (*betaCoefficients)(i) = (*betaCoefficients)(i) / sigmaDiagonal(i);
    }else{
      (*betaCoefficients)(i) = 0;
    }
  }

  Blas_Mat_Trans_Vec_Mult(vSVDTranspose, *betaCoefficients, workAreaVectorMx1, 1, 0); //s=v_trans'*beta
  //Copy over to beta and add old beta
  for(int i = 0; i < numberOfPredictors; ++i){
    (*betaCoefficients)(i) = workAreaVectorMx1(i) + (*betaCoefficientsOld)(i);
  }
}

} /* namespace Serial */
} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */
