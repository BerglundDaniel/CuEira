#include "SimpleLogisticRegression.h"

using namespace LogisticRegression;

namespace LogisticRegression {

SimpleLogisticRegression::SimpleLogisticRegression(const LaVectorDouble& predictor,
    const LaVectorDouble& binaryOutcomes, LaVectorDouble* betaCoefficients, const int MAXIT,
    const double CONVERGENCETHRESHOLD) :
    LogisticRegression(betaCoefficients, MAXIT, CONVERGENCETHRESHOLD), predictor(predictor), binaryOutcomes(
        binaryOutcomes), numberOfInstances(predictor.size()), numberOfPredictors(betaCoefficients->size()) {
  if(predictor.size() != binaryOutcomes.size()){
    throw std::invalid_argument("Predictors and binary outcomes not of same size");
  }
  if(numberOfPredictors != 2){
    throw std::invalid_argument("Number of predictors must be of size 2");
  }
}

SimpleLogisticRegression::~SimpleLogisticRegression() {

}

void SimpleLogisticRegression::calculate() {
  /*
   Uses the Newton-Raphson algorithm to calculate maximum
   likliehood estimates of a simple logistic regression.

   covmat = inverse(info_mat)     --> covariance matrix
   stderr = sqrt(diag(covmat)) --> standard errors for beta
   deviance = -2l              --> scaled deviance statistic
   chi-squared value for -2l is the model chi-squared test.
   */

  LaVectorDouble probabilites = LaVectorDouble(numberOfInstances);
  LaVectorDouble scores = LaVectorDouble(numberOfPredictors);
  LaVectorDouble* betaCoefficientsOld = new LaVectorDouble(numberOfPredictors);

  while(true){
    ++currentIteration;
    betaCoefficientsOld->copy(*betaCoefficients);

    calculateProbabilities(probabilites, betaCoefficients);

    calculateScoreAndLogLikelihood(scores, logLikelihood, probabilites);

    calculateInformationMatrix(informationMatrix, probabilites);

    calculateNewBeta(betaCoefficients, betaCoefficientsOld, informationMatrix, scores); //NOTE Overwrites the informationMatrix

    bool done = checkBreakConditions(betaCoefficients, betaCoefficientsOld);
    if(done){
      calculateInformationMatrix(informationMatrix, probabilites);
      break;
    }
  }

  delete betaCoefficientsOld;

  return;
}

void SimpleLogisticRegression::calculateProbabilities(LaVectorDouble& probabilites,
    const LaVectorDouble* betaCoefficients) {
  for(int i = 0; i < numberOfInstances; ++i){
    probabilites(i) = exp((*betaCoefficients)(0) + (*betaCoefficients)(1) * predictor(i))
        / (1 + exp((*betaCoefficients)(0) + (*betaCoefficients)(1) * predictor(i)));
  }
}

void SimpleLogisticRegression::calculateScoreAndLogLikelihood(LaVectorDouble& scores, double& logLikelihood,
    const LaVectorDouble& probabilites) {
  logLikelihood = 0;
  scores(0) = 0;
  scores(1) = 0;

  for(int i = 0; i < numberOfInstances; ++i){
    logLikelihood += binaryOutcomes(i) * log(probabilites(i)) + (1 - binaryOutcomes(i)) * log(1 - probabilites(i));
    scores(0) += binaryOutcomes(i) - probabilites(i);
    scores(1) += (binaryOutcomes(i) - probabilites(i)) * predictor(i);
  }
}

void SimpleLogisticRegression::calculateInformationMatrix(LaGenMatDouble& informationMatrix,
    const LaVectorDouble& probabilites) {
  informationMatrix(0, 0) = 0;
  informationMatrix(0, 1) = 0;
  informationMatrix(1, 1) = 0;

  for(int i = 0; i < numberOfInstances; ++i){
    informationMatrix(0, 0) += probabilites(i) * (1 - probabilites(i));
    informationMatrix(0, 1) += probabilites(i) * (1 - probabilites(i)) * predictor(i);
    informationMatrix(1, 1) += probabilites(i) * (1 - probabilites(i)) * predictor(i) * predictor(i);
  }
  informationMatrix(1, 0) = informationMatrix(0, 1);
}

void SimpleLogisticRegression::calculateNewBeta(LaVectorDouble* betaCoefficients,
    const LaVectorDouble* betaCoefficientsOld, LaGenMatDouble& informationMatrix, const LaVectorDouble& scores) {
//Calculate new beta=inverse(info_mat)*scores+oldBeta

  LaVectorDouble sigmaDiagonal = LaVectorDouble(numberOfPredictors);
  LaGenMatDouble uSVD = LaGenMatDouble(numberOfPredictors, numberOfPredictors);
  LaGenMatDouble vSVDTranspose = LaGenMatDouble(numberOfPredictors, numberOfPredictors);

  //Get the pseudo-inverse of the information matrix by using signular value decomposition
  LaSVD_IP(informationMatrix, sigmaDiagonal, uSVD, vSVDTranspose); //NOTE overwrites informationMatrix

  Blas_Mat_Trans_Vec_Mult(uSVD, scores, *betaCoefficients, 1, 0); //betaCoefficients=uSVD'*scores //NOTE using betaCoefficients for storage

  //Multiply the pseudo-inverse with beta
  //The pseudo-inverse is the sigmaDiagonal with all non-zero elements inverted
  int size_diag = sigmaDiagonal.size();
  for(int i = 0; i < size_diag; ++i){
    if(fabs(sigmaDiagonal(i)) > 1e-5){
      (*betaCoefficients)(i) = (*betaCoefficients)(i) / sigmaDiagonal(i);
    }else{
      (*betaCoefficients)(i) = 0;
    }
  }

  LaVectorDouble workAreaVector = LaVectorDouble(numberOfPredictors);
  Blas_Mat_Trans_Vec_Mult(vSVDTranspose, *betaCoefficients, workAreaVector, 1, 0); //workAreaVector=vSVDTranspose'*betaCoefficients

  //Copy over to beta and add old beta
  (*betaCoefficients)(0) = workAreaVector(0) + (*betaCoefficientsOld)(0);
  (*betaCoefficients)(1) = workAreaVector(1) + (*betaCoefficientsOld)(1);
}

}

