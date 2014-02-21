#ifndef MULTIPLELOGISTICREGRESSION_H
#define MULTIPLELOGISTICREGRESSION_H

/*
 Multiple logistic regression
 */

#include <iostream>
#include <math.h>
#include <algorithm>

#include <lapackpp/gmd.h> //matrix double
#include <lapackpp/lavd.h> //vector double
#include <lapackpp/laexcp.h> //exceptions
#include <lapackpp/blas1pp.h> //vector - vector operations
#include <lapackpp/blas2pp.h> //matrix - vector operations
#include <lapackpp/blas3pp.h> //matrix-matrix operations
#include <lapackpp/lasvd.h> //singular value decomp
#include "LogisticRegression.h"

namespace LogisticRegression {

class MultipleLogisticRegression: public LogisticRegression {

public:
  MultipleLogisticRegression(const LaGenMatDouble& predictors, const LaVectorDouble& binaryOutcomes,
      LaVectorDouble* betaCoefficients, const int MAXIT = 500, const double CONVERGENCETRESHOLD = 1e-3);
  ~MultipleLogisticRegression();

  void calculate();
private:
  const LaGenMatDouble* addColumnForIntercept(const LaGenMatDouble& predictors);
  void calculateProbabilitiesScoreAndLogLikelihood(LaVectorDouble& probabilites, LaVectorDouble& scores,
      double& logLikelihood, const LaVectorDouble* betaCoefficients, LaVectorDouble workVectorNx1);
  void calculateNewBeta(LaVectorDouble* betaCoefficients, const LaVectorDouble* betaCoefficientsOld,
      LaGenMatDouble& informationMatrix, const LaVectorDouble& scores);
  void calculateInformationMatrix(LaGenMatDouble& informationMatrix, const LaVectorDouble& probabilites,
      LaVectorDouble workVectorNx1);

  const int numberOfInstances;
  const int numberOfPredictors;

  const LaGenMatDouble* predictors;
  const LaVectorDouble& binaryOutcomes;
};
}

#endif // MULTIPLELOGISTICREGRESSION_H
