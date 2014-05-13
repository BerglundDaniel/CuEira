#ifndef SIMPLELOGISTICREGRESSION_H
#define SIMPLELOGISTICREGRESSION_H

/*
 Simple logistic regression
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

class SimpleLogisticRegression: public LogisticRegression {

public:
  SimpleLogisticRegression(const LaVectorDouble& predictor, const LaVectorDouble& binaryOutcomes,
      LaVectorDouble* betaCoefficients, const int MAXIT = 500, const double CONVERGENCETRESHOLD = 1e-3);
  ~SimpleLogisticRegression();

  void calculate();
private:
  void calculateProbabilities(LaVectorDouble& probabilites, const LaVectorDouble* betaCoefficients);
  void calculateScoreAndLogLikelihood(LaVectorDouble& scores, double& logLikelihood,
      const LaVectorDouble& probabilites);
  void calculateInformationMatrix(LaGenMatDouble& informationMatrix, const LaVectorDouble& probabilites);
  void calculateNewBeta(LaVectorDouble* betaCoefficients, const LaVectorDouble* betaCoefficientsOld,
      LaGenMatDouble& informationMatrix, const LaVectorDouble& scores);

  const int numberOfInstances;
  const int numberOfPredictors;

  const LaVectorDouble& predictor;
  const LaVectorDouble& binaryOutcomes;
};
}

#endif // SIMPLELOGISTICREGRESSION_H