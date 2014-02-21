#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

/*
 Abstract class for logistic regression
 */

#include <iostream>
#include <math.h>
#include <lapackpp/gmd.h> //matrix double
#include <lapackpp/lavd.h> //vector double
#include <lapackpp/laexcp.h> //exceptions
namespace LogisticRegression {

class LogisticRegression {

public:
  LogisticRegression(LaVectorDouble* betaCoefficients, const int MAXIT, const double CONVERGENCETHRESHOLD);
  ~LogisticRegression();

  virtual void calculate()=0;

  bool checkBreakConditions(const LaVectorDouble* betaCoefficients, const LaVectorDouble* betaCoefficientsOld);
  const LaGenMatDouble& getInformationMatrix();
  int getNumberOfIterations();
  int getMaximumNumberOfIterations();
  double getConvergenceThreshold();
  double getLogLikelihood();
  const LaVectorDouble& getBeta();
protected:
  int currentIteration;
  const int MAXIT;
  const double CONVERGENCETHRESHOLD;
  double logLikelihood;

  LaVectorDouble* betaCoefficients;
  LaGenMatDouble informationMatrix;
};
}

#endif // LOGISTICREGRESSION_H
