#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <iostream>
#include <math.h>
#include <lapackpp/gmd.h> //matrix double
#include <lapackpp/lavd.h> //vector double
#include <lapackpp/laexcp.h> //exceptions

namespace CuEira {
namespace Model {
namespace LogisticRegression {
namespace Serial {

/**
 * This is an abstract class for logistic regression
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LogisticRegression {

public:
  LogisticRegression(LaVectorDouble* betaCoefficients, const int MAXIT, const double CONVERGENCETHRESHOLD);
  virtual ~LogisticRegression();

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

} /* namespace Serial */
} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

#endif // LOGISTICREGRESSION_H
