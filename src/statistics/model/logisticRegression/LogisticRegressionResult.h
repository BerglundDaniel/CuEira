#ifndef LOGISTICREGRESSIONRESULT_H_
#define LOGISTICREGRESSIONRESULT_H_

#include <HostVector.h>
#include <HostMatrix.h>
#include <Recode.h>

namespace CuEira {
namespace Model {
namespace LogisticRegression {

/**
 * This class hold the results from an logistic regression
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LogisticRegressionResult {
public:
  /**
   * Constructor for the class, when the destructor runs it will delete the pointers.
   */
  LogisticRegressionResult(Container::HostVector* beta, Container::HostMatrix* informationMatrix,
      Container::HostMatrix* inverseInformationMatrixHost, int numberOfIterations, PRECISION logLikelihood);
  virtual ~LogisticRegressionResult();

  /**
   * Get beta
   */
  virtual const Container::HostVector& getBeta() const;

  /**
   * Get the information matrix
   */
  virtual const Container::HostMatrix& getInformationMatrix() const;

  /**
   * Get the inverse of the information matrix
   */
  virtual const Container::HostMatrix& getInverseInformationMatrix() const;

  /**
   * Get the number of iterations
   */
  virtual int getNumberOfIterations() const;

  /**
   * Get the log likelihood
   */
  virtual PRECISION getLogLikelihood() const;

  /**
   * Calculate if there is a need for recoding
   */
  virtual Recode calculateRecode() const;

protected:
  LogisticRegressionResult(); //For the mock

private:
  Container::HostVector* beta;
  Container::HostMatrix* informationMatrix;
  Container::HostMatrix* inverseInformationMatrixHost;
  int numberOfIterations;
  PRECISION logLikelihood;
};

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

#endif /* LOGISTICREGRESSIONRESULT_H_ */
