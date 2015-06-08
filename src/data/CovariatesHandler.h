#ifndef COVARIATESHANDLER_H_
#define COVARIATESHANDLER_H_

namespace CuEira {

/*
 * This class ...
 *
 *  @author Daniel Berglund daniel.k.berglund@gmail.com
 */
template<typename Matrix>
class CovariatesHandler {
public:
  explicit CovariatesHandler(const Matrix* matrix);
  virtual ~CovariatesHandler();

  virtual int getNumberOfCovariates() const;
  virtual const Matrix& getCovariatesMatrix() const;

protected:
  const Matrix* matrix;
  const int numberOfCovariates;
};

} /* namespace CuEira */

#endif /* COVARIATESHANDLER_H_ */
