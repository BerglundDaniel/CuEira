#ifndef COVARIATESMATRIX_H_
#define COVARIATESMATRIX_H_

#include <CovariatesHandler.h>
#include <MissingDataHandler.h>

namespace CuEira {
namespace Container {

/**
 * This class ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
template<typename Matrix>
class CovariatesMatrix {
public:
  explicit CovariatesMatrix(const CovariatesHandler<Matrix>& covariatesHandler);
  virtual ~CovariatesMatrix();

  virtual const Matrix& getCovariatesData() const;
  virtual void applyMissing(const MissingDataHandler& missingDataHandler);
  virtual void applyMissing();

private:
  const Matrix& orgData;
  const Matrix* covariatesExMissing;

  const int numberOfIndividualsTotal;
  const int numberOfCovariates;
  int numberOfIndividualsToInclude;
  bool initialised;
  bool noMissing;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* COVARIATESMATRIX_H_ */
