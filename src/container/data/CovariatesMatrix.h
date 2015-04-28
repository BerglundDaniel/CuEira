#ifndef COVARIATESMATRIX_H_
#define COVARIATESMATRIX_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <CovariatesHandler.h>
#include <MissingDataHandler.h>

namespace CuEira {
namespace Container {
class CpuCovariatesMatrixTest;
class CudaCovariatesMatrixTest;

/**
 * This class ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
template<typename Matrix, typename Vector>
class CovariatesMatrix {
  friend CpuCovariatesMatrixTest;
  friend CudaCovariatesMatrixTest;
  FRIEND_TEST(CpuCovariatesMatrixTest, ApplyMissing);
  FRIEND_TEST(CudaCovariatesMatrixTest, ApplyMissing);
public:
  explicit CovariatesMatrix(const CovariatesHandler<Matrix>& covariatesHandler);
  virtual ~CovariatesMatrix();

  virtual const Matrix& getCovariatesData() const;
  virtual void applyMissing(const MissingDataHandler<Vector>& missingDataHandler);
  virtual void applyMissing();

private:
  const Matrix& orgData;
  Matrix* covariatesExMissing;

  const int numberOfIndividualsTotal;
  const int numberOfCovariates;
  int numberOfIndividualsToInclude;
  bool initialised;
  bool noMissing;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* COVARIATESMATRIX_H_ */
