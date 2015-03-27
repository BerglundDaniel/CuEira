#include "CovariatesHandler.h"

namespace CuEira {

template<typename Matrix>
CovariatesHandler<Matrix>::CovariatesHandler(const Matrix* matrix) :
    matrix(matrix), numberOfCovariates(matrix->getNumberOfColumns()) {
}

template<typename Matrix>
CovariatesHandler<Matrix>::~CovariatesHandler() {
  delete matrix;
}

template<typename Matrix>
int CovariatesHandler<Matrix>::getNumberOfCovariates() const {
  return numberOfCovariates;
}

template<typename Matrix>
const Matrix& CovariatesHandler<Matrix>::getCovariatesMatrix() const {
  return *matrix;
}

} /* namespace CuEira */
