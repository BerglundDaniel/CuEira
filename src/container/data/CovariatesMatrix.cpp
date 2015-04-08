#include "CovariatesMatrix.h"

namespace CuEira {
namespace Container {

template<typename Matrix, typename Vector>
CovariatesMatrix<Matrix, Vector>::CovariatesMatrix(const CovariatesHandler<Matrix>& covariatesHandler) :
    orgData(covariatesHandler.getCovariatesMatrix()), numberOfIndividualsTotal(orgData.getNumberOfRows()), numberOfIndividualsToInclude(
        0), initialised(false), noMissing(false), covariatesExMissing(nullptr), numberOfCovariates(
        covariatesHandler.getNumberOfCovariates()) {

}

template<typename Matrix, typename Vector>
CovariatesMatrix<Matrix, Vector>::~CovariatesMatrix() {
  delete covariatesExMissing;
}

template<typename Matrix, typename Vector>
const Matrix& CovariatesMatrix<Matrix, Vector>::getCovariatesData() const {
#ifdef DEBUG
  if(!initialised){
    throw new InvalidState("CovariatesMatrix not initialised.");
  }
#endif
  return *covariatesExMissing;
}

template<typename Matrix, typename Vector>
void CovariatesMatrix<Matrix, Vector>::applyMissing(const MissingDataHandler<Vector>& missingDataHandler) {
  initialised = true;
  noMissing = false;
  numberOfIndividualsToInclude = missingDataHandler.getNumberOfIndividualsToInclude();

  delete covariatesExMissing;
  covariatesExMissing = new Matrix(numberOfIndividualsToInclude, numberOfCovariates);

  for(int i = 0; i < numberOfCovariates; ++i){
    Vector* orgDataVector = orgData(i);
    Vector* covariatesExMissingVector = covariatesExMissing(i);

    missingDataHandler.copyNonMissing(orgDataVector, covariatesExMissingVector);

    delete orgDataVector;
    delete covariatesExMissingVector;
  }

}

template<typename Matrix, typename Vector>
virtual void CovariatesMatrix<Matrix, Vector>::applyMissing() {
  initialised = true;
  noMissing = true;
  numberOfIndividualsToInclude = numberOfIndividualsTotal;
}

} /* namespace Container */
} /* namespace CuEira */
