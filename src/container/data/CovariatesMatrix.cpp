#include "CovariatesMatrix.h"

namespace CuEira {
namespace Container {

template<typename Matrix>
CovariatesMatrix<Matrix>::CovariatesMatrix(const CovariatesHandler<Matrix>& covariatesHandler) :
    orgData(covariatesHandler.getCovariatesMatrix()), numberOfIndividualsTotal(orgData.getNumberOfRows()), numberOfIndividualsToInclude(
        0), initialised(false), noMissing(false), covariatesExMissing(nullptr), numberOfCovariates(
        covariatesHandler.getNumberOfCovariates()) {

}

template<typename Matrix>
CovariatesMatrix<Matrix>::~CovariatesMatrix() {
  delete covariatesExMissing;
}

template<typename Matrix>
const Matrix& CovariatesMatrix<Matrix>::getCovariatesData() const {
#ifdef DEBUG
  if(!initialised){
    throw new InvalidState("CovariatesMatrix not initialised.");
  }
#endif
  return *covariatesExMissing;
}

template<typename Matrix>
void CovariatesMatrix<Matrix>::applyMissing(const MissingDataHandler& missingDataHandler) {
  initialised = true;
  noMissing = false;
  numberOfIndividualsToInclude = missingDataHandler.getNumberOfIndividualsToInclude();

  delete covariatesExMissing;
  covariatesExMissing = new Matrix(numberOfIndividualsToInclude, numberOfCovariates);

  for(int i = 0; i < numberOfCovariates; ++i){
    //TODO the vectors....
    missingDataHandler.copyNonMissing(orgDataVector, covariatesExMissingVector);
  }

}

template<typename Matrix>
virtual void CovariatesMatrix<Matrix>::applyMissing() {
  initialised = true;
  noMissing = true;
  numberOfIndividualsToInclude = numberOfIndividualsTotal;
}

} /* namespace Container */
} /* namespace CuEira */
