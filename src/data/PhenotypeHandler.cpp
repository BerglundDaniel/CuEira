#include "PhenotypeHandler.h"

namespace CuEira {

template<typename Vector>
PhenotypeHandler<Vector>::PhenotypeHandler(const Vector* vector) :
    vector(vector), numberOfIndividualsTotal(vector->getNumberOfRows()) {

}

template<typename Vector>
PhenotypeHandler<Vector>::~PhenotypeHandler() {
  delete vector;
}

template<typename Vector>
int PhenotypeHandler<Vector>::getNumberOfIndividualsTotal() const {
  return numberOfIndividualsTotal;
}

template<typename Vector>
const Vector& PhenotypeHandler<Vector>::getPhenotypeData() const {
  return *vector;
}

} /* namespace CuEira */
