#include "EnvironmentFactorHandler.h"

namespace CuEira {

template<typename Vector>
EnvironmentFactorHandler<Vector>::EnvironmentFactorHandler(std::shared_ptr<const EnvironmentFactor> environmentFactor,
    const Vector* vector) :
    vector(vector), environmentFactor(environmentFactor), numberOfIndividualsTotal(vector->getNumberOfRows()) {

}

template<typename Vector>
EnvironmentFactorHandler<Vector>::~EnvironmentFactorHandler() {
  delete vector;
}

template<typename Vector>
int EnvironmentFactorHandler<Vector>::getNumberOfIndividualsTotal() const {
  return numberOfIndividualsTotal;
}

template<typename Vector>
const EnvironmentFactor& EnvironmentFactorHandler<Vector>::getEnvironmentFactor() const {
  return environmentFactor;
}

template<typename Vector>
const Vector& EnvironmentFactorHandler<Vector>::getEnvironmentData() const {
  return *vector;
}

} /* namespace CuEira */
