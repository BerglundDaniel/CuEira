#include "PhenotypeVector.h"

namespace CuEira {
namespace Container {

template<typename Vector>
PhenotypeVector<Vector>::PhenotypeVector(const PhenotypeHandler<Vector>& phenotypeHandler) :
    phenotypeHandler(phenotypeHandler), numberOfIndividualsTotal(phenotypeHandler.getNumberOfIndividuals()), numberOfIndividualsToInclude(
        0), initialised(false), noMissing(false), orgData(phenotypeHandler.getPhenotypeData()), phenotypeExMissing(
        nullptr) {

}

template<typename Vector>
PhenotypeVector<Vector>::~PhenotypeVector() {
  delete phenotypeExMissing;
}

template<typename Vector>
int PhenotypeVector<Vector>::getNumberOfIndividualsTotal() const {
  return numberOfIndividualsTotal;
}

template<typename Vector>
int PhenotypeVector<Vector>::getNumberOfIndividualsToInclude() const {
#ifdef DEBUG
  if(!initialised){
    throw new InvalidState("PhenotypeVector not initialised.");
  }
#endif

  return numberOfIndividualsToInclude;
}

template<typename Vector>
const Vector& PhenotypeVector<Vector>::getPhenotypeData() const {
#ifdef DEBUG
  if(!initialised){
    throw new InvalidState("PhenotypeVector not initialised.");
  }
#endif

  if(noMissing){
    return orgData;
  }else{
    return *phenotypeExMissing;
  }
}

template<typename Vector>
void PhenotypeVector<Vector>::applyMissing(const MissingDataHandler& missingDataHandler) {
  initialised = true;
  noMissing = false;
  numberOfIndividualsToInclude = missingDataHandler.getNumberOfIndividualsToInclude();

  delete phenotypeExMissing;
  phenotypeExMissing = new Vector(numberOfIndividualsToInclude);
  missingDataHandler.copyNonMissing(orgData, *phenotypeExMissing);
}

template<typename Vector>
void PhenotypeVector<Vector>::applyMissing() {
  initialised = true;
  noMissing = true;
  numberOfIndividualsToInclude = numberOfIndividualsTotal;
}

} /* namespace Container */
} /* namespace CuEira */
