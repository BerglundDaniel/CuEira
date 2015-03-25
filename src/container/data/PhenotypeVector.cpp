#include "PhenotypeVector.h"

namespace CuEira {
namespace Container {

PhenotypeVector::PhenotypeVector(const PhenotypeHandler& phenotypeHandler) :
    numberOfIndividualsTotal(phenotypeHandler.getNumberOfIndividuals()), numberOfIndividualsToInclude(0), initialised(
        false), noMissing(false) {

}

PhenotypeVector::~PhenotypeVector() {

}

int PhenotypeVector::getNumberOfIndividualsTotal() const {
  return numberOfIndividualsTotal;
}

int PhenotypeVector::getNumberOfIndividualsToInclude() const {
#ifdef DEBUG
  if(!initialised){
    throw new InvalidState("PhenotypeVector not initialised.");
  }
#endif

  return numberOfIndividualsToInclude;
}

void PhenotypeVector::applyMissing(const MissingDataHandler& missingDataHandler) {
  initialised = true;
  noMissing = false;
  numberOfIndividualsToInclude = missingDataHandler.getNumberOfIndividualsToInclude();
}

void PhenotypeVector::applyNoMissing() {
  initialised = true;
  noMissing = true;
  numberOfIndividualsToInclude = numberOfIndividualsTotal;
}

} /* namespace Container */
} /* namespace CuEira */
