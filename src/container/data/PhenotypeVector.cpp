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
  return numberOfIndividualsToInclude;
}

void PhenotypeVector::applyMissing(const std::set<int>& personsToSkip) {
  initialised = true;
  noMissing = false;
  numberOfIndividualsToInclude = numberOfIndividualsTotal - personsToSkip.size();

  copyNonMissingData(indexesToCopy);
}

void PhenotypeVector::applyNoMissing() {
  initialised = true;
  noMissing = true;
  numberOfIndividualsToInclude = numberOfIndividualsTotal;
}

} /* namespace Container */
} /* namespace CuEira */
