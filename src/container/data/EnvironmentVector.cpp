#include "EnvironmentVector.h"

namespace CuEira {
namespace Container {

EnvironmentVector::EnvironmentVector(const EnvironmentFactor& environmentFactor, const int numberOfIndividualsTotal) :
    numberOfIndividualsToInclude(0), initialised(false), noMissing(false), currentRecode(ALL_RISK), environmentFactor(
        environmentFactor) {

}

EnvironmentVector::~EnvironmentVector() {

}

int EnvironmentVector::getNumberOfIndividualsTotal() const {
  return numberOfIndividualsTotal;
}

int EnvironmentVector::getNumberOfIndividualsToInclude() const {
#ifdef DEBUG
  if(!initialised){
    throw InvalidState("EnvironmentVector not initialised.");
  }
#endif
  return numberOfIndividualsToInclude;
}

const EnvironmentFactor& EnvironmentVector::getEnvironmentFactor() const {
  return environmentFactor;
}

} /* namespace Container */
} /* namespace CuEira */
