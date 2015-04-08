#include "EnvironmentVector.h"

namespace CuEira {
namespace Container {

template<typename Vector>
EnvironmentVector<Vector>::EnvironmentVector(const EnvironmentFactorHandler<Vector>& environmentFactorHandler) :
    numberOfIndividualsTotal(environmentFactorHandler.getNumberOfIndividualsTotal()), numberOfIndividualsToInclude(0), initialised(
        false), currentRecode(ALL_RISK), environmentFactor(environmentFactorHandler.getEnvironmentFactor()), envExMissing(
        nullptr), originalData(environmentFactorHandler.getEnvironmentData()), noMissing(false) {

}

template<typename Vector>
EnvironmentVector<Vector>::~EnvironmentVector() {
  delete envExMissing;
}

template<typename Vector>
int EnvironmentVector<Vector>::getNumberOfIndividualsTotal() const {
  return numberOfIndividualsTotal;
}

template<typename Vector>
int EnvironmentVector<Vector>::getNumberOfIndividualsToInclude() const {
#ifdef DEBUG
  if(!initialised){
    throw InvalidState("EnvironmentVector not initialised.");
  }
#endif
  return numberOfIndividualsToInclude;
}

template<typename Vector>
const EnvironmentFactor& EnvironmentVector<Vector>::getEnvironmentFactor() const {
  return environmentFactor;
}

template<typename Vector>
const Vector& EnvironmentVector<Vector>::getEnvironmentData() const {
#ifdef DEBUG
  if(!initialised){
    throw InvalidState("EnvironmentVector not initialised.");
  }
#endif
  return *envExMissing;
}

template<typename Vector>
Vector& EnvironmentVector<Vector>::getEnvironmentData() {
#ifdef DEBUG
  if(!initialised){
    throw InvalidState("EnvironmentVector not initialised.");
  }
#endif
  return *envExMissing;
}

template<typename Vector>
void EnvironmentVector<Vector>::recode(Recode recode) {
#ifdef DEBUG
  initialised = true;
#endif

  currentRecode = recode;
  if(!noMissing){
    delete envExMissing;
    envExMissing = new Vector(numberOfIndividualsTotal);
    numberOfIndividualsToInclude = numberOfIndividualsTotal;
  }

  if(recode == ENVIRONMENT_PROTECT || recode == INTERACTION_PROTECT){
    recodeProtective();
  }else{
    recodeAllRisk();
  }

  noMissing = true;
}

template<typename Vector>
void EnvironmentVector<Vector>::recode(Recode recode, const MissingDataHandler<Vector>& missingDataHandler) {
#ifdef DEBUG
  initialised = true;
#endif

  currentRecode = recode;
  numberOfIndividualsToInclude = missingDataHandler.getNumberOfIndividualsToInclude();
  delete envExMissing;
  envExMissing = new Vector(numberOfIndividualsToInclude);
  missingDataHandler.copyNonMissing(originalData, *envExMissing);

  if(recode == ENVIRONMENT_PROTECT || recode == INTERACTION_PROTECT){
    recodeProtective();
  }
  //No need to recode to all risk since it already is from the copying to remove the missing data

  noMissing = false;
}

} /* namespace Container */
} /* namespace CuEira */
