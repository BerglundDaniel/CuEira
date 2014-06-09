#include "EnvironmentVector.h"

namespace CuEira {
namespace Container {

EnvironmentVector::EnvironmentVector(const EnvironmentFactorHandler& environmentHandler,
    EnvironmentFactor& environmentFactor) :
    numberOfIndividualsToInclude(), currentRecode(ALL_RISK), environmentHandler(environmentHandler), originalData(
        environmentHandler.getData(environmentFactor)),
#ifdef CPU
        recodedData(new LapackppHostVector(new LaVectorDouble(numberOfIndividualsToInclude)))
#else
        recodedData(new PinnedHostVector(numberOfIndividualsToInclude))
#endif
{

}

EnvironmentVector::~EnvironmentVector() {
  delete recodedData;
}

int EnvironmentVector::getNumberOfIndividualsToInclude() const {
  return numberOfIndividualsToInclude;
}

const Container::HostVector& EnvironmentVector::getRecodedData() const {
  return *recodedData;
}

void EnvironmentVector::recode(Recode recode) {
  if(currentRecode == recode){
    return;
  }

  currentRecode = recode;
  if(recode == ALL_RISK){
    for(int i = 0; i < numberOfIndividualsToInclude; ++i){
      (*recodedData)(i) = originalData(i);
    }
  }else if(recode == ENVIRONMENT_PROTECT){
    for(int i = 0; i < numberOfIndividualsToInclude; ++i){
      (*recodedData)(i) = invertEnvironmentFactor(originalData(i));
    }
  }
}

void EnvironmentVector::applyStatisticModel(StatisticModel statisticModel, const HostVector& interactionVector) {
  if(statisticModel == ADDITIVE){
    for(int i = 0; i < numberOfIndividualsToInclude; ++i){
      if(interactionVector(i) != 0){
        (*recodedData)(i) = 0;
      }
    }
  }
  return;
}

PRECISION EnvironmentVector::invertEnvironmentFactor(PRECISION envData) const {
  //FIXME is this correct?

  return envData * -1;
}

} /* namespace Container */
} /* namespace CuEira */
