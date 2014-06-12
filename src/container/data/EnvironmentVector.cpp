#include "EnvironmentVector.h"

namespace CuEira {
namespace Container {

EnvironmentVector::EnvironmentVector(const EnvironmentFactorHandler& environmentHandler,
    EnvironmentFactor& environmentFactor) :
    numberOfIndividualsToInclude(originalData->getNumberOfRows()), currentRecode(ALL_RISK), environmentHandler(
        environmentHandler), originalData(&environmentHandler.getData(environmentFactor)), environmentFactor(
        environmentFactor),
#ifdef CPU
        recodedData(new LapackppHostVector(new LaVectorDouble(numberOfIndividualsToInclude)))
#else
        recodedData(new PinnedHostVector(numberOfIndividualsToInclude))
#endif
{
  recodeAllRisk();
}

EnvironmentVector::~EnvironmentVector() {
  delete recodedData;
}

void EnvironmentVector::switchEnvironmentFactor(EnvironmentFactor& environmentFactor) {
  this->environmentFactor = environmentFactor;

  originalData = &environmentHandler.getData(environmentFactor);

  currentRecode=ALL_RISK;
  recodeAllRisk();
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
    recodeAllRisk();
  }else if(recode == ENVIRONMENT_PROTECT){
    recodeEnvironmentProtective();
  }else if(recode == INTERACTION_PROTECT){
    recodeInteractionProtective();
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

void EnvironmentVector::recodeEnvironmentProtective() {
  if(environmentFactor.getVariableType() == BINARY){
    for(int i = 0; i < numberOfIndividualsToInclude; ++i){
      if((*originalData)(i) == 0){
        (*recodedData)(i) = 1;
      }else{
        (*recodedData)(i) = 0;
      }
    }
  }else{
    for(int i = 0; i < numberOfIndividualsToInclude; ++i){
      //FIXME is this correct?
      (*recodedData)(i) = (*originalData)(i) * -1;
    }
  }
}

void EnvironmentVector::recodeAllRisk() {
  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    (*recodedData)(i) = (*originalData)(i);
  }
}

void EnvironmentVector::recodeInteractionProtective() {
  recodeEnvironmentProtective();
}

const EnvironmentFactor& EnvironmentVector::getCurrentEnvironmentFactor() const {
  return environmentFactor;
}

} /* namespace Container */
} /* namespace CuEira */
