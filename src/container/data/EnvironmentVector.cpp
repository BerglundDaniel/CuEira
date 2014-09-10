#include "EnvironmentVector.h"

namespace CuEira {
namespace Container {

EnvironmentVector::EnvironmentVector(const EnvironmentFactorHandler& environmentHandler) :
    numberOfIndividualsToInclude(environmentHandler.getNumberOfIndividualsToInclude()), currentRecode(ALL_RISK), environmentHandler(
        &environmentHandler), originalData(nullptr), state(NOT_INITIALISED), environmentFactor(nullptr),
#ifdef CPU
        recodedData(new LapackppHostVector(new LaVectorDouble(numberOfIndividualsToInclude)))
#else
        recodedData(new PinnedHostVector(numberOfIndividualsToInclude))
#endif
{

}

EnvironmentVector::EnvironmentVector() :
    recodedData(nullptr) {

}

EnvironmentVector::~EnvironmentVector() {
  delete recodedData;
  delete originalData;
}

void EnvironmentVector::switchEnvironmentFactor(const EnvironmentFactor& environmentFactor) {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    state = INITIALISED;
  }
#endif

  this->environmentFactor = &environmentFactor;
  delete originalData;
  originalData = environmentHandler->getData(environmentFactor);

  currentRecode = ALL_RISK;
  recodeAllRisk();
}

int EnvironmentVector::getNumberOfIndividualsToInclude() const {
  return numberOfIndividualsToInclude;
}

const Container::HostVector& EnvironmentVector::getRecodedData() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Have to use switch on EnvironmentVector before get.");
  }
#endif
  return *recodedData;
}

void EnvironmentVector::recode(Recode recode) {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Have to use switch on EnvironmentVector before recode.");
  }
#endif

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
  }else{
    recodeAllRisk();
  }
}

void EnvironmentVector::applyStatisticModel(StatisticModel statisticModel, const HostVector& interactionVector) {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Have to use switch on EnvironmentVector before applyStatisticModel.");
  }
#endif

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
  if(environmentFactor->getVariableType() == BINARY){
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
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Have to use switch on EnvironmentVector before getCurrentEnvironmentFactor.");
  }
#endif

  return *environmentFactor;
}

} /* namespace Container */
} /* namespace CuEira */
