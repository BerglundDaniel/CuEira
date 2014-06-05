#include "EnvironmentVector.h"

namespace CuEira {
namespace Container {

EnvironmentVector::EnvironmentVector() :
    numberOfIndividualsToInclude(),
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
  if(recode == ALL_RISK){
    for(int i = 0; i < numberOfIndividualsToInclude; ++i){
      (*recodedData)(i) = 0; //FIXME =orgdata
    }
  }else if(recode == ENVIRONMENT_PROTECT){
    for(int i = 0; i < numberOfIndividualsToInclude; ++i){
      (*recodedData)(i) = 0; //FIXME =inverse av orgdata, correct?
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

} /* namespace Container */
} /* namespace CuEira */
