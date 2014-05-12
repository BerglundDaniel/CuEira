#include "SNPVector.h"

namespace CuEira {
namespace Container {

#ifdef CPU
SNPVector::SNPVector(const std::vector<int>* originalSNPData, GeneticModel geneticModel) :
numberOfIndividualsToInclude(originalSNPData->size()), originalSNPData(originalSNPData), modifiedSNPData(
    new LapackppHostVector(new LaVectorDouble(numberOfIndividualsToInclude))), currentRecode(ALL_RISK), geneticModel(
        geneticModel){
#else
SNPVector::SNPVector(const std::vector<int>* originalSNPData, GeneticModel geneticModel) :
    numberOfIndividualsToInclude(originalSNPData->size()), originalSNPData(originalSNPData), modifiedSNPData(
        new PinnedHostVector(numberOfIndividualsToInclude)), currentRecode(ALL_RISK), geneticModel(
        geneticModel) {
#endif
}

SNPVector::~SNPVector() {
  delete originalSNPData;
  delete modifiedSNPData;
}

const std::vector<int>* SNPVector::getOrginalData() const {
  return originalSNPData;
}

const Container::HostVector* SNPVector::getRecodedData() const {
  return modifiedSNPData;
}

Recode SNPVector::getRecode() const {
  return currentRecode;
}

void SNPVector::recode(Recode recode) {
  if(currentRecode == recode){
    return;
  }

  if(recode == ALL_RISK){
    recodeAllRisk();
  }else if(recode == SNP_PROTECT){
    recodeSNPProtective();
  }else if(recode == ENVIRONMENT_PROTECT){
    recodeEnvironmentProtective();
  }else if(recode == INTERACTION_PROTECT){
    recodeInteractionProtective();
  }else{
    throw InvalidState("Unknown recode for a SNPVector.");
  }

  currentRecode = recode;
}

void SNPVector::recodeAllRisk() {
  if(geneticModel == DOMINANT){
    for(int i = 0; i < numberOfIndividualsToInclude; ++i){
      int orgData = (*originalSNPData)[i];
      if(orgData == 0){
        (*modifiedSNPData)(i) = 0;
      }else if(orgData == 1){
        (*modifiedSNPData)(i) = 0;
      }else if(orgData == 2){
        (*modifiedSNPData)(i) = 0;
      }else{
        throw InvalidState("Original SNP data is not 0,1 or 2 in a SNPVector.");
      }
    } /* for i */
  }else if(geneticModel == RECESSIVE){
    for(int i = 0; i < numberOfIndividualsToInclude; ++i){
      int orgData = (*originalSNPData)[i];
      if(orgData == 0){
        (*modifiedSNPData)(i) = 0;
      }else if(orgData == 1){
        (*modifiedSNPData)(i) = 0;
      }else if(orgData == 2){
        (*modifiedSNPData)(i) = 0;
      }else{
        throw InvalidState("Original SNP data is not 0,1 or 2 in a SNPVector.");
      }
    } /* for i */
  }else{
    throw InvalidState("Unknown genetic model in SNPVector");
  }
}

void SNPVector::recodeSNPProtective() {

}

void SNPVector::recodeEnvironmentProtective() {

}

void SNPVector::recodeInteractionProtective() {

}

} /* namespace Container */
} /* namespace CuEira */
