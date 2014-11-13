#include "SNPVector.h"

namespace CuEira {
namespace Container {

SNPVector::SNPVector(SNP& snp, GeneticModel geneticModel, const std::vector<int>* originalSNPData) :
    snp(snp), numberOfIndividualsToInclude(originalSNPData->size()), originalSNPData(originalSNPData), originalGeneticModel(
        geneticModel), currentGeneticModel(geneticModel), currentRecode(ALL_RISK), originalRiskAllele(
        snp.getRiskAllele()),
#ifdef CPU
        modifiedSNPData(
            new RegularHostVector(numberOfIndividualsToInclude))
#else
        modifiedSNPData(new PinnedHostVector(numberOfIndividualsToInclude))
#endif
{
  recodeAllRisk();
  doRecode();
}

SNPVector::SNPVector(SNP& snp) :
    snp(snp), numberOfIndividualsToInclude(0), originalSNPData(nullptr), originalGeneticModel(DOMINANT), currentGeneticModel(
        DOMINANT), currentRecode(ALL_RISK), originalRiskAllele(ALLELE_ONE), modifiedSNPData(nullptr) {

}

SNPVector::~SNPVector() {
  delete modifiedSNPData;
  delete originalSNPData;
}

const std::vector<int>& SNPVector::getOrginalData() const {
  return *originalSNPData;
}

const Container::HostVector& SNPVector::getRecodedData() const {
  return *modifiedSNPData;
}

int SNPVector::getNumberOfIndividualsToInclude() const {
  return numberOfIndividualsToInclude;
}

const SNP & SNPVector::getAssociatedSNP() const {
  return snp;
}

void SNPVector::recode(Recode recode) {
  currentRecode = recode;
  if(recode == ALL_RISK){
    recodeAllRisk();
  }else if(recode == SNP_PROTECT){
    recodeSNPProtective();
  }else if(recode == INTERACTION_PROTECT){
    recodeInteractionProtective();
  }else{
    recodeAllRisk();
  }

  doRecode();
}

void SNPVector::recodeAllRisk() {
  currentGeneticModel = originalGeneticModel;
  snp.setRiskAllele(originalRiskAllele);
}

void SNPVector::recodeSNPProtective() {
  currentGeneticModel = RECESSIVE; //TODO invert genetic
  snp.setRiskAllele(invertRiskAllele(originalRiskAllele));
}

void SNPVector::recodeInteractionProtective() {
  currentGeneticModel = RECESSIVE; //TODO invert genetic
  snp.setRiskAllele(invertRiskAllele(originalRiskAllele));
}

RiskAllele SNPVector::invertRiskAllele(RiskAllele riskAllele) {
  if(riskAllele == ALLELE_ONE){
    return ALLELE_TWO;
  }else if(riskAllele == ALLELE_TWO){
    return ALLELE_ONE;
  }else{
    throw InvalidState("Unknown RiskAllele in SNPVector");
  }
}

void SNPVector::doRecode() {
  RiskAllele currentRiskAllele = snp.getRiskAllele();
  int* snpData0 = new int(-1);
  int* snpData1 = new int(-1);
  int* snpData2 = new int(-1);

  if(currentGeneticModel == DOMINANT){
    if(currentRiskAllele == ALLELE_ONE){
      *snpData0 = 1;
      *snpData1 = 1;
      *snpData2 = 0;
    }else if(currentRiskAllele == ALLELE_TWO){
      *snpData0 = 0;
      *snpData1 = 1;
      *snpData2 = 1;
    }else{
      throw InvalidState("Unknown RiskAllele in SNPVector");
    }
  }else if(currentGeneticModel == RECESSIVE){
    if(currentRiskAllele == ALLELE_ONE){
      *snpData0 = 1;
      *snpData1 = 0;
      *snpData2 = 0;
    }else if(currentRiskAllele == ALLELE_TWO){
      *snpData0 = 0;
      *snpData1 = 0;
      *snpData2 = 1;
    }else{
      throw InvalidState("Unknown RiskAllele in SNPVector");
    }
  }else{
    throw InvalidState("Unknown genetic model in SNPVector");
  }

  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    int orgData = (*originalSNPData)[i];
    if(orgData == 0){
      (*modifiedSNPData)(i) = *snpData0;
    }else if(orgData == 1){
      (*modifiedSNPData)(i) = *snpData1;
    }else if(orgData == 2){
      (*modifiedSNPData)(i) = *snpData2;
    }else{
      throw InvalidState("Original SNP data is not 0,1 or 2 in a SNPVector.");
    }
  } /* for i */

  delete snpData0;
  delete snpData1;
  delete snpData2;
}

void SNPVector::applyStatisticModel(StatisticModel statisticModel, const HostVector& interactionVector) {
  if(statisticModel == ADDITIVE){
    for(int i = 0; i < numberOfIndividualsToInclude; ++i){
      if(interactionVector(i) != 0){
        (*modifiedSNPData)(i) = 0;
      }
    }
  }
  return;
}

} /* namespace Container */
} /* namespace CuEira */
