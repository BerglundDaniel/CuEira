#include "SNPVector.h"

namespace CuEira {
namespace Container {

#ifdef CPU
SNPVector::SNPVector(const std::vector<int>* originalSNPData, SNP& snp, GeneticModel geneticModel) :
snp(snp), numberOfIndividualsToInclude(originalSNPData->size()), originalSNPData(originalSNPData), modifiedSNPData(
    new LapackppHostVector(new LaVectorDouble(numberOfIndividualsToInclude))), currentRecode(ALL_RISK), originalGeneticModel(
    geneticModel), originalRiskAllele(snp.getRiskAllele()), currentRiskAllele(snp.getRiskAllele()), currentGeneticModel(
    geneticModel){
#else
SNPVector::SNPVector(const std::vector<int>* originalSNPData, SNP& snp, GeneticModel geneticModel) :
    snp(snp), numberOfIndividualsToInclude(originalSNPData->size()), originalSNPData(originalSNPData), modifiedSNPData(
        new PinnedHostVector(numberOfIndividualsToInclude)), currentRecode(ALL_RISK), originalGeneticModel(
        geneticModel), originalRiskAllele(snp.getRiskAllele()), currentRiskAllele(snp.getRiskAllele()), currentGeneticModel(
        geneticModel) {
#endif

  currentRecode = SNP_PROTECT; //Because of the if statement in recode
  recode(ALL_RISK);
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
    //Doesn't affect the snp data
    currentRecode = ENVIRONMENT_PROTECT;
    return;
  }else if(recode == INTERACTION_PROTECT){
    recodeInteractionProtective();
  }else{
    throw InvalidState("Unknown recode for a SNPVector.");
  }

  doRecode();
}

void SNPVector::recodeAllRisk() {
  currentRecode = ALL_RISK;
  currentRiskAllele = originalRiskAllele;
  currentGeneticModel = originalGeneticModel;
  snp.setRiskAllele(currentRiskAllele);
}

void SNPVector::recodeSNPProtective() {
  currentRecode = SNP_PROTECT;
  currentRiskAllele = invertRiskAllele(originalRiskAllele);
  currentGeneticModel = RECESSIVE;
  snp.setRiskAllele(currentRiskAllele);
}

void SNPVector::recodeInteractionProtective() {
  //FIXME is this correct?
  currentRecode = INTERACTION_PROTECT;
  currentRiskAllele = invertRiskAllele(originalRiskAllele);
  currentGeneticModel = RECESSIVE;
  snp.setRiskAllele(currentRiskAllele);
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
  int* snpData0;
  int* snpData1;
  int* snpData2;
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

} /* namespace Container */
} /* namespace CuEira */
