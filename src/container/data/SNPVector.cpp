#include "SNPVector.h"

namespace CuEira {
namespace Container {

SNPVector::SNPVector(SNP& snp, GeneticModel geneticModel, const std::vector<int>* originalSNPData,
    const std::vector<int>* numberOfAlleles, const std::vector<double>* alleleFrequencies) :
    snp(snp), numberOfIndividualsToInclude(originalSNPData->size()), originalSNPData(originalSNPData), originalGeneticModel(
        geneticModel), currentGeneticModel(geneticModel), currentRecode(ALL_RISK), alleleFrequencies(alleleFrequencies), numberOfAlleles(
        numberOfAlleles), originalRiskAllele(snp.getRiskAllele()), onlyFrequencies(false),
#ifdef CPU
        modifiedSNPData(
            new LapackppHostVector(new LaVectorDouble(numberOfIndividualsToInclude)))
#else
        modifiedSNPData(new PinnedHostVector(numberOfIndividualsToInclude))
#endif
{
  recodeAllRisk();
  doRecode();
}

SNPVector::SNPVector(SNP& snp, const std::vector<int>* numberOfAlleles, const std::vector<double>* alleleFrequencies,
    int numberOfIndividualsToInclude) :
    snp(snp), originalSNPData(nullptr), modifiedSNPData(nullptr), numberOfAlleles(numberOfAlleles), alleleFrequencies(
        alleleFrequencies), originalRiskAllele(snp.getRiskAllele()), originalGeneticModel(DOMINANT), currentGeneticModel(
        DOMINANT), currentRecode(ALL_RISK), onlyFrequencies(true), numberOfIndividualsToInclude(
        numberOfIndividualsToInclude) {

}

SNPVector::~SNPVector() {
  delete modifiedSNPData;
  delete originalSNPData;
  delete numberOfAlleles;
  delete alleleFrequencies;
}

const std::vector<int>& SNPVector::getOrginalData() const {
#ifdef DEBUG
  if(onlyFrequencies){
    throw InvalidState("SNPVector has only allele frequencies, can't get original data.");
  }
#endif
  return *originalSNPData;
}

const Container::HostVector& SNPVector::getRecodedData() const {
#ifdef DEBUG
  if(onlyFrequencies){
    throw InvalidState("SNPVector has only allele frequencies, can't get recoded data.");
  }
#endif
  return *modifiedSNPData;
}

int SNPVector::getNumberOfIndividualsToInclude() const {
  return numberOfIndividualsToInclude;
}

const SNP & SNPVector::getAssociatedSNP() const {
  return snp;
}

const std::vector<int>& SNPVector::getAlleleNumbers() const {
  return *numberOfAlleles;
}

const std::vector<double>& SNPVector::getAlleleFrequencies() const {
  return *alleleFrequencies;
}

void SNPVector::recode(Recode recode) {
#ifdef DEBUG
  if(onlyFrequencies){
    throw InvalidState("SNPVector has only allele frequencies, can't do recode.");
  }
#endif
  if(currentRecode == recode){
    return;
  }

  currentRecode = recode;
  if(recode == ALL_RISK){
    recodeAllRisk();
  }else if(recode == SNP_PROTECT){
    recodeSNPProtective();
  }else if(recode == INTERACTION_PROTECT){
    recodeInteractionProtective();
  }else{
    return;
  }

  doRecode();
}

void SNPVector::recodeAllRisk() {
  currentGeneticModel = originalGeneticModel;
  snp.setRiskAllele(originalRiskAllele);
}

void SNPVector::recodeSNPProtective() {
  currentGeneticModel = RECESSIVE;
  snp.setRiskAllele(invertRiskAllele(originalRiskAllele));
}

void SNPVector::recodeInteractionProtective() {
  currentGeneticModel = RECESSIVE;
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
#ifdef DEBUG
  if(onlyFrequencies){
    throw InvalidState("SNPVector has only allele frequencies, can't apply statistic model.");
  }
#endif

  if(statisticModel == ADDITIVE){
    for(int i = 0; i < numberOfIndividualsToInclude; ++i){
      if(interactionVector(i) != 0){
        (*modifiedSNPData)(i) = 0;
      }
    }
  }
  return;
}

std::ostream& operator<<(std::ostream& os, const Container::SNPVector& snpVector) {
  ///Print allele numbers
  os << (*(snpVector.numberOfAlleles))[ALLELE_ONE_CASE_POSITION] << ","
      << (*(snpVector.numberOfAlleles))[ALLELE_TWO_CASE_POSITION] << ","
      << (*(snpVector.numberOfAlleles))[ALLELE_ONE_CONTROL_POSITION] << ","
      << (*(snpVector.numberOfAlleles))[ALLELE_TWO_CONTROL_POSITION] << ","
      << (*(snpVector.numberOfAlleles))[ALLELE_ONE_ALL_POSITION] << ","
      << (*(snpVector.numberOfAlleles))[ALLELE_TWO_ALL_POSITION] << ",";

  //Print allele frequencies
  os << (*(snpVector.alleleFrequencies))[ALLELE_ONE_CASE_POSITION] << ","
      << (*(snpVector.alleleFrequencies))[ALLELE_TWO_CASE_POSITION] << ","
      << (*(snpVector.alleleFrequencies))[ALLELE_ONE_CONTROL_POSITION] << ","
      << (*(snpVector.alleleFrequencies))[ALLELE_TWO_CONTROL_POSITION] << ","
      << (*(snpVector.alleleFrequencies))[ALLELE_ONE_ALL_POSITION] << ","
      << (*(snpVector.alleleFrequencies))[ALLELE_TWO_ALL_POSITION] << ",";

  //Print recode
  os << snpVector.currentRecode;

  return os;
}

} /* namespace Container */
} /* namespace CuEira */
