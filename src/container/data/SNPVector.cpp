#include "SNPVector.h"

namespace CuEira {
namespace Container {

template<typename Vector>
SNPVector<Vector>::SNPVector(SNP& snp, GeneticModel geneticModel, const Vector* snpOrgExMissing,
    const std::set<int>* snpMissingData) :
    snp(snp), numberOfIndividualsToInclude(snpOrgExMissing->getNumberOfRows()), snpOrgExMissing(snpOrgExMissing), originalGeneticModel(
        geneticModel), currentGeneticModel(geneticModel), currentRecode(ALL_RISK), originalRiskAllele(
        snp.getRiskAllele()), initialised(false), noMissing(false), snpRecodedExMissing(
        new Vector(numberOfIndividualsToInclude)), snpMissingData(snpMissingData) {

}

template<typename Vector>
SNPVector<Vector>::~SNPVector() {
  delete snpOrgExMissing;
  delete snpRecodedExMissing;
  delete snpMissingData;
}

template<typename Vector>
int SNPVector<Vector>::getNumberOfIndividualsToInclude() const {
  return numberOfIndividualsToInclude;
}

template<typename Vector>
const SNP & SNPVector<Vector>::getAssociatedSNP() const {
  return snp;
}

template<typename Vector>
const Vector& SNPVector<Vector>::getSNPData() const {
#ifdef DEBUG
  if(!initialised){
    throw new InvalidState("SNPVector not initialised.");
  }
#endif

  return *snpRecodedExMissing;
}

template<typename Vector>
Vector& SNPVector<Vector>::getSNPData() {
#ifdef DEBUG
  if(!initialised){
    throw new InvalidState("SNPVector not initialised.");
  }
#endif

  return *snpRecodedExMissing;
}

template<typename Vector>
bool SNPVector<Vector>::hasMissing() const {
  return !noMissing;
}

template<typename Vector>
const std::set<int>& SNPVector<Vector>::getMissing() const {
  return *snpMissingData;
}

template<typename Vector>
void SNPVector<Vector>::recode(Recode recode) {
#ifdef DEBUG
  initialised = true;
#endif

  currentRecode = recode;
  if(recode == ENVIRONMENT_PROTECT || recode == INTERACTION_PROTECT){
    currentGeneticModel = RECESSIVE;
    snp.setRiskAllele(snp.getProtectiveAllele());
  }else{
    currentGeneticModel = originalGeneticModel;
    snp.setRiskAllele(originalRiskAllele);
  }

  RiskAllele currentRiskAllele = snp.getRiskAllele();
  int snpToRisk[3] = {};

  if(currentGeneticModel == DOMINANT){
    if(currentRiskAllele == ALLELE_ONE){
      snpToRisk[0] = 1;
      snpToRisk[1] = 1;
      snpToRisk[2] = 0;
    }else if(currentRiskAllele == ALLELE_TWO){
      snpToRisk[0] = 0;
      snpToRisk[1] = 1;
      snpToRisk[2] = 1;
    }else{
      throw InvalidState("Unknown RiskAllele in SNPVector");
    }
  }else if(currentGeneticModel == RECESSIVE){
    if(currentRiskAllele == ALLELE_ONE){
      snpToRisk[0] = 1;
      snpToRisk[1] = 0;
      snpToRisk[2] = 0;
    }else if(currentRiskAllele == ALLELE_TWO){
      snpToRisk[0] = 0;
      snpToRisk[1] = 0;
      snpToRisk[2] = 1;
    }else{
      throw InvalidState("Unknown RiskAllele in SNPVector");
    }
  }else{
    throw InvalidState("Unknown genetic model in SNPVector");
  }

  doRecode(snpToRisk);
}

} /* namespace Container */
} /* namespace CuEira */
