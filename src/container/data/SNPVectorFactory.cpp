#include "SNPVectorFactory.h"

namespace CuEira {
namespace Container {

SNPVectorFactory::SNPVectorFactory(const Configuration& configuration, int numberOfIndividualsToInclude) :
    configuration(configuration), numberOfIndividualsToInclude(numberOfIndividualsToInclude), geneticModel(
        configuration.getGeneticModel()), minorAlleleFrequencyThreshold(
        configuration.getMinorAlleleFrequencyThreshold()) {

}

SNPVectorFactory::~SNPVectorFactory() {

}

SNPVector* SNPVectorFactory::constructSNPVector(SNP& snp, std::vector<int>* originalSNPData,
    std::vector<int>* numberOfAlleles, bool missingData) const {

  std::vector<double>* alleleFrequencies = convertAlleleNumbersToFrequencies(*numberOfAlleles);
  setSNPRiskAllele(snp, *alleleFrequencies);
  setSNPInclude(snp, *alleleFrequencies, *numberOfAlleles, missingData);

  if(snp.getInclude()){
    return new SNPVector(snp, geneticModel, originalSNPData, numberOfAlleles, alleleFrequencies);
  }else{
    delete originalSNPData;
    return new SNPVector(snp, numberOfAlleles, alleleFrequencies, numberOfIndividualsToInclude);
  }
}

std::vector<double>* SNPVectorFactory::convertAlleleNumbersToFrequencies(
    const std::vector<int>& numberOfAlleles) const {
  const int numberOfAllelesInPopulation = numberOfAlleles[ALLELE_ONE_ALL_POSITION]
      + numberOfAlleles[ALLELE_TWO_ALL_POSITION];
  const int numberOfAllelesInCase = numberOfAlleles[ALLELE_ONE_CASE_POSITION]
      + numberOfAlleles[ALLELE_TWO_CASE_POSITION];
  const int numberOfAllelesInControl = numberOfAlleles[ALLELE_ONE_CONTROL_POSITION]
      + numberOfAlleles[ALLELE_TWO_CONTROL_POSITION];

  std::vector<double>* alleleFrequencies = new std::vector<double>(6);

  (*alleleFrequencies)[ALLELE_ONE_CASE_POSITION] = (double) numberOfAlleles[ALLELE_ONE_CASE_POSITION]
      / numberOfAllelesInCase;
  (*alleleFrequencies)[ALLELE_TWO_CASE_POSITION] = (double) numberOfAlleles[ALLELE_TWO_CASE_POSITION]
      / numberOfAllelesInCase;
  (*alleleFrequencies)[ALLELE_ONE_CONTROL_POSITION] = (double) numberOfAlleles[ALLELE_ONE_CONTROL_POSITION]
      / numberOfAllelesInControl;
  (*alleleFrequencies)[ALLELE_TWO_CONTROL_POSITION] = (double) numberOfAlleles[ALLELE_TWO_CONTROL_POSITION]
      / numberOfAllelesInControl;
  (*alleleFrequencies)[ALLELE_ONE_ALL_POSITION] = (double) numberOfAlleles[ALLELE_ONE_ALL_POSITION]
      / numberOfAllelesInPopulation;
  (*alleleFrequencies)[ALLELE_TWO_ALL_POSITION] = (double) numberOfAlleles[ALLELE_TWO_ALL_POSITION]
      / numberOfAllelesInPopulation;

  return alleleFrequencies;
}

void SNPVectorFactory::setSNPRiskAllele(SNP& snp, const std::vector<double>& alleleFrequencies) const {
  //Check which allele is most frequent in cases
  RiskAllele riskAllele;
  if(alleleFrequencies[ALLELE_ONE_CASE_POSITION] == alleleFrequencies[ALLELE_TWO_CASE_POSITION]){
    if(alleleFrequencies[ALLELE_ONE_CONTROL_POSITION] == alleleFrequencies[ALLELE_TWO_CONTROL_POSITION]){
      riskAllele = ALLELE_ONE;
    }else if(alleleFrequencies[ALLELE_ONE_CONTROL_POSITION] < alleleFrequencies[ALLELE_TWO_CONTROL_POSITION]){
      riskAllele = ALLELE_ONE;
    }else{
      riskAllele = ALLELE_TWO;
    }
  }else if(alleleFrequencies[ALLELE_ONE_CASE_POSITION] > alleleFrequencies[ALLELE_TWO_CASE_POSITION]){
    if(alleleFrequencies[ALLELE_ONE_CASE_POSITION] >= alleleFrequencies[ALLELE_ONE_CONTROL_POSITION]){
      riskAllele = ALLELE_ONE;
    }else{
      riskAllele = ALLELE_TWO;
    }
  }else{
    if(alleleFrequencies[ALLELE_TWO_CASE_POSITION] >= alleleFrequencies[ALLELE_TWO_CONTROL_POSITION]){
      riskAllele = ALLELE_TWO;
    }else{
      riskAllele = ALLELE_ONE;
    }
  }

  snp.setRiskAllele(riskAllele);
}

void SNPVectorFactory::setSNPInclude(SNP& snp, const std::vector<double>& alleleFrequencies,
    const std::vector<int>& numberOfAlleles, bool missingData) const {
  if(missingData){
    snp.setInclude(false);
    return;
  }

  //Check if the number of alleles is to low
  if(numberOfAlleles[ALLELE_ONE_CASE_POSITION] <= ABSOLUTE_FREQUENCY_THRESHOLD
      || numberOfAlleles[ALLELE_TWO_CASE_POSITION] <= ABSOLUTE_FREQUENCY_THRESHOLD
      || numberOfAlleles[ALLELE_ONE_CONTROL_POSITION] <= ABSOLUTE_FREQUENCY_THRESHOLD
      || numberOfAlleles[ALLELE_TWO_CONTROL_POSITION] <= ABSOLUTE_FREQUENCY_THRESHOLD){
    snp.setInclude(false);
    return;
  }

  //Calculate MAF
  double minorAlleleFrequency;
  if(alleleFrequencies[ALLELE_ONE_ALL_POSITION] == alleleFrequencies[ALLELE_TWO_ALL_POSITION]){
    minorAlleleFrequency = alleleFrequencies[ALLELE_ONE_ALL_POSITION];
  }else if(alleleFrequencies[ALLELE_ONE_ALL_POSITION] > alleleFrequencies[ALLELE_TWO_ALL_POSITION]){
    minorAlleleFrequency = alleleFrequencies[ALLELE_TWO_ALL_POSITION];
  }else{
    minorAlleleFrequency = alleleFrequencies[ALLELE_ONE_ALL_POSITION];
  }

  if(minorAlleleFrequency < minorAlleleFrequencyThreshold){
    snp.setInclude(false);
    return;
  }
}

} /* namespace Container */
} /* namespace CuEira */
