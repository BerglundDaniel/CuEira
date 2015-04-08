#include "AlleleStatisticsFactory.h"

namespace CuEira {

AlleleStatisticsFactory::AlleleStatisticsFactory() {

}

AlleleStatisticsFactory::~AlleleStatisticsFactory() {

}

AlleleStatistics* AlleleStatisticsFactory::constructAlleleStatistics(const std::vector<int>* numberOfAlleles) const {
  //TODO have snp data and phenotype data
  //calculate numberOfAlleles from it
  //separate cpu and gpu version

  return new AlleleStatistics(numberOfAlleles, convertAlleleNumbersToFrequencies(*numberOfAlleles));
}

std::vector<double>* AlleleStatisticsFactory::convertAlleleNumbersToFrequencies(
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

} /* namespace CuEira */
