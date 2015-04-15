#include "AlleleStatisticsFactory.h"

namespace CuEira {

template<typename Vector>
AlleleStatisticsFactory<Vector>::AlleleStatisticsFactory() {

}

template<typename Vector>
AlleleStatisticsFactory<Vector>::~AlleleStatisticsFactory() {

}

template<typename Vector>
AlleleStatistics* AlleleStatisticsFactory<Vector>::constructAlleleStatistics(
    const Container::SNPVector<Vector>& snpVector, const Container::PhenotypeVector<Vector>& phenotypeVector) const {

  //First 3 is control, second 3 is case. 0, 1, 2 genotype per group
  std::vector<int>* numberOfAllelesPerGenotype = getNumberOfAllelesPerGenotype();
  std::vector<int>* numberOfAlleles = new std::vector<int>(6);

  (*numberOfAlleles)[ALLELE_ONE_CASE_POSITION] = (*numberOfAllelesPerGenotype)[4]
      + 2 * (*numberOfAllelesPerGenotype)[3];
  (*numberOfAlleles)[ALLELE_TWO_CASE_POSITION] = (*numberOfAllelesPerGenotype)[4]
      + 2 * (*numberOfAllelesPerGenotype)[5];
  (*numberOfAlleles)[ALLELE_ONE_CONTROL_POSITION] = (*numberOfAllelesPerGenotype)[1]
      + 2 * (*numberOfAllelesPerGenotype)[0];
  (*numberOfAlleles)[ALLELE_TWO_CONTROL_POSITION] = (*numberOfAllelesPerGenotype)[1]
      + 2 * (*numberOfAllelesPerGenotype)[2];
  (*numberOfAlleles)[ALLELE_ONE_ALL_POSITION] = (*numberOfAlleles)[ALLELE_ONE_CASE_POSITION]
      + (*numberOfAlleles)[ALLELE_ONE_CONTROL_POSITION];
  (*numberOfAlleles)[ALLELE_TWO_ALL_POSITION] = (*numberOfAlleles)[ALLELE_TWO_CASE_POSITION]
      + (*numberOfAlleles)[ALLELE_TWO_CONTROL_POSITION];

  delete numberOfAllelesPerGenotype;

  std::vector<double>* alleleFrequencies = convertAlleleNumbersToFrequencies(*numberOfAlleles);

  double minorAlleleFrequency;
  if((*alleleFrequencies)[ALLELE_ONE_ALL_POSITION] == (*alleleFrequencies)[ALLELE_TWO_ALL_POSITION]){
    minorAlleleFrequency = (*alleleFrequencies)[ALLELE_ONE_ALL_POSITION];
  }else if((*alleleFrequencies)[ALLELE_ONE_ALL_POSITION] > (*alleleFrequencies)[ALLELE_TWO_ALL_POSITION]){
    minorAlleleFrequency = (*alleleFrequencies)[ALLELE_TWO_ALL_POSITION];
  }else{
    minorAlleleFrequency = (*alleleFrequencies)[ALLELE_ONE_ALL_POSITION];
  }

  return new AlleleStatistics(minorAlleleFrequency, numberOfAlleles, alleleFrequencies);
}

template<typename Vector>
std::vector<double>* AlleleStatisticsFactory<Vector>::convertAlleleNumbersToFrequencies(
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
