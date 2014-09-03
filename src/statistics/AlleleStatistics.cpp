#include "AlleleStatistics.h"

namespace CuEira {

AlleleStatistics::AlleleStatistics(const std::vector<int>* numberOfAlleles,
    const std::vector<double>* alleleFrequencies) :
    numberOfAlleles(numberOfAlleles), alleleFrequencies(alleleFrequencies) {

}

AlleleStatistics::~AlleleStatistics() {
  delete numberOfAlleles;
  delete alleleFrequencies;
}

const std::vector<int>& AlleleStatistics::getAlleleNumbers() const {
  return *numberOfAlleles;
}

const std::vector<double>& AlleleStatistics::getAlleleFrequencies() const {
  return *alleleFrequencies;
}

std::ostream& operator<<(std::ostream& os, const AlleleStatistics& alleleStatistics) {
  ///Print allele numbers
  os << (*(alleleStatistics.numberOfAlleles))[ALLELE_ONE_CASE_POSITION] << ","
      << (*(alleleStatistics.numberOfAlleles))[ALLELE_TWO_CASE_POSITION] << ","
      << (*(alleleStatistics.numberOfAlleles))[ALLELE_ONE_CONTROL_POSITION] << ","
      << (*(alleleStatistics.numberOfAlleles))[ALLELE_TWO_CONTROL_POSITION] << ","
      << (*(alleleStatistics.numberOfAlleles))[ALLELE_ONE_ALL_POSITION] << ","
      << (*(alleleStatistics.numberOfAlleles))[ALLELE_TWO_ALL_POSITION] << ",";

  //Print allele frequencies
  os << (*(alleleStatistics.alleleFrequencies))[ALLELE_ONE_CASE_POSITION] << ","
      << (*(alleleStatistics.alleleFrequencies))[ALLELE_TWO_CASE_POSITION] << ","
      << (*(alleleStatistics.alleleFrequencies))[ALLELE_ONE_CONTROL_POSITION] << ","
      << (*(alleleStatistics.alleleFrequencies))[ALLELE_TWO_CONTROL_POSITION] << ","
      << (*(alleleStatistics.alleleFrequencies))[ALLELE_ONE_ALL_POSITION] << ","
      << (*(alleleStatistics.alleleFrequencies))[ALLELE_TWO_ALL_POSITION];

  return os;
}

} /* namespace CuEira */
