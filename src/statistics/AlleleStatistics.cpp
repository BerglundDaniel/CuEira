#include "AlleleStatistics.h"

namespace CuEira {

AlleleStatistics::AlleleStatistics(const double minorAlleleFrequency, const std::vector<int>* numberOfAlleles,
    const std::vector<double>* alleleFrequencies) :
    minorAlleleFrequency(minorAlleleFrequency), numberOfAlleles(numberOfAlleles), alleleFrequencies(alleleFrequencies) {

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

double AlleleStatistics::getMinorAlleleFrequecy() const {
  return minorAlleleFrequency;
}

void AlleleStatistics::toOstream(std::ostream& os) const {
  ///Print allele numbers
  os << (*(numberOfAlleles))[ALLELE_ONE_CASE_POSITION] << "," << (*(numberOfAlleles))[ALLELE_TWO_CASE_POSITION] << ","
      << (*(numberOfAlleles))[ALLELE_ONE_CONTROL_POSITION] << "," << (*(numberOfAlleles))[ALLELE_TWO_CONTROL_POSITION]
      << "," << (*(numberOfAlleles))[ALLELE_ONE_ALL_POSITION] << "," << (*(numberOfAlleles))[ALLELE_TWO_ALL_POSITION]
      << ",";

  //Print allele frequencies
  os << (*(alleleFrequencies))[ALLELE_ONE_CASE_POSITION] << "," << (*(alleleFrequencies))[ALLELE_TWO_CASE_POSITION]
      << "," << (*(alleleFrequencies))[ALLELE_ONE_CONTROL_POSITION] << ","
      << (*(alleleFrequencies))[ALLELE_TWO_CONTROL_POSITION] << "," << (*(alleleFrequencies))[ALLELE_ONE_ALL_POSITION]
      << "," << (*(alleleFrequencies))[ALLELE_TWO_ALL_POSITION];
}

std::ostream& operator<<(std::ostream& os, const AlleleStatistics& alleleStatistics) {
  alleleStatistics.toOstream(os);
  return os;
}

} /* namespace CuEira */
