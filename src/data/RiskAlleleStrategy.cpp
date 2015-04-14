#include "RiskAlleleStrategy.h"

namespace CuEira {

RiskAlleleStrategy::RiskAlleleStrategy() {

}

RiskAlleleStrategy::~RiskAlleleStrategy() {

}

RiskAllele RiskAlleleStrategy::calculateRiskAllele(const AlleleStatistics& alleleStatistics) const {
  const std::vector<double>& alleleFrequencies = alleleStatistics.getAlleleFrequencies();
  RiskAllele riskAllele;

  //Check which allele is most frequent in cases
  if((alleleFrequencies[ALLELE_ONE_CASE_POSITION] - alleleFrequencies[ALLELE_ONE_CONTROL_POSITION]) > 0){
    riskAllele = ALLELE_ONE;
  }else{
    riskAllele = ALLELE_TWO;
  }

  return riskAllele;
}

} /* namespace CuEira */
