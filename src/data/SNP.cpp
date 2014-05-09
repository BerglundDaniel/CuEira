#include "SNP.h"

namespace CuEira {

SNP::SNP(Id id, std::string alleleOneName, std::string alleleTwoName, unsigned int position, bool include) :
    id(id), alleleOneName(alleleOneName), alleleTwoName(alleleTwoName), include(include), position(position), minorAlleleFrequency(
        1), minorAlleleFrequencyHasBeenSet(false), riskAllele(ALLELE_ONE), riskAlleleHasBeenSet(false) {

}

SNP::~SNP() {

}

Id SNP::getId() const {
  return id;
}

bool SNP::getInclude() const {
  return include;
}

void SNP::setInclude(bool include) {
  this->include = include;
}

void SNP::setMinorAlleleFrequency(double maf) {
  minorAlleleFrequencyHasBeenSet = true;
  minorAlleleFrequency = maf;
}

double SNP::getMinorAlleleFrequency() const {
  if(!minorAlleleFrequencyHasBeenSet){
    std::ostringstream os;
    os << "Can't get minor allele frequency since it has not been set for SNP " << id.getString() << std::endl;
    const std::string& tmp = os.str();
    throw std::invalid_argument(tmp.c_str()); //FIXME
  }
  return minorAlleleFrequency;
}

void SNP::setRiskAllele(RiskAllele riskAllele) {
  riskAlleleHasBeenSet = true;
  this->riskAllele = riskAllele;
}

RiskAllele SNP::getRiskAllele() const {
  if(!riskAlleleHasBeenSet){
    std::ostringstream os;
    os << "Can't get minor allele frequency since it has not been set for SNP " << id.getString() << std::endl;
    const std::string& tmp = os.str();
    throw std::invalid_argument(tmp.c_str()); //FIXME
  }
  return riskAllele;
}

std::string SNP::getAlleleOneName() const {
  return alleleOneName;
}

std::string SNP::getAlleleTwoName() const {
  return alleleTwoName;
}

unsigned int SNP::getPosition() const {
  return position;
}

} /* namespace CuEira */
