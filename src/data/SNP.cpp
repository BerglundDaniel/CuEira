#include "SNP.h"

namespace CuEira {

SNP::SNP(Id id, std::string alleleOneName, std::string alleleTwoName, unsigned int position, bool include) :
    id(id), alleleOneName(alleleOneName), alleleTwoName(alleleTwoName), include(include), position(position), minorAlleleFrequency(
        1), minorAlleleFrequencyHasBeenSet(false), riskAllele(ALLELE_ONE), riskAlleleHasBeenSet(false), caseAlleleHasBeenSet(
        false), controlAlleleHasBeenSet(false), allAlleleHasBeenSet(false), alleleOneCaseFrequency(-1), alleleTwoCaseFrequency(
        -1), alleleOneControlFrequency(-1), alleleTwoControlFrequency(-1), alleleOneAllFrequency(-1), alleleTwoAllFrequency(
        -1) {

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
    throw InvalidState(tmp.c_str());
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
    throw InvalidState(tmp.c_str());
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

void SNP::setCaseAlleleFrequencies(double alleleOneCaseFrequency, double alleleTwoCaseFrequency) {
  caseAlleleHasBeenSet = true;
  this->alleleOneCaseFrequency = alleleOneCaseFrequency;
  this->alleleTwoCaseFrequency = alleleTwoCaseFrequency;
}

void SNP::setControlAlleleFrequencies(double alleleOneControlFrequency, double alleleTwoControlFrequency) {
  controlAlleleHasBeenSet = true;
  this->alleleOneControlFrequency = alleleOneControlFrequency;
  this->alleleTwoControlFrequency = alleleTwoControlFrequency;
}

void SNP::setAllAlleleFrequencies(double alleleOneAllFrequency, double alleleTwoAllFrequency) {
  allAlleleHasBeenSet = true;
  this->alleleOneAllFrequency = alleleOneAllFrequency;
  this->alleleTwoAllFrequency = alleleTwoAllFrequency;
}

double SNP::getAlleleOneCaseFrequency() const {
  if(!caseAlleleHasBeenSet){
    std::ostringstream os;
    os << "Can't get case allele one frequency since it has not been set for SNP " << id.getString() << std::endl;
    const std::string& tmp = os.str();
    throw InvalidState(tmp.c_str());
  }

  return alleleOneCaseFrequency;
}

double SNP::getAlleleTwoCaseFrequency() const {
  if(!caseAlleleHasBeenSet){
    std::ostringstream os;
    os << "Can't get case allele two frequency since it has not been set for SNP " << id.getString() << std::endl;
    const std::string& tmp = os.str();
    throw InvalidState(tmp.c_str());
  }

  return alleleTwoCaseFrequency;
}

double SNP::getAlleleOneControlFrequency() const {
  if(!controlAlleleHasBeenSet){
    std::ostringstream os;
    os << "Can't get control allele one frequency since it has not been set for SNP " << id.getString() << std::endl;
    const std::string& tmp = os.str();
    throw InvalidState(tmp.c_str());
  }

  return alleleOneControlFrequency;
}

double SNP::getAlleleTwoControlFrequency() const {
  if(!controlAlleleHasBeenSet){
    std::ostringstream os;
    os << "Can't get control allele two frequency since it has not been set for SNP " << id.getString() << std::endl;
    const std::string& tmp = os.str();
    throw InvalidState(tmp.c_str());
  }

  return alleleTwoControlFrequency;
}

double SNP::getAlleleOneAllFrequency() const {
  if(!allAlleleHasBeenSet){
    std::ostringstream os;
    os << "Can't get whole population allele one frequency since it has not been set for SNP " << id.getString()
        << std::endl;
    const std::string& tmp = os.str();
    throw InvalidState(tmp.c_str());
  }

  return alleleOneAllFrequency;
}

double SNP::getAlleleTwoAllFrequency() const {
  if(!allAlleleHasBeenSet){
    std::ostringstream os;
    os << "Can't get whole population allele two frequency since it has not been set for SNP " << id.getString()
        << std::endl;
    const std::string& tmp = os.str();
    throw InvalidState(tmp.c_str());
  }

  return alleleTwoAllFrequency;
}

bool SNP::operator<(const SNP& otherSNP) const {
  return id < otherSNP.getId();
#ifdef DEBUG
  if(this == &otherSNP){
    std::cerr << "Something is very wrong with the SNPs." << std::endl;
  }
#endif
}

bool SNP::operator==(const SNP& otherSNP) const {
  return id == otherSNP.getId();
#ifdef DEBUG
  if(this == &otherSNP){
    std::cerr << "Something is very wrong with the SNPs." << std::endl;
  }
#endif
}

} /* namespace CuEira */
