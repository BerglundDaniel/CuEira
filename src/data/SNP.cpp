#include "SNP.h"

namespace CuEira {

SNP::SNP(Id id, std::string alleleOneName, std::string alleleTwoName, unsigned int position,
    SNPIncludeExclude includeExclude) :
    id(id), alleleOneName(alleleOneName), alleleTwoName(alleleTwoName), includeExcludeVector(
        new std::vector<SNPIncludeExclude>()), position(position), riskAllele(ALLELE_ONE), riskAlleleHasBeenSet(false) {
  includeExcludeVector->push_back(includeExclude);
}

SNP::~SNP() {
  delete includeExcludeVector;
}

Id SNP::getId() const {
  return id;
}

bool SNP::shouldInclude() const {
  if((*includeExcludeVector)[0] == INCLUDE){
    return true;
  }else{
    return false;
  }
}

const std::vector<SNPIncludeExclude>& SNP::getInclude() const {
  return *includeExcludeVector;
}

void SNP::setInclude(SNPIncludeExclude includeExclude) {
  if(includeExclude == INCLUDE){
    delete includeExcludeVector;
    includeExcludeVector = new std::vector<SNPIncludeExclude>();
  }
  includeExcludeVector->push_back(includeExclude);
}

void SNP::setRiskAllele(RiskAllele riskAllele) {
  riskAlleleHasBeenSet = true;
  this->riskAllele = riskAllele;
}

RiskAllele SNP::getRiskAllele() const {
  if(!riskAlleleHasBeenSet){
    std::ostringstream os;
    os << "Can't get risk allele since it has not been set for SNP " << id.getString() << std::endl;
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

std::ostream & operator<<(std::ostream& os, const SNP& snp) {
  //Print the id
  os << snp.id.getString() << ",";

  //Print the risk allele name
  if(snp.riskAllele == ALLELE_ONE){
    os << snp.alleleOneName << ",";
  }else{
    os << snp.alleleTwoName << ",";
  }

  //Print the allele names
  os << snp.alleleOneName << "," << snp.alleleTwoName;

  return os;
}

} /* namespace CuEira */
