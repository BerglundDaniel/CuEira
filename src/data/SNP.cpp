#include "SNP.h"

namespace CuEira {

SNP::SNP(Id id, std::string alleleOneName, std::string alleleTwoName, bool include) :
    id(id), alleleOneName(alleleOneName), alleleTwoName(alleleTwoName), include(include), minorAlleleFrequency(1), minorAlleleFrequencyHasBeenSet(
        false) {

}

SNP::~SNP() {

}

Id SNP::getId() {
  return id;
}

bool SNP::getInclude() {
  return include;
}

void SNP::setInclude(bool include) {
  this->include = include;
}

void SNP::setMinorAlleleFrequency(unsigned double maf) {
  minorAlleleFrequencyHasBeenSet = true;
  minorAlleleFrequency = maf;
}

unsigned double SNP::getMinorAlleleFrequency() const {
  if(!minorAlleleFrequencyHasBeenSet){
    std::ostringstream os;
    os << "Can't get minor allele frequency since it has not been set for SNP " << id.getString() << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str()); //FIXME
  }
  return minorAlleleFrequency;
}

bool SNP::hasMinorAlleleFrequencyBeenSet() {
  return minorAlleleFrequencyHasBeenSet;
}

} /* namespace CuEira */
