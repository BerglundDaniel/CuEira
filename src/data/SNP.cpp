#include "SNP.h"

namespace CuEira {

SNP::SNP(Id id, bool include) :
    id(id), include(include) {

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

} /* namespace CuEira */
