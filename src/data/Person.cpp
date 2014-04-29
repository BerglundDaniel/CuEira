#include "Person.h"

namespace CuEira {

Person::Person(Id id, Sex sex, Phenotype phenotype) :
    id(id), sex(sex), phenotype(phenotype), include(shouldPersonBeIncluded()) {
}

Person::~Person() {

}

Id Person::getId() {
  return id;
}

Sex Person::getSex() {
  return sex;
}

Phenotype Person::getPhenotype() {
  return phenotype;
}

void Person::setInclude(bool include) {
  this->include = include;
}

bool Person::getInclude() {
  return include;
}

bool Person::shouldPersonBeIncluded() {
  if(phenotype() == MISSING){
    return false;
  }else{
    return true;
  }
}

} /* namespace CuEira */
