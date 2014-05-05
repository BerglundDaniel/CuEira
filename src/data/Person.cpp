#include "Person.h"

namespace CuEira {

Person::Person(Id id, Sex sex, Phenotype phenotype) :
    id(id), sex(sex), phenotype(phenotype), include(shouldPersonBeIncluded()) {
}

Person::~Person() {

}

Id Person::getId() const {
  return id;
}

Sex Person::getSex() const {
  return sex;
}

Phenotype Person::getPhenotype() const {
  return phenotype;
}

void Person::setInclude(bool include) {
  this->include = include;
}

bool Person::getInclude() const {
  return include;
}

bool Person::shouldPersonBeIncluded() const {
  if(phenotype == MISSING){
    return false;
  }else{
    return true;
  }
}

bool Person::operator<(const Person& otherPerson) const {
  if(id < otherPerson.getId()){
    return true;
  }else{
    return false;
  }
}

} /* namespace CuEira */
