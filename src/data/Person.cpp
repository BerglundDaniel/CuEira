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
  return id < otherPerson.getId();
}

bool Person::operator==(const Person& otherPerson) const {
  return id == otherPerson.getId();
}

} /* namespace CuEira */
