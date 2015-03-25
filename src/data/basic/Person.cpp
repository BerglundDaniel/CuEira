#include "Person.h"

namespace CuEira {

Person::Person(Id id, Sex sex, Phenotype phenotype, bool include) :
    id(id), sex(sex), phenotype(phenotype), include(include) {
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

bool Person::operator<(const Person& otherPerson) const {
  return id < otherPerson.getId();
}

bool Person::operator==(const Person& otherPerson) const {
  return id == otherPerson.getId();
}

} /* namespace CuEira */
