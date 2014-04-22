#include "Person.h"

namespace CuEira {

Person::Person(Id id, Sex sex, Phenotype phenotype, int rowNumber) :
    id(id), sex(sex), phenotype(phenotype), rowNumber(rowNumber) {

}

Person::~Person() {

}

Id Person::getId() {
  return id;
}

Sex Person::getSex() {
  return sex;
}

int Person::getRowNumber() {
  return rowNumber;
}

Phenotype Person::getPhenotype() {
  return phenotype;
}

} /* namespace CuEira */
