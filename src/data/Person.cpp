#include "Person.h"

namespace CuEira {

Person::Person(Id id, Sex sex, int rowNumber) :
    id(id), sex(sex), rowNumber(rowNumber) {

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

} /* namespace CuEira */
