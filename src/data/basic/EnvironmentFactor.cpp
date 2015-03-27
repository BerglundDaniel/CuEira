#include "EnvironmentFactor.h"

namespace CuEira {

EnvironmentFactor::EnvironmentFactor(Id id) :
    id(id), variableType(OTHER) {

}

EnvironmentFactor::~EnvironmentFactor() {
  std::cerr << "EnvironmentFactor destructor " << id << std::endl; //TODO tmp
}

Id EnvironmentFactor::getId() const {
  return id;
}

void EnvironmentFactor::setVariableType(VariableType variableType) {
  this->variableType = variableType;
}

VariableType EnvironmentFactor::getVariableType() const {
  return variableType;
}

void EnvironmentFactor::setMax(int max) {
  this->max = max;
}

void EnvironmentFactor::setMin(int min) {
  this->min = min;
}

int EnvironmentFactor::getMax() const {
  return max;
}

int EnvironmentFactor::getMin() const {
  return min;
}

bool EnvironmentFactor::operator<(const EnvironmentFactor& otherEnvironmentFactor) const {
  return id < otherEnvironmentFactor.getId();
}

bool EnvironmentFactor::operator==(const EnvironmentFactor& otherEnvironmentFactor) const {
  return id == otherEnvironmentFactor.getId();
}

std::ostream & operator<<(std::ostream& os, const EnvironmentFactor& envFactor) {
  os << envFactor.id.getString();

  return os;
}

} /* namespace CuEira */
