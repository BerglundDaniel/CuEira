#include "EnvironmentFactor.h"

namespace CuEira {

EnvironmentFactor::EnvironmentFactor(Id id) :
    id(id), variableType(OTHER) {

}

EnvironmentFactor::~EnvironmentFactor() {

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
