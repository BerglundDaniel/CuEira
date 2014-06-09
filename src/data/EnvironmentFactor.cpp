#include "EnvironmentFactor.h"

namespace CuEira {

EnvironmentFactor::EnvironmentFactor(Id id) :
    id(id), include(shouldEnvironmentFactorBeIncluded()), variableType(OTHER) {

}

EnvironmentFactor::~EnvironmentFactor() {

}

Id EnvironmentFactor::getId() const {
  return id;
}

bool EnvironmentFactor::getInclude() const {
  return include;
}

bool EnvironmentFactor::shouldEnvironmentFactorBeIncluded() const {
  return true;
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

} /* namespace CuEira */
