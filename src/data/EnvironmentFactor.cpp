#include "EnvironmentFactor.h"

namespace CuEira {

EnvironmentFactor::EnvironmentFactor(Id id, bool include) :
    id(id), include(include) {

}

EnvironmentFactor::~EnvironmentFactor() {

}

Id EnvironmentFactor::getId() {
  return id;
}

bool EnvironmentFactor::getInclude() {
  return include;
}

void EnvironmentFactor::setInclude(bool include) {
  this->include = include;
}

} /* namespace CuEira */
