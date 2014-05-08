#include "EnvironmentFactor.h"

namespace CuEira {

EnvironmentFactor::EnvironmentFactor(Id id) :
    id(id), include(shouldEnvironmentFactorBeIncluded()) {

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

} /* namespace CuEira */
