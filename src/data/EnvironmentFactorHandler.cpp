#include "EnvironmentFactorHandler.h"

namespace CuEira {

EnvironmentFactorHandler::EnvironmentFactorHandler(const EnvironmentFactor* environmentFactor,
    int numberOfIndividualsTotal) :
    environmentFactor(environmentFactor), numberOfIndividualsTotal(dnumberOfIndividualsTotal) {

}

EnvironmentFactorHandler::~EnvironmentFactorHandler() {
  delete environmentFactor;
}

int EnvironmentFactorHandler::getNumberOfIndividualsTotal() const {
  return numberOfIndividualsTotal;
}

const EnvironmentFactor& EnvironmentFactorHandler::getEnvironmentFactor() const {
  return environmentFactor;
}

} /* namespace CuEira */
