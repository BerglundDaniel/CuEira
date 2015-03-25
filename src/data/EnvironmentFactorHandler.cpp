#include "EnvironmentFactorHandler.h"

namespace CuEira {

EnvironmentFactorHandler::EnvironmentFactorHandler(const std::vector<const EnvironmentFactor*>* environmentFactors,
    const std::vector<std::set<int>>* personsToSkip, int numberOfIndividualsTotal) :
    environmentFactors(environmentFactors), numberOfEnvironmentFactors(environmentFactors->size()), numberOfIndividualsTotal(
        dnumberOfIndividualsTotal), personsToSkip(personsToSkip) {

}

EnvironmentFactorHandler::~EnvironmentFactorHandler() {
  environmentFactors->clear();
  delete environmentFactors;
  delete personsToSkip;
}

int EnvironmentFactorHandler::getNumberOfIndividualsTotal() const {
  return numberOfIndividualsTotal;
}

int EnvironmentFactorHandler::getNumberOfEnvironmentFactors() const {
  return numberOfEnvironmentFactors;
}

const std::vector<const EnvironmentFactor*>& EnvironmentFactorHandler::getHeaders() const {
  return environmentFactors;
}

} /* namespace CuEira */
