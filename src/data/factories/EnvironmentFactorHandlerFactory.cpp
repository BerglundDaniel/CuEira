#include "EnvironmentFactorHandlerFactory.h"

namespace CuEira {

EnvironmentFactorHandlerFactory::EnvironmentFactorHandlerFactory() {

}

EnvironmentFactorHandlerFactory::~EnvironmentFactorHandlerFactory() {

}

EnvironmentFactorHandler* EnvironmentFactorHandlerFactory::constructEnvironmentFactorHandler(
    const Container::HostMatrix* dataMatrix, const std::vector<EnvironmentFactor*>* environmentFactors) const {

}

} /* namespace CuEira */
