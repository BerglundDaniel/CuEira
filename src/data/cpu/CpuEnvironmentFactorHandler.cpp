#include "CpuEnvironmentFactorHandler.h"

namespace CuEira {
namespace CPU {

CpuEnvironmentFactorHandler::CpuEnvironmentFactorHandler(const Container::HostVector* envData,
    const EnvironmentFactor* environmentFactor) :
    EnvironmentFactorHandler(environmentFactor), envData(envData) {

}

CpuEnvironmentFactorHandler::~CpuEnvironmentFactorHandler() {
  delete envData;
}

const Container::HostVector& CpuEnvironmentFactorHandler::getEnvironmentData() const {
  return envData;
}

} /* namespace CPU */
} /* namespace CuEira */
