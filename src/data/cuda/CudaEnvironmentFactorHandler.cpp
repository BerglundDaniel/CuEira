#include "CudaEnvironmentFactorHandler.h"

namespace CuEira {
namespace CUDA {

CudaEnvironmentFactorHandler::CudaEnvironmentFactorHandler(const Container::DeviceVector* envData,
    const EnvironmentFactor* environmentFactor) :
    EnvironmentFactorHandler(environmentFactor), envData(envData) {

}

CudaEnvironmentFactorHandler::~CudaEnvironmentFactorHandler() {
  delete envData;
}

Container::DeviceVector& CudaEnvironmentFactorHandler::getEnvironmentData() const {
  return envData;
}

} /* namespace CUDA */
} /* namespace CuEira */
