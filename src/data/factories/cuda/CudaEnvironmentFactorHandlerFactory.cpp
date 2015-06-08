#include "CudaEnvironmentFactorHandlerFactory.h"

namespace CuEira {
namespace CUDA {

CudaEnvironmentFactorHandlerFactory::CudaEnvironmentFactorHandlerFactory(const Configuration& configuration,
    const std::vector<std::string>& columnNames, const Container::PinnedHostMatrix& matrix) :
    EnvironmentFactorHandlerFactory(configuration, columnNames, matrix) {

}

CudaEnvironmentFactorHandlerFactory::~CudaEnvironmentFactorHandlerFactory() {

}

EnvironmentFactorHandler<Container::DeviceVector>* CudaEnvironmentFactorHandlerFactory::constructEnvironmentFactorHandler(
    const HostToDevice& hostToDevice) const {
  const Container::DeviceVector* envDataDevice = hostToDevice.transferVector(*envData);
  return new EnvironmentFactorHandler<DeviceVector>(environmentFactor, envDataDevice);
}

} /* namespace CUDA */
} /* namespace CuEira */
