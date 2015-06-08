#ifndef CUDAENVIRONMENTFACTORHANDLERFACTORY_H_
#define CUDAENVIRONMENTFACTORHANDLERFACTORY_H_

#include <memory>
#include <vector>
#include <string>

#include <EnvironmentFactorHandlerFactory.h>
#include <EnvironmentFactorHandler.h>
#include <DeviceVector.h>
#include <HostToDevice.h>
#include <PinnedHostMatrix.h>
#include <PinnedHostVector.h>

namespace CuEira {
namespace CUDA {

/*
 * This class ...
 *
 *  @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaEnvironmentFactorHandlerFactory: public EnvironmentFactorHandlerFactory<PinnedHostMatrix, PinnedHostVector> {
public:
  explicit CudaEnvironmentFactorHandlerFactory(const Configuration& configuration,
      const std::vector<std::string>& columnNames, const Container::PinnedHostMatrix& matrix);
  virtual ~CudaEnvironmentFactorHandlerFactory();

  virtual EnvironmentFactorHandler<Container::DeviceVector>* constructEnvironmentFactorHandler(
      const HostToDevice& hostToDevice) const;

  CudaEnvironmentFactorHandlerFactory(const CudaEnvironmentFactorHandlerFactory&) = delete;
  CudaEnvironmentFactorHandlerFactory(CudaEnvironmentFactorHandlerFactory&&) = delete;
  CudaEnvironmentFactorHandlerFactory& operator=(const CudaEnvironmentFactorHandlerFactory&) = delete;
  CudaEnvironmentFactorHandlerFactory& operator=(CudaEnvironmentFactorHandlerFactory&&) = delete;

protected:

};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* CUDAENVIRONMENTFACTORHANDLERFACTORY_H_ */
