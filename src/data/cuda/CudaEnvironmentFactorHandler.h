#ifndef CUDAENVIRONMENTFACTORHANDLER_H_
#define CUDAENVIRONMENTFACTORHANDLER_H_

#include <EnvironmentFactorHandler.h>
#include <DeviceVector.h>
#include <EnvironmentFactor.h>
#include <CudaEnvironmentVector.h>

namespace CuEira {
namespace CUDA {

/*
 * This class ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaEnvironmentFactorHandler: public EnvironmentFactorHandler {
public:
  explicit CudaEnvironmentFactorHandler(const Container::DeviceVector* envData,
      const EnvironmentFactor* environmentFactor);
  virtual ~CudaEnvironmentFactorHandler();

  virtual Container::DeviceVector& getEnvironmentData() const;

private:
  const Container::DeviceVector* envData;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* CUDAENVIRONMENTFACTORHANDLER_H_ */
