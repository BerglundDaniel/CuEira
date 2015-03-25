#ifndef CUDAENVIRONMENTFACTORHANDLER_H_
#define CUDAENVIRONMENTFACTORHANDLER_H_

#include <vector>
#include <set>

#include <EnvironmentFactorHandler.h>
#include <DeviceMatrix.h>
#include <DeviceVector.h>
#include <EnvironmentFactor.h>
#include <CudaEnvironmentVector.h>
#include <EnvironmentFactorHandlerException.h>

namespace CuEira {
namespace CUDA {

/*
 * This class ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaEnvironmentFactorHandler: public EnvironmentFactorHandler {
public:
  explicit CudaEnvironmentFactorHandler(const Container::DeviceMatrix* dataMatrix,
      const std::vector<const EnvironmentFactor*>* environmentFactors, const std::vector<std::set<int>>* personsToSkip);
  virtual ~CudaEnvironmentFactorHandler();

  virtual Container::CUDA::CudaEnvironmentVector* getEnvironmentVector(
      const EnvironmentFactor& environmentFactor) const;

private:
  const Container::DeviceMatrix* dataMatrix;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* CUDAENVIRONMENTFACTORHANDLER_H_ */
