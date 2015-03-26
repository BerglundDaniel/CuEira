#ifndef CUDAENVIRONMENTVECTOR_H_
#define CUDAENVIRONMENTVECTOR_H_

#include <CudaEnvironmentFactorHandler.h>
#include <CudaMissingDataHandler.h>
#include <EnvironmentVector.h>
#include <DeviceVector.h>
#include <EnvironmentFactor.h>
#include <KernelWrapper.h>
#include <CublasWrapper.h>
#include <StatisticModel.h>
#include <Recode.h>
#include <VariableType.h>
#include <InvalidState.h>

namespace CuEira {
namespace Container {
namespace CUDA {

using namespace CuEira::CUDA;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaEnvironmentVector: public EnvironmentVector {
public:
  CudaEnvironmentVector(const CudaEnvironmentFactorHandler& cudaEnvironmentFactorHandler,
      const KernelWrapper& kernelWrapper, const CublasWrapper& cublasWrapper);
  virtual ~CudaEnvironmentVector();

  virtual const Container::DeviceVector& getEnvironmentData() const;
  virtual void recode(Recode recode);
  virtual void recode(Recode recode, const CudaMissingDataHandler& missingDataHandler);

private:
  void recodeProtective();

  const KernelWrapper& kernelWrapper;
  const CublasWrapper& cublasWrapper;
  const DeviceVector& originalData;
  DeviceVector* recodedData;
};

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */

#endif /* CUDAENVIRONMENTVECTOR_H_ */
