#ifndef CUDAENVIRONMENTVECTOR_H_
#define CUDAENVIRONMENTVECTOR_H_

#include <EnvironmentFactorHandler.h>
#include <EnvironmentVector.h>
#include <DeviceVector.h>
#include <EnvironmentFactor.h>
#include <KernelWrapper.h>
#include <CublasWrapper.h>
#include <Recode.h>
#include <VariableType.h>
#include <InvalidState.h>
#include <Stream.h>

namespace CuEira {
namespace Container {
namespace CUDA {

using namespace CuEira::CUDA;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaEnvironmentVector: public EnvironmentVector<DeviceVector> {
public:
  CudaEnvironmentVector(const EnvironmentFactorHandler<DeviceVector>& environmentFactorHandler, const Stream& stream);
  virtual ~CudaEnvironmentVector();

protected:
  virtual void recodeProtective();
  virtual void recodeAllRisk();

  const Stream& stream;
};

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */

#endif /* CUDAENVIRONMENTVECTOR_H_ */
