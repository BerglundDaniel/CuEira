#ifndef CUDAPHENOTYPEVECTOR_H_
#define CUDAPHENOTYPEVECTOR_H_

#include <set>

#include <PhenotypeVector.h>
#include <DeviceVector.h>
#include <CudaPhenotypeHandler.h>
#include <InvalidState.h>
#include <KernelWrapper.h>
#include <HostToDevice.h>

namespace CuEira {
namespace Container {
namespace CUDA {

using namespace CuEira::CUDA;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaPhenotypeVector: public PhenotypeVector {
public:
  CudaPhenotypeVector(const CudaPhenotypeHandler& cudaPhenotypeHandler, const HostToDevice& hostToDevice,
      const KernelWrapper& kernelWrapper);
  virtual ~CudaPhenotypeVector();

  virtual const DeviceVector& getPhenotypeData() const;

protected:
  virtual void copyNonMissingData(const std::set<int>& personsToSkip);

  const CudaPhenotypeHandler& cudaPhenotypeHandler;
  const HostToDevice& hostToDevice;
  const KernelWrapper& kernelWrapper;
  const DeviceVector& orgData;
  DeviceVector* phenotypeExMissing;
};

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */

#endif /* CUDAPHENOTYPEVECTOR_H_ */
