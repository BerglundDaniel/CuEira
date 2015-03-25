#ifndef CUDAPHENOTYPEVECTOR_H_
#define CUDAPHENOTYPEVECTOR_H_

#include <PhenotypeVector.h>
#include <DeviceVector.h>
#include <CudaPhenotypeHandler.h>
#include <InvalidState.h>
#include <CudaMissingDataHandler.h>

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
  CudaPhenotypeVector(const CudaPhenotypeHandler& cudaPhenotypeHandler);
  virtual ~CudaPhenotypeVector();

  virtual const DeviceVector& getPhenotypeData() const;
  virtual void applyMissing(const CudaMissingDataHandler& missingDataHandler);

protected:
  const CudaPhenotypeHandler& cudaPhenotypeHandler;
  const DeviceVector& orgData;
  DeviceVector* phenotypeExMissing;
};

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */

#endif /* CUDAPHENOTYPEVECTOR_H_ */
