#ifndef PINNEDHOSTVECTOR_H_
#define PINNEDHOSTVECTOR_H_

#include <sstream>

#include <HostVector.h>
#include <DimensionMismatch.h>
#include <CudaAdapter.cu>

namespace CuEira {
namespace CUDA{
class DeviceToHost;
class HostToDevice;
}
namespace Container {
class PinnedHostMatrix;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PinnedHostVector: public HostVector {
  friend CUDA::DeviceToHost;
  friend CUDA::HostToDevice;
  friend PinnedHostMatrix;
public:
  PinnedHostVector(unsigned int numberOfRows);
  virtual ~PinnedHostVector();

  virtual PRECISION& operator()(unsigned int index);
  virtual const PRECISION& operator()(unsigned int index) const;

protected:
  PinnedHostVector(unsigned int numberOfRows, PRECISION* hostVector, bool subview);
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* PINNEDHOSTVECTOR_H_ */
