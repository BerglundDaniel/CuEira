#ifndef PINNEDHOSTVECTOR_H_
#define PINNEDHOSTVECTOR_H_

#include <math.h>
#include <sstream>

#include <HostVector.h>
#include <DimensionMismatch.h>
#include <CudaAdapter.cu>

namespace CuEira {
namespace Container {
class PinnedHostMatrix;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PinnedHostVector: public HostVector {
  friend PinnedHostMatrix;
public:
  PinnedHostVector(int numberOfRows);
  virtual ~PinnedHostVector();

  virtual PRECISION& operator()(int index);
  virtual const PRECISION& operator()(int index) const;

  PinnedHostVector(const PinnedHostVector&) = delete;
  PinnedHostVector(PinnedHostVector&&) = delete;
  PinnedHostVector& operator=(const PinnedHostVector&) = delete;
  PinnedHostVector& operator=(PinnedHostVector&&) = delete;

protected:
  PinnedHostVector(int numberOfRealRows, int numberOfRows, PRECISION* hostVector, bool subview);
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* PINNEDHOSTVECTOR_H_ */
