#ifndef PINNEDHOSTMATRIX_H_
#define PINNEDHOSTMATRIX_H_

#include <math.h>
#include <sstream>

#include <HostMatrix.h>
#include <HostVector.h>
#include <PinnedHostVector.h>
#include <DimensionMismatch.h>
#include <CudaAdapter.cu>

namespace CuEira {
namespace Container {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PinnedHostMatrix: public HostMatrix {
public:
  explicit PinnedHostMatrix(int numberOfRows, int numberOfColumns);
  virtual ~PinnedHostMatrix();

  virtual PinnedHostVector* operator()(int column);
  virtual const PinnedHostVector* operator()(int column) const;
  virtual PRECISION& operator()(int row, int column);
  virtual const PRECISION& operator()(int row, int column) const;

  PinnedHostMatrix(const PinnedHostMatrix&) = delete;
  PinnedHostMatrix(PinnedHostMatrix&&) = delete;
  PinnedHostMatrix& operator=(const PinnedHostMatrix&) = delete;
  PinnedHostMatrix& operator=(PinnedHostMatrix&&) = delete;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* PINNEDHOSTMATRIX_H_ */
