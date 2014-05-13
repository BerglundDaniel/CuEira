#ifndef PINNEDHOSTMATRIX_H_
#define PINNEDHOSTMATRIX_H_

#include <sstream>

#include <HostMatrix.h>
#include <HostVector.h>
#include <PinnedHostVector.h>
#include <DimensionMismatch.h>

namespace CuEira {
namespace Container {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PinnedHostMatrix: public HostMatrix {
public:
  PinnedHostMatrix(unsigned int numberOfRows, unsigned int numberOfColumns);
  virtual ~PinnedHostMatrix();

  virtual HostVector* operator()(unsigned int column);
  virtual const HostVector* operator()(unsigned int column) const;
  virtual PRECISION& operator()(unsigned int row, unsigned int column);
  virtual const PRECISION& operator()(unsigned int row, unsigned int column) const;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* PINNEDHOSTMATRIX_H_ */
