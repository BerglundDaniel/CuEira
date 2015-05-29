#ifndef REGULARHOSTMATRIX_H_
#define REGULARHOSTMATRIX_H_

#include <math.h>
#include <sstream>

#include <HostMatrix.h>
#include <HostVector.h>
#include <RegularHostVector.h>
#include <DimensionMismatch.h>

namespace CuEira {
namespace Container {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class RegularHostMatrix: public HostMatrix {
public:
  explicit RegularHostMatrix(int numberOfRows, int numberOfColumns);
  virtual ~RegularHostMatrix();

  virtual RegularHostVector* operator()(int column);
  virtual const RegularHostVector* operator()(int column) const;
  virtual PRECISION& operator()(int row, int column);
  virtual const PRECISION& operator()(int row, int column) const;

  RegularHostMatrix(const RegularHostMatrix&) = delete;
  RegularHostMatrix(RegularHostMatrix&&) = delete;
  RegularHostMatrix& operator=(const RegularHostMatrix&) = delete;
  RegularHostMatrix& operator=(RegularHostMatrix&&) = delete;

};

} /* namespace Container */
} /* namespace CuEira */

#endif /* REGULARHOSTMATRIX_H_ */
