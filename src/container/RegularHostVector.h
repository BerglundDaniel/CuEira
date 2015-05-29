#ifndef REGULARHOSTVECTOR_H_
#define REGULARHOSTVECTOR_H_

#include <math.h>
#include <sstream>

#include <HostVector.h>
#include <DimensionMismatch.h>

namespace CuEira {
namespace Container {
class RegularHostMatrix;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class RegularHostVector: public HostVector {
  friend RegularHostMatrix;
public:
  explicit RegularHostVector(int numberOfRows);
  virtual ~RegularHostVector();

  virtual PRECISION& operator()(int index);
  virtual const PRECISION& operator()(int index) const;

  RegularHostVector(const RegularHostVector&) = delete;
  RegularHostVector(RegularHostVector&&) = delete;
  RegularHostVector& operator=(const RegularHostVector&) = delete;
  RegularHostVector& operator=(RegularHostVector&&) = delete;

protected:
  RegularHostVector(int numberOfRealRows, int numberOfRows, PRECISION* hostVector, bool subview);

};

} /* namespace Container */
} /* namespace CuEira */

#endif /* REGULARHOSTVECTOR_H_ */
