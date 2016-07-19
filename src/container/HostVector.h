#ifndef HOSTVECTOR_H_
#define HOSTVECTOR_H_

#include <math.h>

#include <DimensionMismatch.h>

namespace CuEira {
namespace Container {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class HostVector {
public:
  explicit HostVector(int numberOfRealRows, int numberOfRows, bool subview, PRECISION* hostVector);
  virtual ~HostVector();

  int getNumberOfRows() const;
  int getNumberOfColumns() const;
  virtual PRECISION& operator()(int index)=0;
  virtual const PRECISION& operator()(int index) const=0;

  int getRealNumberOfRows() const;
  int getRealNumberOfColumns() const;
  void updateSize(int numberOfRows);

  PRECISION* getMemoryPointer();
  const PRECISION* getMemoryPointer() const;

  HostVector(const HostVector&) = delete;
  HostVector(HostVector&&) = delete;
  HostVector& operator=(const HostVector&) = delete;
  HostVector& operator=(HostVector&&) = delete;

protected:
  PRECISION* hostVector;
  const int numberOfRealRows;
  int numberOfRows;
  const bool subview;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* HOSTVECTOR_H_ */
