#ifndef HOSTVECTOR_H_
#define HOSTVECTOR_H_

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
  HostVector(unsigned int numberOfRows, bool subview, PRECISION* hostVector);
  virtual ~HostVector();

  int getNumberOfRows() const;
  int getNumberOfColumns() const;
  virtual PRECISION& operator()(unsigned int index)=0;
  virtual const PRECISION& operator()(unsigned int index) const=0;

  PRECISION* getMemoryPointer();
  const PRECISION* getMemoryPointer() const;

  HostVector(const HostVector&) = delete;
  HostVector(HostVector&&) = delete;
  HostVector& operator=(const HostVector&) = delete;
  HostVector& operator=(HostVector&&) = delete;

protected:
  PRECISION* hostVector;
  const unsigned int numberOfRows;
  const unsigned int numberOfColumns;
  const bool subview;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* HOSTVECTOR_H_ */
