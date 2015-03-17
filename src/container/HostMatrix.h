#ifndef HOSTMATRIX_H_
#define HOSTMATRIX_H_

#include <DimensionMismatch.h>
#include <HostVector.h>

namespace CuEira {
namespace Container {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class HostMatrix {

public:
  HostMatrix(int numberOfRows, int numberOfColumns, PRECISION* hostMatrix);
  virtual ~HostMatrix();

  int getNumberOfRows() const;
  int getNumberOfColumns() const;
  virtual HostVector* operator()(int column)=0;
  virtual const HostVector* operator()(int column) const=0;
  virtual PRECISION& operator()(int row, int column)=0;
  virtual const PRECISION& operator()(int row, int column) const=0;

  PRECISION* getMemoryPointer();
  const PRECISION* getMemoryPointer() const;

  HostMatrix(const HostMatrix&) = delete;
  HostMatrix(HostMatrix&&) = delete;
  HostMatrix& operator=(const HostMatrix&) = delete;
  HostMatrix& operator=(HostMatrix&&) = delete;

protected:
  PRECISION* hostMatrix;
  const int numberOfRows;
  const int numberOfColumns;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* HOSTMATRIX_H_ */
