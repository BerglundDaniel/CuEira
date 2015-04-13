#ifndef HOSTMATRIX_H_
#define HOSTMATRIX_H_

#include <math.h>

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
  HostMatrix(int numberOfRealRows, int numberOfRealColumns, int numberOfRows, int numberOfColumns,
      PRECISION* hostMatrix);
  virtual ~HostMatrix();

  int getNumberOfRows() const;
  int getNumberOfColumns() const;
  virtual HostVector* operator()(int column)=0;
  virtual const HostVector* operator()(int column) const=0;
  virtual PRECISION& operator()(int row, int column)=0;
  virtual const PRECISION& operator()(int row, int column) const=0;

  int getRealNumberOfRows() const;
  int getRealNumberOfColumns() const;
  void updateSize(int numberOfRows, int numberOfColumns);
  void updateNumberOfRows(int numberOfRows);
  void updateNumberOfColumns(int numberOfColumns);

  //TODO add iterators

  PRECISION* getMemoryPointer();
  const PRECISION* getMemoryPointer() const;

  HostMatrix(const HostMatrix&) = delete;
  HostMatrix(HostMatrix&&) = delete;
  HostMatrix& operator=(const HostMatrix&) = delete;
  HostMatrix& operator=(HostMatrix&&) = delete;

protected:
  PRECISION* hostMatrix;
  const int numberOfRealRows;
  const int numberOfRealColumns;
  int numberOfRows;
  int numberOfColumns;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* HOSTMATRIX_H_ */
