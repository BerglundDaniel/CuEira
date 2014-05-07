#ifndef HOSTMATRIX_H_
#define HOSTMATRIX_H_

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
  HostMatrix(int numberOfRows, int numberOfColums, PRECISION* hostMatrix);
  virtual ~HostMatrix();

  int getNumberOfRows();
  int getNumberOfColumns();
  virtual HostVector* operator()(int column)=0;
  virtual const HostVector* operator()(int column) const=0;
  virtual PRECISION& operator()(int row, int column)=0;
  virtual const PRECISION& operator()(int row, int column) const=0;

protected:
  PRECISION* getMemoryPointer();

  PRECISION* hostMatrix;
  const int numberOfRows;
  const int numberOfColumns;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* HOSTMATRIX_H_ */
