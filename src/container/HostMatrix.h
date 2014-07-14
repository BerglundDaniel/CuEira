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
  HostMatrix(unsigned int numberOfRows, unsigned int numberOfColums, PRECISION* hostMatrix);
  virtual ~HostMatrix();

  int getNumberOfRows() const;
  int getNumberOfColumns() const;
  virtual HostVector* operator()(unsigned int column)=0;
  virtual const HostVector* operator()(unsigned int column) const=0;
  virtual PRECISION& operator()(unsigned int row, unsigned int column)=0;
  virtual const PRECISION& operator()(unsigned int row, unsigned int column) const=0;

  PRECISION* getMemoryPointer();
  const PRECISION* getMemoryPointer() const;

protected:
  PRECISION* hostMatrix;
  const unsigned int numberOfRows;
  const unsigned int numberOfColumns;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* HOSTMATRIX_H_ */
