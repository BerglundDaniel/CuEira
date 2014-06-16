#ifndef MKLWRAPPER_H_
#define MKLWRAPPER_H_

#include <mkl.h>

#include <MKLException.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <DimensionMismatch.h>

namespace CuEira {

using namespace CuEira::Container;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class MKLWrapper {
public:
  MKLWrapper();
  virtual ~MKLWrapper();

  void copyVector(const HostVector& vectorFrom, HostVector& vectorTo) const;
  bool svd(HostMatrix& matrix, HostMatrix& uSVD, HostVector& sigma, HostMatrix& vtSVD) const;
  void matrixVectorMultiply() const;
  void matrixTransVectorMultiply(const HostMatrix& matrix1, const HostMatrix& matrix2, HostMatrix& resultMatrix,
      PRECISION alpha, PRECISION beta) const;
  void matrixTransMatrixMultiply() const;
  void absoluteDifferenceElememtWise() const;
  void sum() const;
};

} /* namespace CuEira */

#endif /* MKLWRAPPER_H_ */
