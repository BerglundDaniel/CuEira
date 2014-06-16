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

  void matrixVectorMultiply(const HostMatrix& matrix, const HostVector& vector, HostVector& resultVector,
      PRECISION alpha, PRECISION beta) const;

  void matrixTransVectorMultiply(const HostMatrix& matrix, const HostVector& vector, HostVector& resultVector,
      PRECISION alpha, PRECISION beta) const;

  void matrixTransMatrixMultiply(const HostMatrix& matrix1, const HostMatrix& matrix2, HostMatrix& resultMatrix,
      PRECISION alpha, PRECISION beta) const;

  /**
   * vector2(i)-vector1(i)
   */
  void differenceElememtWise(const HostVector& vector1, HostVector& vector2) const;

  void absoluteSum(const HostVector& vector, PRECISION* result) const;
};

} /* namespace CuEira */

#endif /* MKLWRAPPER_H_ */
