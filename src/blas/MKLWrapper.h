#ifndef MKLWRAPPER_H_
#define MKLWRAPPER_H_

#ifdef MKL_BLAS
#include <mkl_vml.h>
#include <mkl.h>
#else
extern "C" {
  #include <cblas.h>
  #include <clapack.h>
}
#endif

#include <iostream>

#include <BlasException.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <DimensionMismatch.h>

namespace CuEira {
namespace Blas {

using namespace CuEira::Container;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */

void copyVector(const HostVector& vectorFrom, HostVector& vectorTo);

void svd(HostMatrix& matrix, HostMatrix& uSVD, HostVector& sigma, HostMatrix& vtSVD);

void matrixVectorMultiply(const HostMatrix& matrix, const HostVector& vector, HostVector& resultVector,
    PRECISION alpha = 1, PRECISION beta = 0);

void matrixTransVectorMultiply(const HostMatrix& matrix, const HostVector& vector, HostVector& resultVector,
    PRECISION alpha = 1, PRECISION beta = 0);

void matrixMatrixMultiply(const HostMatrix& matrix1, const HostMatrix& matrix2, HostMatrix& resultMatrix,
    PRECISION alpha = 1, PRECISION beta = 0);

void matrixTransMatrixMultiply(const HostMatrix& matrix1, const HostMatrix& matrix2, HostMatrix& resultMatrix,
    PRECISION alpha = 1, PRECISION beta = 0);

/**
 * vector2(i) = vector2(i)-vector1(i)
 */
void differenceElememtWise(const HostVector& vector1, HostVector& vector2);

void multiplicationElementWise(const HostVector& vector1, const HostVector& vector2, HostVector& result);

void absoluteSum(const HostVector& vector, PRECISION& result);

} /* namespace Blas */
} /* namespace CuEira */

#endif /* MKLWRAPPER_H_ */
