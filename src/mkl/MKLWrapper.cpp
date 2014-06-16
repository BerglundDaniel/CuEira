#include "MKLWrapper.h"

namespace CuEira {

MKLWrapper::MKLWrapper() {

}

MKLWrapper::~MKLWrapper() {

}

void MKLWrapper::copyVector(const HostVector& vectorFrom, HostVector& vectorTo) const {
#ifdef DEBUG
  if(vectorFrom.getNumberOfRows() != vectorTo.getNumberOfRows()){
    throw DimensionMismatch("Length of vectors in copyVector doesn't match.");
  }
#endif

  int length = vectorFrom.getNumberOfRows();
#ifdef DOUBLEPRECISION
  cblas_dcopy(length, vectorFrom.getMemoryPointer(), 1, vectorTo.getMemoryPointer(), 1);
#else
  cblas_scopy(length, vectorFrom.getMemoryPointer(), 1, vectorTo.getMemoryPointer(), 1);
#endif
}

bool MKLWrapper::svd(HostMatrix& matrix, HostMatrix& uSVD, HostVector& sigma, HostMatrix& vtSVD) const {
#ifdef DEBUG
  if((matrix.getNumberOfRows() != uSVD.getNumberOfRows()) || (matrix.getNumberOfRows() != sigma.getNumberOfRows())
      || (matrix.getNumberOfRows() != vtSVD.getNumberOfRows())){
    throw DimensionMismatch("Numbers of rows doesn't match in SVD");
  }

  if((matrix.getNumberOfColumns() != uSVD.getNumberOfColumns())
      || (matrix.getNumberOfColumns() != vtSVD.getNumberOfColumns())){
    throw DimensionMismatch("Numbers of columns doesn't match in SVD");
  }
#endif

  int size = matrix.getNumberOfRows();
#ifdef DOUBLEPRECISION
  MKL_INT status = LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'A', size, size, matrix.getMemoryPointer(), size,
      sigma.getMemoryPointer(), uSVD.getMemoryPointer(), size, vtSVD.getMemoryPointer(), size);
#else
  MKL_INT status = LAPACKE_sgesdd(LAPACK_COL_MAJOR, 'A', size, size, matrix.getMemoryPointer(), size,
      sigma.getMemoryPointer(), uSVD.getMemoryPointer(), size, vtSVD.getMemoryPointer(), size);
#endif

  if(status < 0){
    throw new MKLException("Illegal values in matrix.");
  }else if(status > 0){
    return false;
  }
  return true;
}

void MKLWrapper::matrixVectorMultiply(const HostMatrix& matrix, const HostVector& vector, HostVector& resultVector,
    PRECISION alpha, PRECISION beta) const {
#ifdef DEBUG
  if((matrix.getNumberOfColumns() != vector.getNumberOfRows())
      || (matrix.getNumberOfRows() != resultVector.getNumberOfRows())){
    throw DimensionMismatch("Sizes doesn't match in matrixTransVectorMultiply");
  }
#endif

//kanske fel med storleken m och n FIXME
  int m = matrix.getNumberOfRows();
  int n = matrix.getNumberOfColumns();

#ifdef DOUBLEPRECISION
  cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, matrix.getMemoryPointer(), m, vector.getMemoryPointer(), 1,
      beta, resultVector.getMemoryPointer(), 1);
#else
  cblas_sgemv(CblasColMajor, CblasNoTrans, m, n, alpha, matrix.getMemoryPointer(), m, vector.getMemoryPointer(), 1,
      beta, resultVector.getMemoryPointer(), 1);
#endif
}

void MKLWrapper::matrixTransVectorMultiply(const HostMatrix& matrix, const HostVector& vector, HostVector& resultVector,
    PRECISION alpha, PRECISION beta) const {
#ifdef DEBUG
  if((matrix.getNumberOfRows() != vector.getNumberOfRows())
      || (matrix.getNumberOfColumns() != resultVector.getNumberOfRows())){
    throw DimensionMismatch("Sizes doesn't match in matrixTransVectorMultiply");
  }
#endif

//kanske fel med storleken m och n FIXME
  int m = matrix.getNumberOfRows();
  int n = matrix.getNumberOfColumns();

#ifdef DOUBLEPRECISION
  cblas_dgemv(CblasColMajor, CblasTrans, m, n, alpha, matrix.getMemoryPointer(), m, vector.getMemoryPointer(), 1,
      beta, resultVector.getMemoryPointer(), 1);
#else
  cblas_sgemv(CblasColMajor, CblasTrans, m, n, alpha, matrix.getMemoryPointer(), m, vector.getMemoryPointer(), 1, beta,
      resultVector.getMemoryPointer(), 1);
#endif
}

void MKLWrapper::matrixTransMatrixMultiply(const HostMatrix& matrix1, const HostMatrix& matrix2,
    HostMatrix& resultMatrix, PRECISION alpha, PRECISION beta) const {
#ifdef DEBUG
  if((matrix1.getNumberOfRows() != matrix2.getNumberOfRows()) || (matrix1.getNumberOfColumns() != resultMatrix.getNumberOfRows())
      || (matrix2.getNumberOfColumns() != resultMatrix.getNumberOfColumns())){
    throw DimensionMismatch("Matrix sizes doesn't match in matrixTransMatrixMultiply");
  }
#endif

  int numberOfColumnsMatrix1 = matrix1.getNumberOfColumns(); //Rows after transpose
  int numberOfColumnsMatrix2 = matrix2.getNumberOfColumns();
  int numberOfRowsMatrix2 = matrix2.getNumberOfRows();

#ifdef DOUBLEPRECISION
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, numberOfColumnsMatrix1, numberOfColumnsMatrix2,
      numberOfRowsMatrix2, alpha, matrix1.getMemoryPointer(), numberOfRowsMatrix2, matrix2.getMemoryPointer(),
      numberOfRowsMatrix2, beta, resultMatrix.getMemoryPointer(), numberOfColumnsMatrix1);
#else
  cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, numberOfColumnsMatrix1, numberOfColumnsMatrix2,
      numberOfRowsMatrix2, alpha, matrix1.getMemoryPointer(), numberOfRowsMatrix2, matrix2.getMemoryPointer(),
      numberOfRowsMatrix2, beta, resultMatrix.getMemoryPointer(), numberOfColumnsMatrix1);
#endif
}

void MKLWrapper::differenceElememtWise(const HostVector& vector1, HostVector& vector2) const {
#ifdef DEBUG
  if(vector1.getNumberOfRows() != vector2.getNumberOfRows()){
    throw DimensionMismatch("Length of vectors in copyVector doesn't match.");
  }
#endif

  int size = vector1.getNumberOfRows();
#ifdef DOUBLEPRECISION
  cblas_daxpy(size, -1.0, vector1.getMemoryPointer(), 1, vector2.getMemoryPointer(), 1);
#else
  cblas_saxpy(size, -1.0, vector1.getMemoryPointer(), 1, vector2.getMemoryPointer(), 1);
#endif
}

void MKLWrapper::absoluteSum(const HostVector& vector, PRECISION* result) const {
#ifdef DOUBLEPRECISION
  *result = cblas_dasum(vector.getNumberOfRows(), vector.getMemoryPointer(), 1);
#else
  *result = cblas_sasum(vector.getNumberOfRows(), vector.getMemoryPointer(), 1);
#endif
}

} /* namespace CuEira */
