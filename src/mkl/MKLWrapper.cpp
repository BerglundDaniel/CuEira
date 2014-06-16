#include "MKLWrapper.h"

namespace CuEira {

MKLWrapper::MKLWrapper() {

}

MKLWrapper::~MKLWrapper() {

}

void MKLWrapper::copyVector(const HostVector& vectorFrom, HostVector& vectorTo) const {
#ifdef DEBUG
  if(vectorFrom.getNumberOfRows() != vectorTo.getNumberOfRows()){
    throw new DimensionMismatch("Length of vectors in copyVector doesn't match.");
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
    throw new DimensionMismatch("Numbers of rows doesn't match in SVD");
  }

  if((matrix.getNumberOfColumns() != uSVD.getNumberOfColumns())
      || (matrix.getNumberOfColumns() != vtSVD.getNumberOfColumns())){
    throw new DimensionMismatch("Numbers of columns doesn't match in SVD");
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

void MKLWrapper::matrixVectorMultiply() const {

}

void MKLWrapper::matrixTransVectorMultiply() const {

}

void MKLWrapper::matrixTransMatrixMultiply(const HostMatrix& matrix1, const HostMatrix& matrix2,
    HostMatrix& resultMatrix, PRECISION alpha, PRECISION beta) const {
#ifdef DEBUG
  if((matrix1.getNumberOfRows() != matrix2.getNumberOfRows()) || (matrix1.getNumberOfColumns() != resultMatrix.getNumberOfRows())
      || (matrix2.getNumberOfColumns() != resultMatrix.getNumberOfColumns())){
    throw new DimensionMismatch("Matrix sizes doesn't match in matrixTransMatrixMultiply");
  }
#endif

  int numberOfColumnsMatrix1 = matrix1.getNumberOfColumns(); //Rows after transpose
  int numberOfColumnsMatrix2 = matrix2.getNumberOfColumns();
  int numberOfRowsMatrix2 = matrix2.getNumberOfRows();

  cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, numberOfColumnsMatrix1, numberOfColumnsMatrix2,
      numberOfRowsMatrix2, alpha, matrix1.getMemoryPointer(), numberOfRowsMatrix2, matrix2.getMemoryPointer(),
      numberOfRowsMatrix2, beta, resultMatrix.getMemoryPointer(), numberOfColumnsMatrix1);
}

void MKLWrapper::absoluteDifferenceElememtWise() const {

}

void MKLWrapper::sum() const {

}

} /* namespace CuEira */
