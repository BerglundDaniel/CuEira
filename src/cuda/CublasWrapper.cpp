#include "CublasWrapper.h"

namespace CuEira {
namespace CUDA {

CublasWrapper::CublasWrapper(const Stream& stream) :
    stream(stream), cudaStream(stream.getCudaStream()), cublasHandle(stream.getCublasHandle()) {

}

CublasWrapper::~CublasWrapper() {

}

void CublasWrapper::copyVector(const DeviceVector& vectorFrom, DeviceVector& vectorTo) const {
#ifdef DEBUG
  if(vectorFrom.getNumberOfRows() != vectorTo.getNumberOfRows()){
    std::ostringstream os;
    os << "Number of rows doesn't match in copyVector function, they are " << vectorFrom.getNumberOfRows()
    << " and " << vectorTo.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

#ifdef DOUBLEPRECISION
  cublasDcopy(cublasHandle, vectorFrom.getNumberOfRows(), vectorFrom.getMemoryPointer(), 1, vectorTo.getMemoryPointer(),
      1);
#else
  cublasScopy(cublasHandle, vectorFrom.getNumberOfRows(), vectorFrom.getMemoryPointer(), 1, vectorTo.getMemoryPointer(),
      1);
#endif

#ifdef FERMI
  syncStream();
#endif
}

void CublasWrapper::matrixVectorMultiply(const DeviceMatrix& matrix, const DeviceVector& vector,
    DeviceVector& result) const {
#ifdef DEBUG
  if((matrix.getNumberOfRows() != result.getNumberOfRows()) || (vector.getNumberOfRows() != matrix.getNumberOfColumns())){
    std::ostringstream os;
    os << "Sizes doesn't match in matrixVectorMultiply function." << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

#ifdef DOUBLEPRECISION
  cublasDgemv(cublasHandle, CUBLAS_OP_N, matrix.getNumberOfRows(), matrix.getNumberOfColumns(), constOne,
      matrix.getMemoryPointer(), matrix.getNumberOfRows(), vector.getMemoryPointer(), 1, constZero, result.getMemoryPointer(),
      1);
#else
  cublasSgemv(cublasHandle, CUBLAS_OP_N, matrix.getNumberOfRows(), matrix.getNumberOfColumns(), constOne,
      matrix.getMemoryPointer(), matrix.getNumberOfRows(), vector.getMemoryPointer(), 1, constZero,
      result.getMemoryPointer(), 1);
#endif

#ifdef FERMI
  syncStream();
#endif
}

void CublasWrapper::matrixTransVectorMultiply(const DeviceMatrix& matrix, const DeviceVector& vector,
    DeviceVector& result) const {
#ifdef DEBUG
  if((matrix.getNumberOfColumns() != result.getNumberOfRows()) || (vector.getNumberOfRows() != matrix.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows(columns for matrix) doesn't match in matrixTransVectorMultiply function" << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

#ifdef DOUBLEPRECISION
  cublasDgemv(cublasHandle, CUBLAS_OP_T, matrix.getNumberOfRows(), matrix.getNumberOfColumns(), constOne,
      matrix.getMemoryPointer(), matrix.getNumberOfRows(), vector.getMemoryPointer(), 1, constZero, result.getMemoryPointer(),
      1);
#else
  cublasSgemv(cublasHandle, CUBLAS_OP_T, matrix.getNumberOfRows(), matrix.getNumberOfColumns(), constOne,
      matrix.getMemoryPointer(), matrix.getNumberOfRows(), vector.getMemoryPointer(), 1, constZero,
      result.getMemoryPointer(), 1);
#endif

#ifdef FERMI
  syncStream();
#endif
}

void CublasWrapper::matrixTransMatrixMultiply(const DeviceMatrix& matrix1, const DeviceMatrix& matrix2,
    DeviceMatrix& result) const {
#ifdef DEBUG
  if((matrix1.getNumberOfRows() != matrix2.getNumberOfRows()) || (matrix1.getNumberOfColumns() != result.getNumberOfRows())
      || (matrix2.getNumberOfColumns() != result.getNumberOfColumns())){
    throw DimensionMismatch("Matrix sizes doesn't match in matrixTransMatrixMultiply");
  }
#endif

#ifdef DOUBLEPRECISION
  cublasDgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, matrix1.getNumberOfColumns(), matrix2.getNumberOfColumns(),
      matrix1.getNumberOfRows(), constOne, matrix1.getMemoryPointer(), matrix1.getNumberOfRows(),
      matrix2.getMemoryPointer(), matrix2.getNumberOfRows(), constZero, result.getMemoryPointer(),
      result.getNumberOfRows());
#else
  cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, matrix1.getNumberOfColumns(), matrix2.getNumberOfColumns(),
      matrix1.getNumberOfRows(), constOne, matrix1.getMemoryPointer(), matrix1.getNumberOfRows(),
      matrix2.getMemoryPointer(), matrix2.getNumberOfRows(), constZero, result.getMemoryPointer(),
      result.getNumberOfRows());
#endif

#ifdef FERMI
  syncStream();
#endif
}

void CublasWrapper::sumResultToHost(const DeviceVector& vector, const DeviceVector& oneVector,
    PRECISION& sumHost) const {
#ifdef DOUBLEPRECISION
  cublasDdot(cublasHandle, vector.getNumberOfRows(), vector.getMemoryPointer(), 1, oneVector.getMemoryPointer(), 1,
      &sumHost);
#else
  cublasSdot(cublasHandle, vector.getNumberOfRows(), vector.getMemoryPointer(), 1, oneVector.getMemoryPointer(), 1,
      &sumHost);
#endif

#ifdef FERMI
  syncStream();
#endif
}

} /* namespace CUDA */
} /* namespace CuEira */
