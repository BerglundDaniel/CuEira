#include "CublasWrapper.h"

namespace CuEira {
namespace CUDA {
namespace Kernel {

void copyVector(const Stream& stream, const DeviceVector& vectorFrom, DeviceVector& vectorTo){
#ifdef DEBUG
  if(vectorFrom.getNumberOfRows() != vectorTo.getNumberOfRows()){
    std::ostringstream os;
    os << "Number of rows doesn't match in copyVector function, they are " << vectorFrom.getNumberOfRows()
    << " and " << vectorTo.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const cublasHandle_t& cublasHandle = stream.getCublasHandle();

#ifdef DOUBLEPRECISION
  cublasDcopy(cublasHandle, vectorFrom.getNumberOfRows(), vectorFrom.getMemoryPointer(), 1, vectorTo.getMemoryPointer(),
      1);
#else
  cublasScopy(cublasHandle, vectorFrom.getNumberOfRows(), vectorFrom.getMemoryPointer(), 1, vectorTo.getMemoryPointer(),
      1);
#endif

#ifdef FERMI
  stream.syncStream();
#endif
}

void matrixVectorMultiply(const Stream& stream, const DeviceMatrix& matrix, const DeviceVector& vector,
    DeviceVector& result){
#ifdef DEBUG
  if((matrix.getNumberOfRows() != result.getNumberOfRows()) || (vector.getNumberOfRows() != matrix.getNumberOfColumns())){
    std::ostringstream os;
    os << "Sizes doesn't match in matrixVectorMultiply function." << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const cublasHandle_t& cublasHandle = stream.getCublasHandle();

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
  stream.syncStream();
#endif
}

void matrixTransVectorMultiply(const Stream& stream, const DeviceMatrix& matrix, const DeviceVector& vector,
    DeviceVector& result){
#ifdef DEBUG
  if((matrix.getNumberOfColumns() != result.getNumberOfRows()) || (vector.getNumberOfRows() != matrix.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows(columns for matrix) doesn't match in matrixTransVectorMultiply function" << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const cublasHandle_t& cublasHandle = stream.getCublasHandle();

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
  stream.syncStream();
#endif
}

void matrixTransMatrixMultiply(const Stream& stream, const DeviceMatrix& matrix1, const DeviceMatrix& matrix2,
    DeviceMatrix& result){
#ifdef DEBUG
  if((matrix1.getNumberOfRows() != matrix2.getNumberOfRows()) || (matrix1.getNumberOfColumns() != result.getNumberOfRows())
      || (matrix2.getNumberOfColumns() != result.getNumberOfColumns())){
    throw DimensionMismatch("Matrix sizes doesn't match in matrixTransMatrixMultiply");
  }
#endif

  const cublasHandle_t& cublasHandle = stream.getCublasHandle();

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
  stream.syncStream();
#endif
}

void sumResultToHost(const Stream& stream, const DeviceVector& vector, const DeviceVector& oneVector,
    PRECISION& sumHost){
  const cublasHandle_t& cublasHandle = stream.getCublasHandle();

#ifdef DOUBLEPRECISION
  cublasDdot(cublasHandle, vector.getNumberOfRows(), vector.getMemoryPointer(), 1, oneVector.getMemoryPointer(), 1,
      &sumHost);
#else
  cublasSdot(cublasHandle, vector.getNumberOfRows(), vector.getMemoryPointer(), 1, oneVector.getMemoryPointer(), 1,
      &sumHost);
#endif

#ifdef FERMI
  stream.syncStream();
#endif
}

void absoluteSumToHost(const Stream& stream, const DeviceVector& vector, PRECISION& sumHost){
  const cublasHandle_t& cublasHandle = stream.getCublasHandle();

#ifdef DOUBLEPRECISION
  cublasDasum(cublasHandle, vector.getNumberOfRows(), vector.getMemoryPointer(), 1, &sumHost);
#else
  cublasSasum(cublasHandle, vector.getNumberOfRows(), vector.getMemoryPointer(), 1, &sumHost);
#endif

#ifdef FERMI
  stream.syncStream();
#endif
}

} /* namespace Kernel */
} /* namespace CUDA */
} /* namespace CuEira */
