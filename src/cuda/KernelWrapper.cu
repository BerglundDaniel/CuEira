#include "KernelWrapper.h"

__constant__ int numberOfRowsDeviceConstant;
__constant__ int numberOfPredictorsDeviceConstant;

#include <LogisticTransform.cuh>
#include <ElementWiseAbsoluteDifference.cuh>
#include <LogLikelihoodParts.cuh>
#include <VectorMultiply1MinusVector.cuh>
#include <ElementWiseDivision.cuh>
#include <ElementWiseDifference.cuh>
#include <ElementWiseAddition.cuh>
#include <ElementWiseMultiplication.cuh>

namespace CuEira {
namespace CUDA {

KernelWrapper::KernelWrapper(const cudaStream_t& cudaStream, const cublasHandle_t& cublasHandle) :
    cudaStream(cudaStream), cublasHandle(cublasHandle) {

}

KernelWrapper::~KernelWrapper() {

}

void KernelWrapper::logisticTransform(const DeviceVector& logitVector, DeviceVector& probabilites) const {
#ifdef DEBUG
  if(logitVector.getNumberOfRows() != probabilites.getNumberOfRows()){
    std::ostringstream os;
    os << "Number of rows doesn't match in logistic transform function, they are " << logitVector.getNumberOfRows()
    << " and " << probabilites.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const int numberOfBlocks = std::ceil(((double) logitVector.getNumberOfRows()) / numberOfThreadsPerBlock);
  Kernel::LogisticTransform<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(logitVector.getMemoryPointer(), probabilites.getMemoryPointer());
}

void KernelWrapper::elementWiseDivision(const DeviceVector& numeratorVector, const DeviceVector& denomitorVector,
    DeviceVector& result) const {
#ifdef DEBUG
  if((numeratorVector.getNumberOfRows() != denomitorVector.getNumberOfRows()) || ( numeratorVector.getNumberOfRows() != result.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in elementWiseDivision function, they are " << numeratorVector.getNumberOfRows()
    << " , " << denomitorVector.getNumberOfRows() << " and " << result.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const int numberOfBlocks = std::ceil(((double) numeratorVector.getNumberOfRows()) / numberOfThreadsPerBlock);
  Kernel::ElementWiseDivision<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(numeratorVector.getMemoryPointer(), denomitorVector.getMemoryPointer(),
      result.getMemoryPointer());
}

void KernelWrapper::logLikelihoodParts(const DeviceVector& outcomesVector, const DeviceVector& probabilites,
    DeviceVector& result) const {
#ifdef DEBUG
  if((outcomesVector.getNumberOfRows() != probabilites.getNumberOfRows()) || (outcomesVector.getNumberOfRows() != result.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in logLikelihoodParts function, they are " << outcomesVector.getNumberOfRows()
    << " , " << probabilites.getNumberOfRows() << " and " << result.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const int numberOfBlocks = std::ceil(((double) outcomesVector.getNumberOfRows()) / numberOfThreadsPerBlock);
  Kernel::LogLikelihoodParts<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(outcomesVector.getMemoryPointer(), probabilites.getMemoryPointer(),
      result.getMemoryPointer());
}

void KernelWrapper::elementWiseAbsoluteDifference(const DeviceVector& vector1, const DeviceVector& vector2,
    DeviceVector& result) const {
#ifdef DEBUG
  if((vector1.getNumberOfRows() != vector2.getNumberOfRows()) || (vector1.getNumberOfRows() != result.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in absoluteDifference function, they are " << vector1.getNumberOfRows()
    << " , " << vector2.getNumberOfRows() << " and " << result.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const int numberOfBlocks = std::ceil(((double) vector1.getNumberOfRows()) / numberOfThreadsPerBlock);
  Kernel::ElementWiseAbsoluteDifference<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(vector1.getMemoryPointer(), vector2.getMemoryPointer(), result.getMemoryPointer());
}

void KernelWrapper::copyVector(const DeviceVector& vectorFrom, DeviceVector& vectorTo) const {
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
}

void KernelWrapper::svd(const DeviceMatrix& matrix, DeviceMatrix& uSVD, DeviceVector& sigmaSVD,
    DeviceMatrix& vtSVD) const {
#ifdef DEBUG
  if((matrix.getNumberOfRows() != uSVD.getNumberOfRows()) || (sigmaSVD.getNumberOfRows() != vtSVD.getNumberOfRows())
      || (matrix.getNumberOfRows() != sigmaSVD.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in svd function, they are " << matrix.getNumberOfRows() << " , "
    << uSVD.getNumberOfRows() << " , " << sigmaSVD.getNumberOfRows() << " and " << vtSVD.getNumberOfRows()
    << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }

  if((matrix.getNumberOfColumns() != uSVD.getNumberOfColumns())
      || (sigmaSVD.getNumberOfColumns() != vtSVD.getNumberOfColumns())
      || (matrix.getNumberOfColumns() != sigmaSVD.getNumberOfColumns())){
    std::ostringstream os;
    os << "Number of columns doesn't match in svd function, they are " << matrix.getNumberOfColumns() << " , "
    << uSVD.getNumberOfColumns() << " , " << sigmaSVD.getNumberOfColumns() << " and " << vtSVD.getNumberOfColumns()
    << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  //TODO do svd

}

void KernelWrapper::matrixTransRowByRowInverseSigma(const DeviceMatrix& matrix, const DeviceVector& sigma,
    DeviceMatrix& result) const {
  //FIXME check for square
#ifdef DEBUG
  if((matrix.getNumberOfRows() != sigma.getNumberOfRows()) || (matrix.getNumberOfRows() != result.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in matrixTransRowByRowInverseSigma function, they are " << matrix.getNumberOfRows() << " , "
    << sigma.getNumberOfRows() << " and " << result.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }

  if(matrix.getNumberOfColumns() != result.getNumberOfColumns()){
    std::ostringstream os;
    os << "Number of columns doesn't match in matrixTransRowByRowInverseSigma function, they are " << matrix.getNumberOfColumns() <<
    " and " << result.getNumberOfColumns() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  //TODO do a kernel call here
  //need a custom kernel

  //for each sigma number I, sigma_inv=1/sigma if sigma!=0 else sigma_inv=0
  //sigma_inv* each element k in col I from matrix
  //put res in res(i,k)

}

void KernelWrapper::setSymbolNumberOfRows(int numberOfRows) const {
  cudaMemcpyToSymbol(numberOfRowsDeviceConstant, &numberOfRows, sizeof(int));
  handleCudaStatus(cudaGetLastError(), "Error in memcopy to symbol numberOfRows : ");
}

void KernelWrapper::setSymbolNumberOfPredictors(int numberOfPredictors) const {
  cudaMemcpyToSymbol(numberOfPredictorsDeviceConstant, &numberOfPredictors, sizeof(int));
  handleCudaStatus(cudaGetLastError(), "Error in memcopy to symbol numberOfPredictors : ");
}

void KernelWrapper::probabilitesMultiplyProbabilites(const DeviceVector& probabilitesDevice,
    DeviceVector& result) const {
#ifdef DEBUG
  if(probabilitesDevice.getNumberOfRows() != result.getNumberOfRows()){
    std::ostringstream os;
    os << "Number of rows doesn't match in probabilitesMultiplyProbabilites function, they are " << probabilitesDevice.getNumberOfRows()
    << " and " << result.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const int numberOfBlocks = std::ceil(((double) probabilitesDevice.getNumberOfRows()) / numberOfThreadsPerBlock);
  Kernel::VectorMultiply1MinusVector<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(probabilitesDevice.getMemoryPointer(), result.getMemoryPointer());
}

void KernelWrapper::elementWiseDifference(const DeviceVector& vector1, const DeviceVector& vector2,
    DeviceVector& result) const {
#ifdef DEBUG
  if((vector1.getNumberOfRows() != vector2.getNumberOfRows()) || (vector1.getNumberOfRows() != result.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in elementWiseDifference function, they are " << vector1.getNumberOfRows()
    << " , " << vector2.getNumberOfRows() << " and " << result.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const int numberOfBlocks = std::ceil(((double) vector1.getNumberOfRows()) / numberOfThreadsPerBlock);
  Kernel::ElementWiseDifference<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(vector1.getMemoryPointer(), vector2.getMemoryPointer(), result.getMemoryPointer());
}

void KernelWrapper::matrixVectorMultiply(const DeviceMatrix& matrix, const DeviceVector& vector,
    DeviceVector& result) const {
#ifdef DEBUG
  if((matrix.getNumberOfRows() != vector.getNumberOfRows()) || (vector.getNumberOfRows() != result.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in matrixVectorMultiply function, they are " << matrix.getNumberOfRows()
    << " , " << vector.getNumberOfRows() << " and " << result.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const PRECISION* const1 = new PRECISION(1);
#ifdef DOUBLEPRECISION
  cublasDgemv(cublasHandle, CUBLAS_OP_N, matrix.getNumberOfRows(), matrix.getNumberOfColumns(), const1,
      matrix.getMemoryPointer(), matrix.getNumberOfRows(), vector.getMemoryPointer(), 1, const1, result.getMemoryPointer(),
      1);
#else
  cublasSgemv(cublasHandle, CUBLAS_OP_N, matrix.getNumberOfRows(), matrix.getNumberOfColumns(), const1,
      matrix.getMemoryPointer(), matrix.getNumberOfRows(), vector.getMemoryPointer(), 1, const1,
      result.getMemoryPointer(), 1);
#endif
}

void KernelWrapper::matrixTransVectorMultiply(const DeviceMatrix& matrix, const DeviceVector& vector,
    DeviceVector& result) const {
#ifdef DEBUG
  if((matrix.getNumberOfColumns() != vector.getNumberOfRows()) || (vector.getNumberOfRows() != result.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows(columns for matrix) doesn't match in matrixTransVectorMultiply function, they are " << matrix.getNumberOfColumns()
    << " , " << vector.getNumberOfRows() << " and " << result.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const PRECISION* const1 = new PRECISION(1);
#ifdef DOUBLEPRECISION
  cublasDgemv(cublasHandle, CUBLAS_OP_T, matrix.getNumberOfRows(), matrix.getNumberOfColumns(), const1,
      matrix.getMemoryPointer(), matrix.getNumberOfRows(), vector.getMemoryPointer(), 1, const1, result.getMemoryPointer(),
      1);
#else
  cublasSgemv(cublasHandle, CUBLAS_OP_T, matrix.getNumberOfRows(), matrix.getNumberOfColumns(), const1,
      matrix.getMemoryPointer(), matrix.getNumberOfRows(), vector.getMemoryPointer(), 1, const1,
      result.getMemoryPointer(), 1);
#endif
}

void KernelWrapper::matrixTransMatrixMultiply(const DeviceMatrix& matrix1, const DeviceMatrix& matrix2,
    DeviceMatrix& result) const {
#ifdef DEBUG
  if((matrix1.getNumberOfColumns() != matrix2.getNumberOfRows()) || (matrix2.getNumberOfRows() != result.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows(columns for the first matrix) doesn't match in matrixTransMatrixMultiply function, they are " << matrix1.getNumberOfColumns()
    << " , " << matrix2.getNumberOfRows() << " and " << result.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }

  if((matrix1.getNumberOfRows() != matrix2.getNumberOfColumns()) || (matrix1.getNumberOfColumns() != result.getNumberOfColumns())){
    std::ostringstream os;
    os << "Number of columns(rows for first matrix) doesn't match in matrixTransMatrixMultiply function, they are " << matrix1.getNumberOfRows()
    << " , " << matrix2.getNumberOfColumns() << " and " << result.getNumberOfColumns() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const PRECISION* const1 = new PRECISION(1);
#ifdef DOUBLEPRECISION
  cublasDgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, matrix1.getNumberOfColumns(), matrix2.getNumberOfColumns(),
      matrix1.getNumberOfRows(), const1, matrix1.getMemoryPointer(), matrix1.getNumberOfRows(),
      matrix2.getMemoryPointer(), matrix2.getNumberOfRows(), const1, result.getMemoryPointer(),
      result.getNumberOfRows());
#else
  cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, matrix1.getNumberOfColumns(), matrix2.getNumberOfColumns(),
      matrix1.getNumberOfRows(), const1, matrix1.getMemoryPointer(), matrix1.getNumberOfRows(),
      matrix2.getMemoryPointer(), matrix2.getNumberOfRows(), const1, result.getMemoryPointer(),
      result.getNumberOfRows());
#endif
}

void KernelWrapper::columnByColumnMatrixVectorElementWiseMultiply(const DeviceMatrix& matrix,
    const DeviceVector& vector, DeviceMatrix& result) const {
#ifdef DEBUG
  if((matrix.getNumberOfRows() != vector.getNumberOfRows()) || (vector.getNumberOfRows() != result.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in columnByColumnMatrixVectorElementWiseMultiply function, they are " << matrix.getNumberOfRows()
    << " , " << vector.getNumberOfRows() << " and " << result.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }

  if(matrix.getNumberOfColumns() != result.getNumberOfColumns()){
    std::ostringstream os;
    os << "Number of columns doesn't match in columnByColumnMatrixVectorElementWiseMultiply function, they are " << matrix.getNumberOfColumns() <<
    " and " << result.getNumberOfColumns() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const int numberOfColumns = matrix.getNumberOfColumns();
  for(int k = 0; k < numberOfColumns; ++k){
    const DeviceVector* columnVector = matrix(k);
    DeviceVector* columnResultVector = result(k);
    elementWiseMultiplication(*columnVector, vector, *columnResultVector);

    delete columnVector;
    delete columnResultVector;
  }
}

void KernelWrapper::elementWiseAddition(const DeviceVector& vector1, const DeviceVector& vector2,
    DeviceVector& result) const {
#ifdef DEBUG
  if((vector1.getNumberOfRows() != vector2.getNumberOfRows()) || (vector1.getNumberOfRows() != result.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in elementWiseAddition function, they are " << vector1.getNumberOfRows()
    << " , " << vector2.getNumberOfRows() << " and " << result.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const int numberOfBlocks = std::ceil(((double) vector1.getNumberOfRows()) / numberOfThreadsPerBlock);
  Kernel::ElementWiseAddition<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(vector1.getMemoryPointer(), vector2.getMemoryPointer(), result.getMemoryPointer());
}

void KernelWrapper::elementWiseMultiplication(const DeviceVector& vector1, const DeviceVector& vector2,
    DeviceVector& result) const {
#ifdef DEBUG
  if((vector1.getNumberOfRows() != vector2.getNumberOfRows()) || (vector1.getNumberOfRows() != result.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in elementWiseMultiplication function, they are " << vector1.getNumberOfRows()
    << " , " << vector2.getNumberOfRows() << " and " << result.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const int numberOfBlocks = std::ceil(((double) vector1.getNumberOfRows()) / numberOfThreadsPerBlock);
  Kernel::ElementWiseMultiplication<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(vector1.getMemoryPointer(), vector2.getMemoryPointer(), result.getMemoryPointer());
}

void KernelWrapper::sumResultToHost(const DeviceVector& vector, PRECISION* sumHost) const {
#ifdef DOUBLEPRECISION
  cublasDasum(cublasHandle, vector.getNumberOfRows(), vector.getMemoryPointer(), 1, sumHost);
#else
  cublasSasum(cublasHandle, vector.getNumberOfRows(), vector.getMemoryPointer(), 1, sumHost);
#endif
}

} /* namespace CUDA */
} /* namespace CuEira */
