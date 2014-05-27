#include "KernelWrapper.h"
#include <LogisticTransform.cuh>
#include <AbsoluteDifference.cuh>
#include <ElementWiseDivision.cuh>
#include <LogLikelihoodParts.cuh>

namespace CuEira {
namespace CUDA {
//__device__ int ASDF;

KernelWrapper::KernelWrapper(const cudaStream_t& cudaStream, const cublasHandle_t& cublasHandle) :
    cudaStream(cudaStream), cublasHandle(cublasHandle) {
  /*int numberOfRows = 10000;
   handleCudaStatus(cudaGetLastError(), "Asdf: ");
   cudaMemcpyToSymbol(ASDF, &numberOfRows, sizeof(int));
   handleCudaStatus(cudaGetLastError(), "Error in memcopy: ");*/
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

void KernelWrapper::absoluteDifference(const DeviceVector& vector1, const DeviceVector& vector2,
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
  Kernel::AbsoluteDifference<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(vector1.getMemoryPointer(), vector2.getMemoryPointer(), result.getMemoryPointer());
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

void KernelWrapper::probabilitesMultiplyProbabilites(const DeviceVector& probabilitesDevice,
    DeviceVector& result) const {

}

void KernelWrapper::elementWiseDifference(const DeviceVector& vector1, const DeviceVector& vector2,
    DeviceVector& result) const {

}

void KernelWrapper::matrixVectorMultiply(const DeviceMatrix& matrix, const DeviceVector& vector,
    DeviceVector& result) const {

}

void KernelWrapper::matrixTransVectorMultiply(const DeviceMatrix& matrix, const DeviceVector& vector,
    DeviceVector& result) const {

}

void KernelWrapper::matrixTransMatrixMultiply(const DeviceMatrix& matrix1, const DeviceMatrix& matrix2,
    DeviceMatrix& result) const {

}

void KernelWrapper::columnByColumnMatrixVectorMultiply(const DeviceMatrix& matrix, const DeviceVector& vector,
    DeviceMatrix& result) const {

}

void KernelWrapper::svd(const DeviceMatrix& matrix, DeviceMatrix& uSVD, DeviceVector& sigmaSVD,
    DeviceMatrix& vtSVD) const {

}

void KernelWrapper::elementWiseAddition(const DeviceVector& vector1, const DeviceVector& vector2,
    DeviceVector& result) const {

}

void KernelWrapper::sumResultToHost(const DeviceVector& vector, PRECISION* sumHost) const {

}

} /* namespace CUDA */
} /* namespace CuEira */
