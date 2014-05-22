#include "KernelWrapper.h"
#include <LogisticTransform.cuh>
#include <AbsoluteDifference.cuh>
#include <ElementWiseDivision.cuh>
#include <LogLikelihoodParts.cuh>

namespace CuEira {
namespace CUDA {
//__device__ int ASDF;

KernelWrapper::KernelWrapper(const cudaStream_t& cudaStream) :
    cudaStream(cudaStream) {
  /*int numberOfRows = 10000;
   handleCudaStatus(cudaGetLastError(), "Asdf: ");
   cudaMemcpyToSymbol(ASDF, &numberOfRows, sizeof(int));
   handleCudaStatus(cudaGetLastError(), "Error in memcopy: ");*/
}

KernelWrapper::~KernelWrapper() {

}

//TODO fix error checks for the length

void KernelWrapper::logisticTransform(const DeviceVector& logitVector, DeviceVector& probabilites) const {
#ifdef DEBUG
  if(logitVector.getNumberOfRows() != probabilites.getNumberOfRows()){
    throw CudaException("Number of rows doesn't match in logistic transform function.");
  }
#endif

  const int numberOfBlocks = std::ceil(((double) logitVector.getNumberOfRows()) / numberOfThreadsPerBlock);
  Kernel::LogisticTransform<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(logitVector.getMemoryPointer(), probabilites.getMemoryPointer());
}

void KernelWrapper::elementWiseDivision(const DeviceVector& numeratorVector, const DeviceVector& denomitorVector,
    DeviceVector& result) const {
#ifdef DEBUG
  if(numeratorVector.getNumberOfRows() != denomitorVector.getNumberOfRows() != result.getNumberOfRows()){
    throw CudaException("Number of rows doesn't match in logistic transform function.");
  }
#endif

  const int numberOfBlocks = std::ceil(((double) numeratorVector.getNumberOfRows()) / numberOfThreadsPerBlock);
  Kernel::ElementWiseDivision<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(numeratorVector.getMemoryPointer(), denomitorVector.getMemoryPointer(),
      result.getMemoryPointer());
}

void KernelWrapper::logLikelihoodParts(const DeviceVector& outcomesVector, const DeviceVector& probabilites,
    DeviceVector& result) const {
#ifdef DEBUG
  if(outcomesVector.getNumberOfRows() != probabilites.getNumberOfRows() != result.getNumberOfRows()){
    throw CudaException("Number of rows doesn't match in logistic transform function.");
  }
#endif

  const int numberOfBlocks = std::ceil(((double) outcomesVector.getNumberOfRows()) / numberOfThreadsPerBlock);
  Kernel::LogLikelihoodParts<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(outcomesVector.getMemoryPointer(), probabilites.getMemoryPointer(),
      result.getMemoryPointer());
}

void KernelWrapper::absoluteDifference(const DeviceVector& vector1, const DeviceVector& vector2,
    DeviceVector& result) const {
#ifdef DEBUG
  if(vector1.getNumberOfRows() != vector2.getNumberOfRows() != result.getNumberOfRows()){
    throw CudaException("Number of rows doesn't match in logistic transform function.");
  }
#endif

  const int numberOfBlocks = std::ceil(((double) vector1.getNumberOfRows()) / numberOfThreadsPerBlock);
  Kernel::AbsoluteDifference<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(vector1.getMemoryPointer(), vector2.getMemoryPointer(), result.getMemoryPointer());
}

} /* namespace CUDA */
} /* namespace CuEira */
