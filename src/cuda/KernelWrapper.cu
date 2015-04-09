#include "KernelWrapper.h"

#include <LogisticTransform.cuh>
#include <ElementWiseAbsoluteDifference.cuh>
#include <LogLikelihoodParts.cuh>
#include <VectorMultiply1MinusVector.cuh>
#include <ElementWiseDivision.cuh>
#include <ElementWiseDifference.cuh>
#include <ElementWiseAddition.cuh>
#include <ElementWiseMultiplication.cuh>
#include <VectorCopyIndexes.cuh>
#include <ApplyGeneticModel.cuh>
#include <ApplyAdditiveModel.cuh>
#include <ConstSubtractVector.cuh>

namespace CuEira {
namespace CUDA {

KernelWrapper::KernelWrapper(const Stream& stream) :
    stream(stream), cudaStream(stream.getCudaStream()), cublasHandle(stream.getCublasHandle()), constOne(
        new PRECISION(1)), constZero(new PRECISION(0)) {

}

KernelWrapper::~KernelWrapper() {
  delete constOne;
  delete constZero;
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
  Kernel::LogisticTransform<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(logitVector.getMemoryPointer(), probabilites.getMemoryPointer(), logitVector.getNumberOfRows());

#ifdef FERMI
  syncStream();
#endif
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
      result.getMemoryPointer(), numeratorVector.getNumberOfRows());

#ifdef FERMI
  syncStream();
#endif
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
  Kernel::ElementWiseAddition<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(vector1.getMemoryPointer(), vector2.getMemoryPointer(), result.getMemoryPointer(), vector1.getNumberOfRows());

#ifdef FERMI
  syncStream();
#endif
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
  Kernel::ElementWiseMultiplication<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(vector1.getMemoryPointer(), vector2.getMemoryPointer(), result.getMemoryPointer(), vector1.getNumberOfRows());

#ifdef FERMI
  syncStream();
#endif
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
      result.getMemoryPointer(), outcomesVector.getNumberOfRows());

#ifdef FERMI
  syncStream();
#endif
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
  Kernel::ElementWiseAbsoluteDifference<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(vector1.getMemoryPointer(), vector2.getMemoryPointer(), result.getMemoryPointer(), vector1.getNumberOfRows());

#ifdef FERMI
  syncStream();
#endif
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
  Kernel::VectorMultiply1MinusVector<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(probabilitesDevice.getMemoryPointer(), result.getMemoryPointer(), probabilitesDevice.getNumberOfRows());

#ifdef FERMI
  syncStream();
#endif
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
  Kernel::ElementWiseDifference<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(vector1.getMemoryPointer(), vector2.getMemoryPointer(), result.getMemoryPointer(), vector1.getNumberOfRows());

#ifdef FERMI
  syncStream();
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

#ifdef FERMI
  syncStream();
#endif
}

void KernelWrapper::vectorCopyIndexes(const DeviceVector& indexes, const DeviceVector& from, DeviceVector& to) const {
#ifdef DEBUG
  if((indexes.getNumberOfRows() != from.getNumberOfRows()) || (indexes.getNumberOfRows() != to.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in vectorCopyIndexes function, they are " << indexes.getNumberOfRows()
    << " , " << from.getNumberOfRows() << " and " << to.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const int numberOfRows = indexes.getNumberOfRows();
  const int numberOfBlocks = std::ceil(((double) numberOfRows) / numberOfThreadsPerBlock);
  Kernel::VectorCopyIndexes<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(indexes.getMemoryPointer(), from.getMemoryPointer(), to.getMemoryPointer(), numberOfRows);
}

void KernelWrapper::constSubtractVector(const int c, DeviceVector& vector) const {
  const int numberOfRows = vector.getNumberOfRows();
  const int numberOfBlocks = std::ceil(((double) numberOfRows) / numberOfThreadsPerBlock);
  Kernel::ConstSubtractVector<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(c, vector.getMemoryPointer(), numberOfRows);
}

void KernelWrapper::applyGeneticModel(const int snpToRisk[3], const DeviceVector& from, DeviceVector& to) const {
#ifdef DEBUG
  if(from.getNumberOfRows() != to.getNumberOfRows()){
    std::ostringstream os;
    os << "Number of rows doesn't match in applyGeneticModel function, they are " << from.getNumberOfRows()
    << " and " << to.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const int numberOfRows = from.getNumberOfRows();
  const int numberOfBlocks = std::ceil(((double) numberOfRows) / numberOfThreadsPerBlock);
  Kernel::ApplyGeneticModel<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(snpToRisk, from.getMemoryPointer(), to.getMemoryPointer(),numberOfRows);
}

void KernelWrapper::applyAdditiveModel(DeviceVector& vector1, DeviceVector& vector2, DeviceVector& interaction) const {
#ifdef DEBUG
  if((vector1.getNumberOfRows() != vector2.getNumberOfRows()) || (vector1.getNumberOfRows() != interaction.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in applyAdditiveModel function, they are " << vector1.getNumberOfRows()
    << " , " << vector2.getNumberOfRows() << " and " << interaction.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const int numberOfRows = vector1.getNumberOfRows();
  const int numberOfBlocks = std::ceil(((double) numberOfRows) / numberOfThreadsPerBlock);
  Kernel::ApplyAdditiveModel<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(vector1.getMemoryPointer(), vector2.getMemoryPointer(), interaction.getMemoryPointer(), numberOfRows);
}

} /* namespace CUDA */
} /* namespace CuEira */
