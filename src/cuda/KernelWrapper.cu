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
#include <CalculateNumberOfAllelesPerGenotype.cuh>
#include <CalculateContingencyTable.cuh>

namespace CuEira {
namespace CUDA {
namespace Kernel {

void logisticTransform(const Stream& stream, const DeviceVector& logitVector, DeviceVector& probabilites){
#ifdef DEBUG
  if(logitVector.getNumberOfRows() != probabilites.getNumberOfRows()){
    std::ostringstream os;
    os << "Number of rows doesn't match in logistic transform function, they are " << logitVector.getNumberOfRows()
    << " and " << probabilites.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const cudaStream_t& cudaStream = stream.getCudaStream();
  const int numberOfBlocks = std::ceil(((double) logitVector.getNumberOfRows()) / numberOfThreadsPerBlock);
  LogisticTransform<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(logitVector.getMemoryPointer(),
      probabilites.getMemoryPointer(), logitVector.getNumberOfRows());

#ifdef FERMI
  stream.syncStream();
#endif
}

void elementWiseDivision(const Stream& stream, const DeviceVector& numeratorVector, const DeviceVector& denomitorVector,
    DeviceVector& result){
#ifdef DEBUG
  if((numeratorVector.getNumberOfRows() != denomitorVector.getNumberOfRows()) || ( numeratorVector.getNumberOfRows() != result.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in elementWiseDivision function, they are " << numeratorVector.getNumberOfRows()
    << " , " << denomitorVector.getNumberOfRows() << " and " << result.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const cudaStream_t& cudaStream = stream.getCudaStream();
  const int numberOfBlocks = std::ceil(((double) numeratorVector.getNumberOfRows()) / numberOfThreadsPerBlock);
  ElementWiseDivision<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(numeratorVector.getMemoryPointer(),
      denomitorVector.getMemoryPointer(), result.getMemoryPointer(), numeratorVector.getNumberOfRows());

#ifdef FERMI
  stream.syncStream();
#endif
}

void elementWiseAddition(const Stream& stream, const DeviceVector& vector1, const DeviceVector& vector2,
    DeviceVector& result){
#ifdef DEBUG
  if((vector1.getNumberOfRows() != vector2.getNumberOfRows()) || (vector1.getNumberOfRows() != result.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in elementWiseAddition function, they are " << vector1.getNumberOfRows()
    << " , " << vector2.getNumberOfRows() << " and " << result.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const cudaStream_t& cudaStream = stream.getCudaStream();
  const int numberOfBlocks = std::ceil(((double) vector1.getNumberOfRows()) / numberOfThreadsPerBlock);
  ElementWiseAddition<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(vector1.getMemoryPointer(),
      vector2.getMemoryPointer(), result.getMemoryPointer(), vector1.getNumberOfRows());

#ifdef FERMI
  stream.syncStream();
#endif
}

void elementWiseMultiplication(const Stream& stream, const DeviceVector& vector1, const DeviceVector& vector2,
    DeviceVector& result){
#ifdef DEBUG
  if((vector1.getNumberOfRows() != vector2.getNumberOfRows()) || (vector1.getNumberOfRows() != result.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in elementWiseMultiplication function, they are " << vector1.getNumberOfRows()
    << " , " << vector2.getNumberOfRows() << " and " << result.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const cudaStream_t& cudaStream = stream.getCudaStream();
  const int numberOfBlocks = std::ceil(((double) vector1.getNumberOfRows()) / numberOfThreadsPerBlock);
  ElementWiseMultiplication<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(vector1.getMemoryPointer(),
      vector2.getMemoryPointer(), result.getMemoryPointer(), vector1.getNumberOfRows());

#ifdef FERMI
  stream.syncStream();
#endif
}

void logLikelihoodParts(const Stream& stream, const DeviceVector& outcomesVector, const DeviceVector& probabilites,
    DeviceVector& result){
#ifdef DEBUG
  if((outcomesVector.getNumberOfRows() != probabilites.getNumberOfRows()) || (outcomesVector.getNumberOfRows() != result.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in logLikelihoodParts function, they are " << outcomesVector.getNumberOfRows()
    << " , " << probabilites.getNumberOfRows() << " and " << result.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const cudaStream_t& cudaStream = stream.getCudaStream();
  const int numberOfBlocks = std::ceil(((double) outcomesVector.getNumberOfRows()) / numberOfThreadsPerBlock);
  LogLikelihoodParts<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(outcomesVector.getMemoryPointer(),
      probabilites.getMemoryPointer(), result.getMemoryPointer(), outcomesVector.getNumberOfRows());

#ifdef FERMI
  stream.syncStream();
#endif
}

void elementWiseAbsoluteDifference(const Stream& stream, const DeviceVector& vector1, const DeviceVector& vector2,
    DeviceVector& result){
#ifdef DEBUG
  if((vector1.getNumberOfRows() != vector2.getNumberOfRows()) || (vector1.getNumberOfRows() != result.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in absoluteDifference function, they are " << vector1.getNumberOfRows()
    << " , " << vector2.getNumberOfRows() << " and " << result.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const cudaStream_t& cudaStream = stream.getCudaStream();
  const int numberOfBlocks = std::ceil(((double) vector1.getNumberOfRows()) / numberOfThreadsPerBlock);
  ElementWiseAbsoluteDifference<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(vector1.getMemoryPointer(),
      vector2.getMemoryPointer(), result.getMemoryPointer(), vector1.getNumberOfRows());

#ifdef FERMI
  stream.syncStream();
#endif
}

void probabilitesMultiplyProbabilites(const Stream& stream, const DeviceVector& probabilitesDevice,
    DeviceVector& result){
#ifdef DEBUG
  if(probabilitesDevice.getNumberOfRows() != result.getNumberOfRows()){
    std::ostringstream os;
    os << "Number of rows doesn't match in probabilitesMultiplyProbabilites function, they are " << probabilitesDevice.getNumberOfRows()
    << " and " << result.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const cudaStream_t& cudaStream = stream.getCudaStream();
  const int numberOfBlocks = std::ceil(((double) probabilitesDevice.getNumberOfRows()) / numberOfThreadsPerBlock);
  VectorMultiply1MinusVector<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(
      probabilitesDevice.getMemoryPointer(), result.getMemoryPointer(), probabilitesDevice.getNumberOfRows());

#ifdef FERMI
  stream.syncStream();
#endif
}

void elementWiseDifference(const Stream& stream, const DeviceVector& vector1, const DeviceVector& vector2,
    DeviceVector& result){
#ifdef DEBUG
  if((vector1.getNumberOfRows() != vector2.getNumberOfRows()) || (vector1.getNumberOfRows() != result.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in elementWiseDifference function, they are " << vector1.getNumberOfRows()
    << " , " << vector2.getNumberOfRows() << " and " << result.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const cudaStream_t& cudaStream = stream.getCudaStream();
  const int numberOfBlocks = std::ceil(((double) vector1.getNumberOfRows()) / numberOfThreadsPerBlock);
  ElementWiseDifference<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(vector1.getMemoryPointer(),
      vector2.getMemoryPointer(), result.getMemoryPointer(), vector1.getNumberOfRows());

#ifdef FERMI
  stream.syncStream();
#endif
}

void vectorCopyIndexes(const Stream& stream, const DeviceVector& indexes, const DeviceVector& from, DeviceVector& to){
#ifdef DEBUG
  if((indexes.getNumberOfRows() != from.getNumberOfRows()) || (indexes.getNumberOfRows() != to.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in vectorCopyIndexes function, they are " << indexes.getNumberOfRows()
    << " , " << from.getNumberOfRows() << " and " << to.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const cudaStream_t& cudaStream = stream.getCudaStream();
  const int numberOfRows = indexes.getNumberOfRows();
  const int numberOfBlocks = std::ceil(((double) numberOfRows) / numberOfThreadsPerBlock);
  VectorCopyIndexes<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(indexes.getMemoryPointer(),
      from.getMemoryPointer(), to.getMemoryPointer(), numberOfRows);

#ifdef FERMI
  stream.syncStream();
#endif
}

void constSubtractVector(const Stream& stream, const int c, DeviceVector& vector){
  const cudaStream_t& cudaStream = stream.getCudaStream();
  const int numberOfRows = vector.getNumberOfRows();
  const int numberOfBlocks = std::ceil(((double) numberOfRows) / numberOfThreadsPerBlock);
  ConstSubtractVector<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(c, vector.getMemoryPointer(),
      numberOfRows);

#ifdef FERMI
  stream.syncStream();
#endif
}

void applyGeneticModel(const Stream& stream, const int snpToRisk[3], const DeviceVector& from, DeviceVector& to){
#ifdef DEBUG
  if(from.getNumberOfRows() != to.getNumberOfRows()){
    std::ostringstream os;
    os << "Number of rows doesn't match in applyGeneticModel function, they are " << from.getNumberOfRows()
    << " and " << to.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const cudaStream_t& cudaStream = stream.getCudaStream();
  const int numberOfRows = from.getNumberOfRows();
  const int numberOfBlocks = std::ceil(((double) numberOfRows) / numberOfThreadsPerBlock);
  ApplyGeneticModel<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(snpToRisk, from.getMemoryPointer(),
      to.getMemoryPointer(), numberOfRows);

#ifdef FERMI
  stream.syncStream();
#endif
}

void applyAdditiveModel(const Stream& stream, DeviceVector& vector1, DeviceVector& vector2, DeviceVector& interaction){
#ifdef DEBUG
  if((vector1.getNumberOfRows() != vector2.getNumberOfRows()) || (vector1.getNumberOfRows() != interaction.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in applyAdditiveModel function, they are " << vector1.getNumberOfRows()
    << " , " << vector2.getNumberOfRows() << " and " << interaction.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const cudaStream_t& cudaStream = stream.getCudaStream();
  const int numberOfRows = vector1.getNumberOfRows();
  const int numberOfBlocks = std::ceil(((double) numberOfRows) / numberOfThreadsPerBlock);
  ApplyAdditiveModel<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(vector1.getMemoryPointer(),
      vector2.getMemoryPointer(), interaction.getMemoryPointer(), numberOfRows);

#ifdef FERMI
  stream.syncStream();
#endif
}

Container::DeviceMatrix* calculateNumberOfAllelesPerGenotype(const Stream& stream,
    const Container::DeviceVector& snpData, const Container::DeviceVector& phenotypeData){
#ifdef DEBUG
  if(snpData.getNumberOfRows() != phenotypeData.getNumberOfRows()){
    std::ostringstream os;
    os << "Number of rows doesn't match in calculateNumberOfAllelesPerGenotype function, they are "
    << snpData.getNumberOfRows() << " and " << phenotypeData.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const cudaStream_t& cudaStream = stream.getCudaStream();
  const int numberOfRows = snpData.getNumberOfRows();
  const int numberOfBlocks = std::ceil(((double) numberOfRows) / numberOfThreadsPerBlock);
  Container::DeviceMatrix* numberOfAllelesPerGenotype = new Container::DeviceMatrix(numberOfBlocks, 6);

  CalculateNumberOfAllelesPerGenotype<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(
      snpData.getMemoryPointer(), phenotypeData.getMemoryPointer(), numberOfAllelesPerGenotype->getMemoryPointer(),
      numberOfRows, numberOfBlocks);

#ifdef FERMI
  stream.syncStream();
#endif

  return numberOfAllelesPerGenotype;
}

Container::DeviceMatrix* calculateContingencyTable(const Stream& stream, const Container::DeviceVector& snpData,
    const Container::DeviceVector& envData, const Container::DeviceVector& phenotypeData){
#ifdef DEBUG
  if(snpData.getNumberOfRows() != phenotypeData.getNumberOfRows() || snpData.getNumberOfRows() != envData.getNumberOfRows()){
    std::ostringstream os;
    os << "Number of rows doesn't match in calculateContingencyTable function, they are "
    << snpData.getNumberOfRows() << " , " << envData.getNumberOfRows() << " and " << phenotypeData.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const cudaStream_t& cudaStream = stream.getCudaStream();
  const int numberOfRows = snpData.getNumberOfRows();
  const int numberOfBlocks = std::ceil(((double) numberOfRows) / numberOfThreadsPerBlock);
  Container::DeviceMatrix* table = new Container::DeviceMatrix(numberOfBlocks, 8);

  CalculateContingencyTable<<<numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStream>>>(snpData.getMemoryPointer(),
      envData.getMemoryPointer(), phenotypeData.getMemoryPointer(), table->getMemoryPointer(), numberOfRows,
      numberOfBlocks);

#ifdef FERMI
  stream.syncStream();
#endif

  return table;
}

} /* namespace Kernel */
} /* namespace CUDA */
} /* namespace CuEira */
