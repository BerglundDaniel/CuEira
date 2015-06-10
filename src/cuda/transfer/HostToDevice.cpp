#include "HostToDevice.h"

namespace CuEira {
namespace CUDA {

DeviceMatrix* transferMatrix(const Stream& stream, const PinnedHostMatrix& matrixHost){
  const cudaStream_t& cudaStream = stream.getCudaStream();
  const int numberOfRows = matrixHost.getNumberOfRows();
  const int numberOfColumns = matrixHost.getNumberOfColumns();

  DeviceMatrix* deviceMatrix = new DeviceMatrix(numberOfRows, numberOfColumns);
  PRECISION* deviceMatrixPointer = deviceMatrix->getMemoryPointer();
  const PRECISION* hostMatrixPointer = matrixHost.getMemoryPointer();

  handleCublasStatus(
      cublasSetMatrixAsync(numberOfRows, numberOfColumns, sizeof(PRECISION), hostMatrixPointer, numberOfRows,
          deviceMatrixPointer, numberOfRows, cudaStream), "Error when transferring matrix from host to device: ");

  return deviceMatrix;
}

DeviceVector* transferVector(const Stream& stream, const PinnedHostVector& vectorHost){
  const cudaStream_t& cudaStream = stream.getCudaStream();
  const int numberOfRows = vectorHost.getNumberOfRows();

  DeviceVector* deviceVector = new DeviceVector(numberOfRows);
  PRECISION* deviceVectorPointer = deviceVector->getMemoryPointer();
  const PRECISION* hostVectorPointer = vectorHost.getMemoryPointer();

  handleCublasStatus(
      cublasSetVectorAsync(numberOfRows, sizeof(PRECISION), hostVectorPointer, 1, deviceVectorPointer, 1, cudaStream),
      "Error when transferring vector from host to device: ");

  return deviceVector;
}

void transferMatrix(const Stream& stream, const PinnedHostMatrix& matrixHost, PRECISION* deviceMemoryPosition){
  const cudaStream_t& cudaStream = stream.getCudaStream();
  const int numberOfRows = matrixHost.getNumberOfRows();
  const int numberOfColumns = matrixHost.getNumberOfColumns();
  const PRECISION* hostMatrixPointer = matrixHost.getMemoryPointer();

  handleCublasStatus(
      cublasSetMatrixAsync(numberOfRows, numberOfColumns, sizeof(PRECISION), hostMatrixPointer, numberOfRows,
          deviceMemoryPosition, numberOfRows, cudaStream),
      "Error when transferring matrix from host to device point: ");
}

void transferVector(const Stream& stream, const PinnedHostVector& vectorHost, PRECISION* deviceMemoryPosition){
  const cudaStream_t& cudaStream = stream.getCudaStream();
  const int numberOfRows = vectorHost.getNumberOfRows();
  const PRECISION* hostVectorPointer = vectorHost.getMemoryPointer();

  handleCublasStatus(
      cublasSetVectorAsync(numberOfRows, sizeof(PRECISION), hostVectorPointer, 1, deviceMemoryPosition, 1, cudaStream),
      "Error when transferring vector from host to device point: ");
}

} /* namespace CUDA */
} /* namespace CuEira */
