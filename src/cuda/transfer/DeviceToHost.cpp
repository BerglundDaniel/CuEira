#include "DeviceToHost.h"

namespace CuEira {
namespace CUDA {

DeviceToHost::DeviceToHost(const cudaStream_t& cudaStream) :
    cudaStream(cudaStream) {

}

DeviceToHost::~DeviceToHost() {

}

HostMatrix* DeviceToHost::transferMatrix(const DeviceMatrix* matrixDevice) const {
  const int numberOfRows = matrixDevice->getNumberOfRows();
  const int numberOfColumns = matrixDevice->getNumberOfColumns();

  PinnedHostMatrix* hostMatrix = new PinnedHostMatrix(numberOfRows, numberOfColumns);
  PRECISION* matrixPointerHost = hostMatrix->getMemoryPointer();
  const PRECISION* matrixPointerDevice = matrixDevice->getMemoryPointer();

  handleCublasStatus(
      cublasGetMatrixAsync(numberOfRows, numberOfColumns, sizeof(PRECISION), matrixPointerDevice, numberOfRows,
          matrixPointerHost, numberOfRows, cudaStream), "Error when transferring matrix from device to host: ");

  return hostMatrix;
}

HostVector* DeviceToHost::transferVector(const DeviceVector* vectorDevice) const {
  const int numberOfRows = vectorDevice->getNumberOfRows();

  PinnedHostVector* hostVector = new PinnedHostVector(numberOfRows);
  PRECISION* vectorPointerHost = hostVector->getMemoryPointer();
  const PRECISION* vectorPointerDevice = vectorDevice->getMemoryPointer();

  handleCublasStatus(
      cublasGetVectorAsync(numberOfRows, sizeof(PRECISION), vectorPointerDevice, 1, vectorPointerHost, 1, cudaStream),
      "Error when transferring vector from device to host point: ");

  return hostVector;
}

void DeviceToHost::transferMatrix(const DeviceMatrix* matrixDevice, PRECISION* hostMemoryPosition) const {
  const int numberOfRows = matrixDevice->getNumberOfRows();
  const int numberOfColumns = matrixDevice->getNumberOfColumns();

  const PRECISION* matrixPointerDevice = matrixDevice->getMemoryPointer();

  handleCublasStatus(
      cublasGetMatrixAsync(numberOfRows, numberOfColumns, sizeof(PRECISION), matrixPointerDevice, numberOfRows,
          hostMemoryPosition, numberOfRows, cudaStream), "Error when transferring matrix from device to host: ");

}

void DeviceToHost::transferVector(const DeviceVector* vectorDevice, PRECISION* hostMemoryPosition) const {
  const int numberOfRows = vectorDevice->getNumberOfRows();

  const PRECISION* vectorPointerDevice = vectorDevice->getMemoryPointer();

  handleCublasStatus(
      cublasGetVectorAsync(numberOfRows, sizeof(PRECISION), vectorPointerDevice, 1, hostMemoryPosition, 1, cudaStream),
      "Error when transferring vector from device to host point: ");
}

} /* namespace CUDA */
} /* namespace CuEira */
