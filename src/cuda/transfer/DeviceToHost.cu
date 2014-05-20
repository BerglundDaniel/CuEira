#include "DeviceToHost.h"

namespace CuEira {
namespace CUDA {

DeviceToHost::DeviceToHost(cublasHandle_t cublasHandle, cudaStream_t cudaStream) :
    cublasHandle(cublasHandle), cudaStream(cudaStream) {

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
  PRECISION* matrixPointerHost = hostVector->getMemoryPointer();
  const PRECISION* vectorPointerDevice = vectorDevice->getMemoryPointer();

  handleCublasStatus(
      cublasGetVectorAsync(1, sizeof(PRECISION), vectorPointerDevice, 1, matrixPointerHost, 1, cudaStream),
      "Error when transferring vector from device to host: ");

  return hostVector;
}

} /* namespace CUDA */
} /* namespace CuEira */
