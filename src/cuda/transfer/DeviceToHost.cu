#include "DeviceToHost.h"

namespace CuEira {
namespace CUDA {

DeviceToHost::DeviceToHost(cublasHandle_t& cublasHandle) :
    cublasHandle(cublasHandle) {

}

DeviceToHost::~DeviceToHost() {

}

HostMatrix* DeviceToHost::transferMatrix(const DeviceMatrix* matrixDevice) const {
  const int numberOfRows = matrixDevice->getNumberOfRows();
  const int numberOfColumns = matrixDevice->getNumberOfColumns();

  cudaStream_t* cudaStream;
  handleCublasStatus(cublasGetStream(cublasHandle, cudaStream), "Failed to get cuda stream from cublas handle:");

  PinnedHostMatrix* hostMatrix = new PinnedHostMatrix(numberOfRows, numberOfColumns);
  PRECISION* matrixPointerHost = hostMatrix->getMemoryPointer();
  const PRECISION* matrixPointerDevice = matrixDevice->getMemoryPointer();

  handleCublasStatus(
      cublasGetMatrix(numberOfRows, numberOfColumns, sizeof(PRECISION), matrixPointerDevice, numberOfRows,
          matrixPointerHost, numberOfRows), "Error when transferring matrix from device to host: ");

  /*
   handleCublasStatus(
   cublasGetMatrixAsync(numberOfRows, numberOfColumns, sizeof(PRECISION), matrixPointerDevice, numberOfRows,
   matrixPointerHost, numberOfRows, *cudaStream), "Error when transferring matrix from device to host: ");
   */

  return hostMatrix;
}

HostVector* DeviceToHost::transferVector(const DeviceVector* vectorDevice) const {
  const int numberOfRows = vectorDevice->getNumberOfRows();

  cudaStream_t* cudaStream;
  handleCublasStatus(cublasGetStream(cublasHandle, cudaStream), "Failed to get cuda stream from cublas handle:");

  PinnedHostVector* hostVector = new PinnedHostVector(numberOfRows);
  PRECISION* matrixPointerHost = hostVector->getMemoryPointer();
  const PRECISION* vectorPointerDevice = vectorDevice->getMemoryPointer();

  handleCublasStatus(cublasGetVector(numberOfRows, sizeof(PRECISION), vectorPointerDevice, 1, matrixPointerHost, 1),
      "Error when transferring vector from device to host: ");

  /*
   handleCublasStatus(
   cublasGetVectorAsync(numberOfRows, sizeof(PRECISION), vectorPointerDevice, 1, matrixPointerHost, 1, *cudaStream),
   "Error when transferring vector from device to host: ");
   */

  return hostVector;
}

} /* namespace CUDA */
} /* namespace CuEira */
