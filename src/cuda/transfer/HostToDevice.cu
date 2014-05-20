#include "HostToDevice.h"

namespace CuEira {
namespace CUDA {

HostToDevice::HostToDevice(cublasHandle_t& cublasHandle) :
    cublasHandle(cublasHandle) {

}

HostToDevice::~HostToDevice() {

}

DeviceMatrix* HostToDevice::transferMatrix(const HostMatrix* matrixHost) const {
  const int numberOfRows = matrixHost->getNumberOfRows();
  const int numberOfColumns = matrixHost->getNumberOfColumns();

  cudaStream_t* cudaStream;
  handleCublasStatus(cublasGetStream(cublasHandle, cudaStream), "Failed to get cuda stream from cublas handle:");

  DeviceMatrix* deviceMatrix = new DeviceMatrix(numberOfRows, numberOfColumns);
  PRECISION* deviceMatrixPointer = deviceMatrix->getMemoryPointer();
  const PRECISION* hostMatrixPointer = matrixHost->getMemoryPointer();

  handleCublasStatus(
      cublasSetMatrixAsync(numberOfRows, numberOfColumns, sizeof(PRECISION), hostMatrixPointer, numberOfRows,
          deviceMatrixPointer, numberOfRows, *cudaStream), "Error when transferring matrix from host to device: ");

  return deviceMatrix;
}

DeviceVector* HostToDevice::transferVector(const HostVector* vectorHost) const {
  const int numberOfRows = vectorHost->getNumberOfRows();

  cudaStream_t* cudaStream;
  handleCublasStatus(cublasGetStream(cublasHandle, cudaStream), "Failed to get cuda stream from cublas handle:");

  DeviceVector* deviceVector = new DeviceVector(numberOfRows);
  PRECISION* deviceVectorPointer = deviceVector->getMemoryPointer();
  const PRECISION* hostVectorPointer = vectorHost->getMemoryPointer();

  handleCublasStatus(
      cublasSetVectorAsync(numberOfRows, sizeof(PRECISION), hostVectorPointer, 1, deviceVectorPointer, 1, *cudaStream),
      "Error when transferring vector from host to device: ");

  return deviceVector;
}

} /* namespace CUDA */
} /* namespace CuEira */
