#include "HostToDevice.h"

namespace CuEira {
namespace CUDA {

HostToDevice::HostToDevice(const Stream& stream) :
    cudaStream(&stream.getCudaStream()) {

}

HostToDevice::HostToDevice() :
    cudaStream(nullptr) {
}

HostToDevice::~HostToDevice() {

}

DeviceMatrix* HostToDevice::transferMatrix(const PinnedHostMatrix& matrixHost) const {
  const int numberOfRows = matrixHost.getNumberOfRows();
  const int numberOfColumns = matrixHost.getNumberOfColumns();

  DeviceMatrix* deviceMatrix = new DeviceMatrix(numberOfRows, numberOfColumns);
  PRECISION* deviceMatrixPointer = deviceMatrix->getMemoryPointer();
  const PRECISION* hostMatrixPointer = matrixHost.getMemoryPointer();

  handleCublasStatus(
      cublasSetMatrixAsync(numberOfRows, numberOfColumns, sizeof(PRECISION), hostMatrixPointer, numberOfRows,
          deviceMatrixPointer, numberOfRows, *cudaStream), "Error when transferring matrix from host to device: ");

  return deviceMatrix;
}

DeviceVector* HostToDevice::transferVector(const PinnedHostVector& vectorHost) const {
  const int numberOfRows = vectorHost.getNumberOfRows();

  DeviceVector* deviceVector = new DeviceVector(numberOfRows);
  PRECISION* deviceVectorPointer = deviceVector->getMemoryPointer();
  const PRECISION* hostVectorPointer = vectorHost.getMemoryPointer();

  handleCublasStatus(
      cublasSetVectorAsync(numberOfRows, sizeof(PRECISION), hostVectorPointer, 1, deviceVectorPointer, 1, *cudaStream),
      "Error when transferring vector from host to device: ");

  return deviceVector;
}

void HostToDevice::transferMatrix(const PinnedHostMatrix& matrixHost, PRECISION* deviceMemoryPosition) const {
  const int numberOfRows = matrixHost.getNumberOfRows();
  const int numberOfColumns = matrixHost.getNumberOfColumns();
  const PRECISION* hostMatrixPointer = matrixHost.getMemoryPointer();

  handleCublasStatus(
      cublasSetMatrixAsync(numberOfRows, numberOfColumns, sizeof(PRECISION), hostMatrixPointer, numberOfRows,
          deviceMemoryPosition, numberOfRows, *cudaStream),
      "Error when transferring matrix from host to device point: ");
}

void HostToDevice::transferVector(const PinnedHostVector& vectorHost, PRECISION* deviceMemoryPosition) const {
  const int numberOfRows = vectorHost.getNumberOfRows();
  const PRECISION* hostVectorPointer = vectorHost.getMemoryPointer();

  handleCublasStatus(
      cublasSetVectorAsync(numberOfRows, sizeof(PRECISION), hostVectorPointer, 1, deviceMemoryPosition, 1, *cudaStream),
      "Error when transferring vector from host to device point: ");
}

} /* namespace CUDA */
} /* namespace CuEira */
