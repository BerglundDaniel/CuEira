#include "DeviceMatrix.h"

namespace CuEira {
namespace Container {

DeviceMatrix::DeviceMatrix(int numberOfRows, int numberOfColumns) :
    numberOfRows(numberOfRows), numberOfColumns(numberOfColumns), matrixDevice(new PRECISION()), subview(false) {
  CuEira::CUDA::allocateDeviceMemory((void**) &matrixDevice, numberOfRows * numberOfColumns);
}

DeviceMatrix::DeviceMatrix(int numberOfRows, int numberOfColumns, PRECISION* matrixDevice) :
    numberOfRows(numberOfRows), numberOfColumns(numberOfColumns), matrixDevice(matrixDevice), subview(true) {

}

DeviceMatrix::~DeviceMatrix() {
  if(!subview){
    CuEira::CUDA::freeDeviceMemory(matrixDevice);
  }
}

__device__ __host__ int DeviceMatrix::getNumberOfRows() const {
  return numberOfRows;
}

__device__ __host__ int DeviceMatrix::getNumberOfColumns() const {
  return numberOfColumns;
}

DeviceVector* DeviceMatrix::operator()(unsigned int column) {
  return new DeviceVector(numberOfRows, matrixDevice + (numberOfRows * column));
}

const DeviceVector* DeviceMatrix::operator()(unsigned int column) const {
  return new DeviceVector(numberOfRows, matrixDevice + (numberOfRows * column));
}

__device__ __host__ PRECISION* DeviceMatrix::operator()(unsigned int row, unsigned int column) {
  return matrixDevice + (numberOfRows * column) + row;
}

__device__ __host__ const PRECISION* DeviceMatrix::operator()(unsigned int row, unsigned int column) const {
  return matrixDevice + (numberOfRows * column) + row;
}

__device__ __host__ PRECISION* DeviceMatrix::getMemoryPointer() {
  return matrixDevice;
}

__device__ __host__ const PRECISION* DeviceMatrix::getMemoryPointer() const {
  return matrixDevice;
}

} /* namespace Container */
} /* namespace CuEira */
