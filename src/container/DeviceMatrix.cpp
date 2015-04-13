#include "DeviceMatrix.h"

namespace CuEira {
namespace Container {

DeviceMatrix::DeviceMatrix(int numberOfRows, int numberOfColumns) :
    numberOfRealRows(ceil(((double) numberOfRows) / GPU_UNROLL) * GPU_UNROLL), numberOfRealColumns(numberOfColumns), numberOfRows(
        numberOfRows), numberOfColumns(numberOfColumns), matrixDevice(nullptr), subview(false) {
  if(numberOfRows <= 0 || numberOfColumns <= 0 || numberOfRealRows <= 0 || numberOfRealColumns <= 0){
    throw DimensionMismatch("Number of rows and columns for DeviceMatrix must be > 0");
  }

  CuEira::CUDA::allocateDeviceMemory((void**) &matrixDevice, numberOfRealRows * numberOfRealColumns);
}

DeviceMatrix::~DeviceMatrix() {
  CuEira::CUDA::freeDeviceMemory(matrixDevice);
}

__device__ __host__ int DeviceMatrix::getNumberOfRows() const{
  return numberOfRows;
}

__device__ __host__ int DeviceMatrix::getNumberOfColumns() const{
  return numberOfColumns;
}

__host__ DeviceVector* DeviceMatrix::operator()(int column){
  return new DeviceVector(numberOfRows, matrixDevice + (numberOfRows * column));
}

__host__ const DeviceVector* DeviceMatrix::operator()(int column) const{
  return new DeviceVector(numberOfRows, matrixDevice + (numberOfRows * column));
}

__device__ __host__ PRECISION* DeviceMatrix::operator()(int row, int column){
  return matrixDevice + (numberOfRows * column) + row;
}

__device__ __host__ const PRECISION* DeviceMatrix::operator()(int row, int column) const{
  return matrixDevice + (numberOfRows * column) + row;
}

__device__ __host__ int DeviceMatrix::getRealNumberOfRows() const{
  return numberOfRealRows;
}

__device__ __host__ int DeviceMatrix::getRealNumberOfColumns() const{
  return numberOfRealColumns;
}

__host__ void DeviceMatrix::updateSize(int numberOfRows, int numberOfColumns){
#ifdef DEBUG
  if(numberOfRows > numberOfRealRows){
    throw DimensionMismatch("Number of rows for DeviceMatrix can't be larger than the real number of rows.");
  }
  if(numberOfColumns > numberOfRealColumns){
    throw DimensionMismatch("Number of columns for DeviceMatrix can't be larger than the real number of columns.");
  }
#endif

  this->numberOfRows = numberOfRows;
  this->numberOfColumns = numberOfColumns;
}

__host__ void DeviceMatrix::updateNumberOfRows(int numberOfRows){
  updateSize(numberOfRows, numberOfColumns);
}

__host__ void DeviceMatrix::updateNumberOfColumns(int numberOfColumns){
  updateSize(numberOfRows, numberOfColumns);
}

__device__ __host__ PRECISION* DeviceMatrix::getMemoryPointer(){
  return matrixDevice;
}

__device__ __host__ const PRECISION* DeviceMatrix::getMemoryPointer() const{
  return matrixDevice;
}

}
/* namespace Container */
} /* namespace CuEira */
