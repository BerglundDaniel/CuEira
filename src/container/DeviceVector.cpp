#include "DeviceVector.h"

namespace CuEira {
namespace Container {

DeviceVector::DeviceVector(int numberOfRows) :
    numberOfRealRows(ceil(((double) numberOfRows) / GPU_UNROLL) * GPU_UNROLL), numberOfRows(numberOfRows), subview(
        false), vectorDevice(nullptr) {
  if(numberOfRows <= 0 || numberOfRealRows <= 0){
    throw DimensionMismatch("Number of rows for DeviceVector must be > 0");
  }

  CuEira::CUDA::allocateDeviceMemory((void**) &vectorDevice, numberOfRealRows);
}

DeviceVector::DeviceVector(int numberOfRealRows, int numberOfRows, PRECISION* vectorDevice) :
    numberOfRealRows(numberOfRealRows), numberOfRows(numberOfRows), subview(true), vectorDevice(vectorDevice) {
  if(numberOfRows <= 0 || numberOfRealRows <= 0){
    throw DimensionMismatch("Number of rows for DeviceVector must be > 0");
  }
}

DeviceVector::~DeviceVector() {
  if(!subview){
    CuEira::CUDA::freeDeviceMemory(vectorDevice);
  }
}

__device__ __host__ int DeviceVector::getNumberOfRows() const{
  return numberOfRows;
}

__device__ __host__ int DeviceVector::getNumberOfColumns() const{
  return 1;
}

__device__ __host__ PRECISION* DeviceVector::operator()(int row){
  return vectorDevice + row;
}

__device__ __host__ const PRECISION* DeviceVector::operator()(int row) const{
  return vectorDevice + row;
}

__device__ __host__ int DeviceVector::getRealNumberOfRows() const{
  return numberOfRealRows;
}

__device__ __host__ int DeviceVector::getRealNumberOfColumns() const{
  return 1;
}

__host__ void DeviceVector::updateSize(int numberOfRows){
#ifdef DEBUG
  if(numberOfRows > numberOfRealRows){
    throw DimensionMismatch("Number of rows for DeviceVector can't be larger than the real number of rows.");
  }
#endif

  this->numberOfRows = numberOfRows;
}

__device__ __host__ PRECISION* DeviceVector::getMemoryPointer(){
  return vectorDevice;
}

__device__ __host__ const PRECISION* DeviceVector::getMemoryPointer() const{
  return vectorDevice;
}

}
/* namespace Container */
} /* namespace CuEira */
