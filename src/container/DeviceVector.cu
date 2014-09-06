#include "DeviceVector.h"

namespace CuEira {
namespace Container {

DeviceVector::DeviceVector(int numberOfRows) :
    numberOfRows(numberOfRows), numberOfColumns(1), subview(false), vectorDevice(new PRECISION()) {
  if(numberOfRows < 0){
    std::ostringstream os;
    os << "Number of rows for DeviceVector must be > 0" << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }
  CuEira::CUDA::allocateDeviceMemory((void**) &vectorDevice, numberOfRows * numberOfColumns);
}

DeviceVector::DeviceVector(int numberOfRows, PRECISION* vectorDevice) :
    numberOfRows(numberOfRows), numberOfColumns(1), subview(true), vectorDevice(vectorDevice) {
  if(numberOfRows < 0){
    std::ostringstream os;
    os << "Number of rows for DeviceVector must be > 0" << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }
}

DeviceVector::~DeviceVector() {
  if(!subview){
    CuEira::CUDA::freeDeviceMemory(vectorDevice);
  }
}

__device__ __host__ int DeviceVector::getNumberOfRows() const {
  return numberOfRows;
}

__device__ __host__ int DeviceVector::getNumberOfColumns() const {
  return numberOfColumns;
}

__device__ __host__ PRECISION* DeviceVector::operator()(unsigned int row) {
  return vectorDevice + row;
}

__device__ __host__ const PRECISION* DeviceVector::operator()(unsigned int row) const {
  return vectorDevice + row;
}

__device__ __host__ PRECISION* DeviceVector::getMemoryPointer() {
  return vectorDevice;
}

__device__ __host__ const PRECISION* DeviceVector::getMemoryPointer() const {
  return vectorDevice;
}

} /* namespace Container */
} /* namespace CuEira */
