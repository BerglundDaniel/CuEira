#include "PinnedHostMatrix.h"

namespace CuEira {
namespace Container {

PinnedHostMatrix::PinnedHostMatrix(int numberOfRows, int numberOfColumns) :
    HostMatrix(numberOfRows, numberOfColumns, nullptr) {
  CuEira::CUDA::allocateHostPinnedMemory((void**) &hostMatrix, numberOfRows * numberOfColumns);
}

PinnedHostMatrix::~PinnedHostMatrix() {
  CuEira::CUDA::freePinnedMemory(hostMatrix);
}

PinnedHostVector* PinnedHostMatrix::operator()(int column) {
  if(column >= numberOfColumns || column < 0){
    std::ostringstream os;
    os << "Index " << column << " is larger than the number of columns " << numberOfColumns << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }

  PRECISION* hostVector = hostMatrix + numberOfRows * column;
  return new PinnedHostVector(numberOfRows, hostVector, true);
}

const PinnedHostVector* PinnedHostMatrix::operator()(int column) const {
  if(column >= numberOfColumns || column < 0){
    std::ostringstream os;
    os << "Index " << column << " is larger than the number of columns " << numberOfColumns << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }

  PRECISION* hostVector = hostMatrix + numberOfRows * column;
  return new PinnedHostVector(numberOfRows, hostVector, true);
}

PRECISION& PinnedHostMatrix::operator()(int row, int column) {
  if(row >= numberOfRows || row < 0){
    std::ostringstream os;
    os << "Index " << row << " is larger than the number of rows " << numberOfRows << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }
  if(column >= numberOfColumns || column < 0){
    std::ostringstream os;
    os << "Index" << column << " is larger than the number of columns " << numberOfColumns << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }

  return *(hostMatrix + (numberOfRows * column) + row);
}

const PRECISION& PinnedHostMatrix::operator()(int row, int column) const {
  if(row >= numberOfRows || row < 0){
    std::ostringstream os;
    os << "Index " << row << " is larger than the number of rows " << numberOfRows << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }
  if(column >= numberOfColumns || column < 0){
    std::ostringstream os;
    os << "Index " << column << " is larger than the number of columns " << numberOfColumns << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }

  return *(hostMatrix + (numberOfRows * column) + row);
}

} /* namespace Container */
} /* namespace CuEira */
