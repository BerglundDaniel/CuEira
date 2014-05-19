#include "PinnedHostMatrix.h"

namespace CuEira {
namespace Container {

PinnedHostMatrix::PinnedHostMatrix(unsigned int numberOfRows, unsigned int numberOfColumns) :
    HostMatrix(numberOfRows, numberOfColumns, new PRECISION()) {
  CuEira::CUDA::allocateHostPinnedMemory((void**) &hostMatrix, numberOfRows * numberOfColumns);
}

PinnedHostMatrix::~PinnedHostMatrix() {
  CuEira::CUDA::freePinnedMemory(hostMatrix);
}

HostVector* PinnedHostMatrix::operator()(unsigned int column) {
  if(column >= numberOfColumns){
    std::ostringstream os;
    os << "Index" << column << " is larger than the number of columns " << numberOfColumns << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }

  PRECISION* hostVector = hostMatrix + numberOfRows * column;
  return new PinnedHostVector(numberOfRows, hostVector, true);
}

const HostVector* PinnedHostMatrix::operator()(unsigned int column) const {
  if(column >= numberOfColumns){
    std::ostringstream os;
    os << "Index" << column << " is larger than the number of columns " << numberOfColumns << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }

  PRECISION* hostVector = hostMatrix + numberOfRows * column;
  return new PinnedHostVector(numberOfRows, hostVector, true);
}

PRECISION& PinnedHostMatrix::operator()(unsigned int row, unsigned int column) {
  if(row >= numberOfRows){
    std::ostringstream os;
    os << "Index" << row << " is larger than the number of rows " << numberOfRows << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }
  if(column >= numberOfColumns){
    std::ostringstream os;
    os << "Index" << column << " is larger than the number of columns " << numberOfColumns << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }

  return *(hostMatrix + (numberOfRows * column) + row);
}

const PRECISION& PinnedHostMatrix::operator()(unsigned int row, unsigned int column) const {
  if(row >= numberOfRows){
    std::ostringstream os;
    os << "Index" << row << " is larger than the number of rows " << numberOfRows << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }
  if(column >= numberOfColumns){
    std::ostringstream os;
    os << "Index" << column << " is larger than the number of columns " << numberOfColumns << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }

  return *(hostMatrix + (numberOfRows * column) + row);
}

} /* namespace Container */
} /* namespace CuEira */
