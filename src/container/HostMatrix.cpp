#include "HostMatrix.h"

namespace CuEira {
namespace Container {

HostMatrix::HostMatrix(unsigned int numberOfRows, unsigned int numberOfColumns, PRECISION* hostMatrix) :
    numberOfRows(numberOfRows), numberOfColumns(numberOfColumns), hostMatrix(hostMatrix) {
  if(numberOfRows < 0 || numberOfColumns < 0){
    std::ostringstream os;
    os << "Number of rows and columns for HostMatrix must be > 0" << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }
}

HostMatrix::~HostMatrix() {

}

int HostMatrix::getNumberOfRows() const {
  return numberOfRows;
}

int HostMatrix::getNumberOfColumns() const {
  return numberOfColumns;
}

PRECISION* HostMatrix::getMemoryPointer() {
  return hostMatrix;
}

const PRECISION* HostMatrix::getMemoryPointer() const {
  return hostMatrix;
}

} /* namespace Container */
} /* namespace CuEira */
