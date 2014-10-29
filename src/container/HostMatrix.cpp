#include "HostMatrix.h"

namespace CuEira {
namespace Container {

HostMatrix::HostMatrix(int numberOfRows, int numberOfColumns, PRECISION* hostMatrix) :
    numberOfRows(numberOfRows), numberOfColumns(numberOfColumns), hostMatrix(hostMatrix) {
  if(numberOfRows <= 0 || numberOfColumns <= 0){
    throw DimensionMismatch("Number of rows and columns for HostMatrix must be > 0");
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
