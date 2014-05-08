#include "HostMatrix.h"

namespace CuEira {
namespace Container {

HostMatrix::HostMatrix(unsigned int numberOfRows, unsigned int numberOfColums, PRECISION* hostMatrix) :
    numberOfRows(numberOfRows), numberOfColumns(numberOfColums), hostMatrix(hostMatrix) {

}

HostMatrix::~HostMatrix() {

}

int HostMatrix::getNumberOfRows() {
  return numberOfRows;
}

int HostMatrix::getNumberOfColumns() {
  return numberOfColumns;
}

PRECISION* HostMatrix::getMemoryPointer() {
  return hostMatrix;
}

} /* namespace Container */
} /* namespace CuEira */
