#include "HostMatrix.h"

namespace CuEira {
namespace Container {

HostMatrix::HostMatrix(int numberOfRealRows, int numberOfRealColumns, int numberOfRows, int numberOfColumns,
    PRECISION* hostMatrix) :
    numberOfRealRows(numberOfRealRows), numberOfRealColumns(numberOfRealColumns), numberOfRows(numberOfRows), numberOfColumns(
        numberOfColumns), hostMatrix(hostMatrix) {
  if(numberOfRows <= 0 || numberOfColumns <= 0 || numberOfRealRows <= 0 || numberOfRealColumns <= 0){
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

int HostMatrix::getRealNumberOfRows() const {
  return numberOfRealRows;
}

int HostMatrix::getRealNumberOfColumns() const {
  return numberOfRealColumns;
}

void HostMatrix::updateSize(int numberOfRows, int numberOfColumns) {
#ifdef DEBUG
  if(numberOfRows > numberOfRealRows){
    throw DimensionMismatch("Number of rows for HostMatrix can't be larger than the real number of rows.");
  }
  if(numberOfColumns > numberOfRealColumns){
    throw DimensionMismatch("Number of columns for HostMatrix can't be larger than the real number of columns.");
  }
#endif

  this->numberOfRows = numberOfRows;
  this->numberOfColumns = numberOfColumns;
}

void HostMatrix::updateNumberOfRows(int numberOfRows) {
  updateSize(numberOfRows, numberOfColumns);
}

void HostMatrix::updateNumberOfColumns(int numberOfColumns) {
  updateSize(numberOfRows, numberOfColumns);
}

PRECISION* HostMatrix::getMemoryPointer() {
  return hostMatrix;
}

const PRECISION* HostMatrix::getMemoryPointer() const {
  return hostMatrix;
}

} /* namespace Container */
} /* namespace CuEira */
