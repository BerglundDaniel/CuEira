#include "HostVector.h"

namespace CuEira {
namespace Container {

HostVector::HostVector(int numberOfRows, bool subview, PRECISION* hostVector) :
    numberOfRows(numberOfRows), numberOfColumns(1), subview(subview), hostVector(hostVector) {
  if(numberOfRows <= 0){
    throw DimensionMismatch("Number of rows for HostVector must be > 0");
  }
}

HostVector::~HostVector() {

}

int HostVector::getNumberOfRows() const {
  return numberOfRows;
}

int HostVector::getNumberOfColumns() const {
  return numberOfColumns;
}

PRECISION* HostVector::getMemoryPointer() {
  return hostVector;
}

const PRECISION* HostVector::getMemoryPointer() const {
  return hostVector;
}

} /* namespace Container */
} /* namespace CuEira */
