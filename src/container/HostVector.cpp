#include "HostVector.h"

namespace CuEira {
namespace Container {

HostVector::HostVector(unsigned int numberOfRows, bool subview, PRECISION* hostVector) :
    numberOfRows(numberOfRows), numberOfColumns(1), subview(subview), hostVector(hostVector) {

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
