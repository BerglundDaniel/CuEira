#include "HostVector.h"

namespace CuEira {
namespace Container {

HostVector::HostVector(int numberOfRows, int numberOfColumns, bool subview, PRECISION* hostVector) :
    numberOfRows(numberOfRows), numberOfColumns(numberOfColumns), subview(subview), hostVector(hostVector) {

}

HostVector::~HostVector() {

}

int HostVector::getNumberOfRows() {
  return numberOfRows;
}

int HostVector::getNumberOfColumns() {
  return numberOfColumns;
}

PRECISION* HostVector::getMemoryPointer() {
  return hostVector;
}

} /* namespace Container */
} /* namespace CuEira */
