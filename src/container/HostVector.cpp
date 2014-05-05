#include "HostVector.h"

namespace CuEira {
namespace Container {

HostVector::HostVector(int numberOfRows, bool subview, PRECISION* hostVector) :
    numberOfRows(numberOfRows), numberOfColumns(1), subview(subview), hostVector(hostVector) {

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
