#include "HostVector.h"

namespace CuEira {
namespace Container {

HostVector::HostVector(int numberOfRealRows, int numberOfRows, bool subview, PRECISION* hostVector) :
    numberOfRealRows(numberOfRealRows), numberOfRows(numberOfRows), subview(subview), hostVector(hostVector) {
  if(numberOfRows <= 0 || numberOfRealRows <= 0){
    throw DimensionMismatch("Number of rows for HostVector must be > 0");
  }
}

HostVector::~HostVector() {

}

int HostVector::getNumberOfRows() const {
  return numberOfRows;
}

int HostVector::getNumberOfColumns() const {
  return 1;
}

int HostVector::getRealNumberOfRows() const {
  return numberOfRealRows;
}

int HostVector::getRealNumberOfColumns() const {
  return 1;
}

void HostVector::updateSize(int numberOfRows) {
#ifdef DEBUG
  if(numberOfRows > numberOfRealRows){
    throw DimensionMismatch("Number of rows for HostVector can't be larger than the real number of rows.");
  }
#endif

  this->numberOfRows = numberOfRows;
}

PRECISION* HostVector::getMemoryPointer() {
  return hostVector;
}

const PRECISION* HostVector::getMemoryPointer() const {
  return hostVector;
}

} /* namespace Container */
} /* namespace CuEira */
