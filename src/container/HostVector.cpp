#include "HostVector.h"

namespace CuEira {
namespace Container {

HostVector::HostVector(unsigned int numberOfRows, bool subview, PRECISION* hostVector) :
    numberOfRows(numberOfRows), numberOfColumns(1), subview(subview), hostVector(hostVector) {
  if(numberOfRows < 0){
    std::ostringstream os;
    os << "Number of rows for HostVector must be > 0" << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
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
