#include "RegularHostVector.h"

namespace CuEira {
namespace Container {

RegularHostVector::RegularHostVector(int numberOfRows) :
    HostVector(numberOfRows, false, nullptr) {
  hostVector = (PRECISION*) malloc(sizeof(PRECISION) * numberOfRows);
}

RegularHostVector::RegularHostVector(int numberOfRows, PRECISION* hostVector, bool subview) :
    HostVector(numberOfRows, subview, hostVector) {

}

RegularHostVector::~RegularHostVector() {
  if(!subview){
    free(hostVector);
  }
}

PRECISION& RegularHostVector::operator()(int index) {
  if(index >= numberOfRows || index < 0){
    std::ostringstream os;
    os << "Index " << index << " is larger than the number of rows " << numberOfRows << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }

  return *(hostVector + index);
}

const PRECISION& RegularHostVector::operator()(int index) const {
  if(index >= numberOfRows || index < 0){
    std::ostringstream os;
    os << "Index " << index << " is larger than the number of rows " << numberOfRows << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }

  return *(hostVector + index);
}

} /* namespace Container */
} /* namespace CuEira */
