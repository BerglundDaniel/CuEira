#include "RegularHostVector.h"

namespace CuEira {
namespace Container {

RegularHostVector::RegularHostVector(int numberOfRows) :
    HostVector(ceil(((double) numberOfRows) / CPU_UNROLL) * CPU_UNROLL, numberOfRows, false, nullptr) {
  hostVector = (PRECISION*) malloc(sizeof(PRECISION) * numberOfRealRows);
}

RegularHostVector::RegularHostVector(int numberOfRealRows, int numberOfRows, PRECISION* hostVector, bool subview) :
    HostVector(numberOfRealRows, numberOfRows, subview, hostVector) {

}

RegularHostVector::~RegularHostVector() {
  if(!subview){
    free(hostVector);
  }
}

PRECISION& RegularHostVector::operator()(int index) {
#ifdef DEBUG
  if(index >= numberOfRealRows || index < 0){
    std::ostringstream os;
    os << "Index " << index << " is larger than the number of rows " << numberOfRows << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }
#endif

  return *(hostVector + index);
}

const PRECISION& RegularHostVector::operator()(int index) const {
#ifdef DEBUG
  if(index >= numberOfRealRows || index < 0){
    std::ostringstream os;
    os << "Index " << index << " is larger than the number of rows " << numberOfRows << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }
#endif

  return *(hostVector + index);
}

} /* namespace Container */
} /* namespace CuEira */
