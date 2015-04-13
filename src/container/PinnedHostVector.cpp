#include "PinnedHostVector.h"

namespace CuEira {
namespace Container {

PinnedHostVector::PinnedHostVector(int numberOfRows) :
    HostVector(ceil(((double) numberOfRows) / CPU_UNROLL) * CPU_UNROLL, numberOfRows, false, nullptr) {
  CuEira::CUDA::allocateHostPinnedMemory((void**) &hostVector, numberOfRealRows);
}

PinnedHostVector::PinnedHostVector(int numberOfRealRows, int numberOfRows, PRECISION* hostVector, bool subview) :
    HostVector(numberOfRealRows, numberOfRows, subview, hostVector) {

}

PinnedHostVector::~PinnedHostVector() {
  if(!subview){
    CuEira::CUDA::freePinnedMemory((void*) hostVector);
  }
}

PRECISION& PinnedHostVector::operator()(int index) {
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

const PRECISION& PinnedHostVector::operator()(int index) const {
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
