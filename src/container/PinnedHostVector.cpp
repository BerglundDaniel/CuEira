#include "PinnedHostVector.h"

namespace CuEira {
namespace Container {

PinnedHostVector::PinnedHostVector(int numberOfRows) :
    HostVector(numberOfRows, false, nullptr) {
  CuEira::CUDA::allocateHostPinnedMemory((void**) &hostVector, numberOfRows);
}

PinnedHostVector::PinnedHostVector(int numberOfRows, PRECISION* hostVector, bool subview) :
    HostVector(numberOfRows, subview, hostVector) {

}

PinnedHostVector::~PinnedHostVector() {
  if(!subview){
    CuEira::CUDA::freePinnedMemory((void*) hostVector);
  }
}

PRECISION& PinnedHostVector::operator()(int index) {
  if(index >= numberOfRows || index < 0){
    std::ostringstream os;
    os << "Index " << index << " is larger than the number of rows " << numberOfRows << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }

  return *(hostVector + index);
}

const PRECISION& PinnedHostVector::operator()(int index) const {
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
