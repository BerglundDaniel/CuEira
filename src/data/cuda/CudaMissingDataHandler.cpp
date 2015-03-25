#include "CudaMissingDataHandler.h"

namespace CuEira {
namespace CUDA {

CudaMissingDataHandler::CudaMissingDataHandler(const int numberOfIndividualsTotal, const HostToDevice& hostToDevice,
    const KernelWrapper& kernelWrapper) :
    MissingDataHandler(numberOfIndividualsTotal), hostToDevice(hostToDevice), kernelWrapper(kernelWrapper), indexesToCopyDevice(
        nullptr) {

}

CudaMissingDataHandler::~CudaMissingDataHandler() {
  delete indexesToCopyDevice;
}

void CudaMissingDataHandler::setMissing(const std::set<int>& snpPersonsToSkip, const std::set<int>& envPersonsToSkip) {
  MissingDataHandler::setMissing(snpPersonsToSkip, envPersonsToSkip);

  delete indexesToCopyDevice;
  indexesToCopyDevice = hostToDevice.transferVector(indexesToCopy);
}

Container::DeviceVector* CudaMissingDataHandler::copyNonMissing(const Container::DeviceVector& fromVector) const {
#ifdef DEBUG
  if(!initialised){
    throw new InvalidState("MissingDataHandler not initialised.");
  }
#endif

  Container::DeviceVector* toVector = new Container::DeviceVector(numberOfIndividualsToInclude);
  kernelWrapper.vectorCopyIndexes(toVector, fromVector, indexesToCopyDevice);

  return toVector;
}

} /* namespace CUDA */
} /* namespace CuEira */
