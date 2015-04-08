#include "CudaMissingDataHandler.h"

namespace CuEira {
namespace CUDA {

CudaMissingDataHandler::CudaMissingDataHandler(const int numberOfIndividualsTotal, const HostToDevice& hostToDevice,
    const KernelWrapper& kernelWrapper) :
    MissingDataHandler<Container::DeviceVector>(numberOfIndividualsTotal), hostToDevice(hostToDevice), kernelWrapper(kernelWrapper), indexesToCopyDevice(
        nullptr) {

}

CudaMissingDataHandler::~CudaMissingDataHandler() {
  delete indexesToCopyDevice;
}

void CudaMissingDataHandler::setMissing(const std::set<int>& snpPersonsToSkip) {
  MissingDataHandler::setMissing(snpPersonsToSkip);

  delete indexesToCopyDevice;
  indexesToCopyDevice = hostToDevice.transferVector(indexesToCopy);
}

void CudaMissingDataHandler::copyNonMissing(const Container::DeviceVector& fromVector,
    Container::DeviceVector& toVector) const {
#ifdef DEBUG
  if(!initialised){
    throw new InvalidState("MissingDataHandler not initialised.");
  }
#endif

  kernelWrapper.vectorCopyIndexes(indexesToCopyDevice, fromVector, toVector);
}

} /* namespace CUDA */
} /* namespace CuEira */
