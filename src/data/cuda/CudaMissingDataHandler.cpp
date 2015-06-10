#include "CudaMissingDataHandler.h"

namespace CuEira {
namespace CUDA {

CudaMissingDataHandler::CudaMissingDataHandler(const int numberOfIndividualsTotal, const Stream& stream) :
    MissingDataHandler<Container::DeviceVector>(numberOfIndividualsTotal), stream(stream), indexesToCopyDevice(nullptr){

}

CudaMissingDataHandler::~CudaMissingDataHandler(){
  delete indexesToCopyDevice;
}

void CudaMissingDataHandler::setMissing(const std::set<int>& snpPersonsToSkip){
  MissingDataHandler::setMissing(snpPersonsToSkip);

  delete indexesToCopyDevice;
  indexesToCopyDevice = transferVector(stream, *indexesToCopy);
}

void CudaMissingDataHandler::copyNonMissing(const Container::DeviceVector& fromVector,
    Container::DeviceVector& toVector) const{
#ifdef DEBUG
  if(!initialised){
    throw new InvalidState("MissingDataHandler not initialised.");
  }
#endif

  Kernel::vectorCopyIndexes(stream, *indexesToCopyDevice, fromVector, toVector);
}

} /* namespace CUDA */
} /* namespace CuEira */
