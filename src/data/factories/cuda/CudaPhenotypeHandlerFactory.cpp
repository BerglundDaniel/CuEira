#include "CudaPhenotypeHandlerFactory.h"

namespace CuEira {
namespace CUDA {

CudaPhenotypeHandlerFactory::CudaPhenotypeHandlerFactory() :
    PhenotypeHandlerFactory(){

}

CudaPhenotypeHandlerFactory::~CudaPhenotypeHandlerFactory(){

}

PhenotypeHandler<Container::DeviceVector>* CudaPhenotypeHandlerFactory::constructPhenotypeHandler(
    const PersonHandler& personHandler, const HostToDevice& hostToDevice) const{

  Container::PinnedHostVector* phenotypeOriginal = createVectorOfPhenotypes(personHandler);
  Container::DeviceVector* phenotypeOriginalDevice = hostToDevice.transferVector(*phenotypeOriginal);
  delete phenotypeOriginal;

  return new PhenotypeHandler<Container::DeviceVector>(phenotypeOriginalDevice);
}

} /* namespace CUDA */
} /* namespace CuEira */
