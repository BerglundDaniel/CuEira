#include "CudaPhenotypeHandlerFactory.h"

namespace CuEira {
namespace CUDA {

CudaPhenotypeHandlerFactory::CudaPhenotypeHandlerFactory() :
    PhenotypeHandlerFactory(){

}

CudaPhenotypeHandlerFactory::~CudaPhenotypeHandlerFactory(){

}

PhenotypeHandler<Container::DeviceVector>* CudaPhenotypeHandlerFactory::constructPhenotypeHandler(const Stream& stream,
    const PersonHandlerLocked& personHandlerLocked) const{

  Container::PinnedHostVector* phenotypeOriginal = createVectorOfPhenotypes(personHandlerLocked);
  Container::DeviceVector* phenotypeOriginalDevice = transferVector(stream, *phenotypeOriginal);
  delete phenotypeOriginal;

  return new PhenotypeHandler<Container::DeviceVector>(phenotypeOriginalDevice);
}

} /* namespace CUDA */
} /* namespace CuEira */
