#include "CudaPhenotypeHandlerFactory.h"

namespace CuEira {
namespace CUDA {

CudaPhenotypeHandlerFactory::CudaPhenotypeHandlerFactory() :
    PhenotypeHandlerFactory(){

}

CudaPhenotypeHandlerFactory::~CudaPhenotypeHandlerFactory(){

}

PhenotypeHandler<Container::DeviceVector>* CudaPhenotypeHandlerFactory::constructPhenotypeHandler(const Stream& stream,
    const PersonHandler& personHandler) const{

  Container::PinnedHostVector* phenotypeOriginal = createVectorOfPhenotypes(personHandler);
  Container::DeviceVector* phenotypeOriginalDevice = transferVector(stream, *phenotypeOriginal);
  delete phenotypeOriginal;

  return new PhenotypeHandler<Container::DeviceVector>(phenotypeOriginalDevice);
}

} /* namespace CUDA */
} /* namespace CuEira */
