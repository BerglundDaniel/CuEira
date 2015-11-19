#include "CpuPhenotypeHandlerFactory.h"

namespace CuEira {
namespace CPU {

CpuPhenotypeHandlerFactory::CpuPhenotypeHandlerFactory() :
    PhenotypeHandlerFactory(){

}

CpuPhenotypeHandlerFactory::~CpuPhenotypeHandlerFactory(){

}

PhenotypeHandler<Container::HostVector>* CpuPhenotypeHandlerFactory::constructPhenotypeHandler(
    const PersonHandlerLocked& personHandlerLocked) const{
  return new PhenotypeHandler<Container::HostVector>(createVectorOfPhenotypes(personHandlerLocked));
}

} /* namespace CPU */
} /* namespace CuEira */
