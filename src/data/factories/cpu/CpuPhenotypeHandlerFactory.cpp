#include "CpuPhenotypeHandlerFactory.h"

namespace CuEira {
namespace CPU {

CpuPhenotypeHandlerFactory::CpuPhenotypeHandlerFactory() :
    PhenotypeHandlerFactory(){

}

CpuPhenotypeHandlerFactory::~CpuPhenotypeHandlerFactory(){

}

PhenotypeHandler<Container::HostVector>* CpuPhenotypeHandlerFactory::constructPhenotypeHandler(
    const PersonHandler& personHandler) const{
  return new PhenotypeHandler<Container::HostVector>(createVectorOfPhenotypes(personHandler));
}

} /* namespace CPU */
} /* namespace CuEira */
