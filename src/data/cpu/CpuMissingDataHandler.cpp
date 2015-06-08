#include "CpuMissingDataHandler.h"

namespace CuEira {
namespace CPU {

CpuMissingDataHandler::CpuMissingDataHandler(const int numberOfIndividualsTotal) :
    MissingDataHandler(numberOfIndividualsTotal) {
}

CpuMissingDataHandler::~CpuMissingDataHandler() {

}

void CpuMissingDataHandler::copyNonMissing(const Container::HostVector& fromVector,
    Container::HostVector& toVector) const {
#ifdef DEBUG
  if(!initialised){
    throw new InvalidState("CpuMissingDataHandler not initialised.");
  }
#endif

  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    toVector(i) = fromVector((*indexesToCopy)(i));
  }
}

} /* namespace CPU */
} /* namespace CuEira */
