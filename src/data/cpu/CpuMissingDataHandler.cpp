#include "CpuMissingDataHandler.h"

namespace CuEira {
namespace CPU {

CpuMissingDataHandler::CpuMissingDataHandler(const int numberOfIndividualsTotal) :
    MisssingDataHandler(numberOfIndividualsTotal) {
}

CpuMissingDataHandler::~CpuMissingDataHandler() {

}

Container::HostVector* CpuMissingDataHandler::copyNonMissing(const Container::HostVector& fromVector) const {
#ifdef DEBUG
  if(!initialised){
    throw new InvalidState("CpuMissingDataHandler not initialised.");
  }
#endif

  Container::HostVector* toVector = new Container::RegularHostVector(numberOfIndividualsToInclude);

  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    (*toVector)(i) = fromVector((*indexesToCopy)(i)); //FIXME the indexes might not be pointer
  }

  return toVector;
}

} /* namespace CPU */
} /* namespace CuEira */
