#include "CpuPhenotypeVector.h"

namespace CuEira {
namespace Container {
namespace CPU {

CpuPhenotypeVector::CpuPhenotypeVector(const CpuPhenotypeHandler& cpuPhenotypeHandler) :
    PhenotypeVector(cpuPhenotypeHandler), cpuPhenotypeHandler(cpuPhenotypeHandler), orgData(
        cpuPhenotypeHandler.getPhenotypeData()), phenotypeExMissing(nullptr) {

}

CpuPhenotypeVector::~CpuPhenotypeVector() {
  delete phenotypeExMissing;
}

const RegularHostVector& CpuPhenotypeVector::getPhenotypeData() const {
#ifdef DEBUG
  if(!initialised){
    throw new InvalidState("CpuPhenotypeVector not initialised.");
  }
#endif

  if(noMissing){
    return orgData;
  }else{
    return *phenotypeExMissing;
  }
}

void CpuPhenotypeVector::applyMissing(const CpuMissingDataHandler& missingDataHandler) {
  delete phenotypeExMissing;
  phenotypeExMissing = missingDataHandler.copyNonMissing(orgData);

  PhenotypeHandler::applyMissing(missingDataHandler);
}

} /* namespace CPU */
} /* namespace Container */
} /* namespace CuEira */
