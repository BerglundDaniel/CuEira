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
  if(!initialised){
    throw new InvalidState("PhenotypeVector not initialised.");
  }

  if(noMissing){
    return orgData;
  }else{
    return *phenotypeExMissing;
  }
}

void CpuPhenotypeVector::copyNonMissingData(const std::set<int>& personsToSkip) {
  delete phenotypeExMissing;
  phenotypeExMissing = new RegularHostVector(numberOfIndividualsToIncludeNext);

  auto personSkip = personsToSkip.begin();
  int orgDataIndex = 0;

  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    if(personSkip != personsToSkip.end()){
      if(*personSkip == orgDataIndex){
        ++orgDataIndex;
        ++personSkip;
      }
    }

    (*phenotypeExMissing)(i) = orgData(orgDataIndex);
    ++orgDataIndex;
  }

}

} /* namespace CPU */
} /* namespace Container */
} /* namespace CuEira */
