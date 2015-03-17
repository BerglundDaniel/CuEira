#include "CudaPhenotypeVector.h"

namespace CuEira {
namespace Container {
namespace CUDA {

CudaPhenotypeVector::CudaPhenotypeVector(const CudaPhenotypeHandler& cudaPhenotypeHandler,
    const HostToDevice& hostToDevice, const KernelWrapper& kernelWrapper) :
    PhenotypeVector(cudaPhenotypeHandler), cudaPhenotypeHandler(cudaPhenotypeHandler), hostToDevice(hostToDevice), kernelWrapper(
        kernelWrapper), orgData(cudaPhenotypeHandler.getPhenotypeData()), phenotypeExMissing(nullptr) {

}

CudaPhenotypeVector::~CudaPhenotypeVector() {
  delete phenotypeExMissing;
}

const DeviceVector& CudaPhenotypeVector::getPhenotypeData() const {
  if(!initialised){
    throw new InvalidState("PhenotypeVector not initialised.");
  }

  if(noMissing){
    return orgData;
  }else{
    return *phenotypeExMissing;
  }
}

void CudaPhenotypeVector::copyNonMissingData(const std::set<int>& personsToSkip) {
  delete phenotypeExMissing;
  phenotypeExMissing = new DeviceVector(numberOfIndividualsToIncludeNext);

  auto personSkip = personsToSkip.begin();
  int orgDataIndex = 0;
  PinnedHostVector indexesToCopy(numberOfIndividualsToInclude);

  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    if(personSkip != personsToSkip.end()){
      if(*personSkip == orgDataIndex){
        ++orgDataIndex;
        ++personSkip;
      }
    }
    indexesToCopy(i) = orgDataIndex;
    ++orgDataIndex;
  }

  DeviceVector* indexesToCopyDevice = hostToDevice.transferVector(indexesToCopy);
  kernelWrapper.vectorCopyIndexes(*phenotypeExMissing, orgData, *indexesToCopyDevice);

  delete indexesToCopyDevice;
}

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */
