#include "CudaPhenotypeVector.h"

namespace CuEira {
namespace Container {
namespace CUDA {

CudaPhenotypeVector::CudaPhenotypeVector(const CudaPhenotypeHandler& cudaPhenotypeHandler) :
    PhenotypeVector(cudaPhenotypeHandler), cudaPhenotypeHandler(cudaPhenotypeHandler), orgData(
        cudaPhenotypeHandler.getPhenotypeData()), phenotypeExMissing(nullptr) {

}

CudaPhenotypeVector::~CudaPhenotypeVector() {
  delete phenotypeExMissing;
}

const DeviceVector& CudaPhenotypeVector::getPhenotypeData() const {
#ifdef DEBUG
  if(!initialised){
    throw new InvalidState("CudaPhenotypeVector not initialised.");
  }
#endif

  if(noMissing){
    return orgData;
  }else{
    return *phenotypeExMissing;
  }
}

void CudaPhenotypeVector::applyMissing(const CudaMissingDataHandler& missingDataHandler) {
  delete phenotypeExMissing;
  phenotypeExMissing = missingDataHandler.copyNonMissing(orgData);

  PhenotypeHandler::applyMissing(missingDataHandler);
}

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */
