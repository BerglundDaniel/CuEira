#include "CudaPhenotypeHandler.h"

namespace CuEira {
namespace CUDA {

CudaPhenotypeHandler::CudaPhenotypeHandler(const Container::DeviceVector* phenotypeData) :
    PhenotypeHandler(phenotypeData->getNumberOfRows()), phenotypeData(phenotypeData) {

}

CudaPhenotypeHandler::~CudaPhenotypeHandler() {
  delete phenotypeData;
}

const Container::DeviceVector& CudaPhenotypeHandler::getPhenotypeData() const {
  return *phenotypeData;
}

} /* namespace CUDA */
} /* namespace CuEira */
