#include "CpuPhenotypeHandler.h"

namespace CuEira {
namespace CPU {

CpuPhenotypeHandler::CpuPhenotypeHandler(const Container::RegularHostVector* phenotypeData) :
    PhenotypeHandler(phenotypeData->getNumberOfRows()), phenotypeData(phenotypeData) {

}

CpuPhenotypeHandler::~CpuPhenotypeHandler() {
  delete phenotypeData;
}

const Container::RegularHostVector& CpuPhenotypeHandler::getPhenotypeData() const {
  return *phenotypeData;
}

} /* namespace CPU */
} /* namespace CuEira */
