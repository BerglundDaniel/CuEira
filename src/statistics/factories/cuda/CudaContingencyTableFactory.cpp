#include "CudaContingencyTableFactory.h"

namespace CuEira {
namespace CUDA {

CudaContingencyTableFactory::CudaContingencyTableFactory(){

}

CudaContingencyTableFactory::~CudaContingencyTableFactory(){

}

const ContingencyTable* CudaContingencyTableFactory::constructContingencyTable(
    const Container::SNPVector<Container::DeviceVector>& snpVector,
    const Container::EnvironmentVector<Container::DeviceVector>& environmentVector,
    const Container::PhenotypeVector<Container::DeviceVector>& phenotypeVector) const{
  std::vector<int>* tableCellNumbers = new std::vector<int>(tableSize);

  //TODO

  return new ContingencyTable(tableCellNumbers);
}

} /* namespace CUDA */
} /* namespace CuEira */
