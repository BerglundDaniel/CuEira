#include "CpuContingencyTableFactory.h"

namespace CuEira {
namespace CPU {

CpuContingencyTableFactory::CpuContingencyTableFactory(){

}

CpuContingencyTableFactory::~CpuContingencyTableFactory(){

}

const ContingencyTable* CpuContingencyTableFactory::constructContingencyTable(
    const Container::SNPVector<Container::RegularHostVector>& snpVector,
    const Container::EnvironmentVector<Container::RegularHostVector>& environmentVector,
    const Container::PhenotypeVector<Container::RegularHostVector>& phenotypeVector) const{
  std::vector<int>* tableCellNumbers = new std::vector<int>(tableSize);

  const int numberOfIndividualsToInclude = snpVector.getNumberOfIndividualsToInclude();
  const Container::HostVector& snpData = snpVector.getSNPData();
  const Container::HostVector& envData = environmentVector.getEnvironmentData();

  for(int i = 0; i < tableSize; ++i){
    (*tableCellNumbers)[i] = 0;
  }

  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    ((*tableCellNumbers)[outcomes(i) * (tableSize / 2) + snpData(i) + 2 * envData(i)])++;
  }

  return new ContingencyTable(tableCellNumbers);
}

} /* namespace CPU */
} /* namespace CuEira */
