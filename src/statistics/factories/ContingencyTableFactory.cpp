#include "ContingencyTableFactory.h"

namespace CuEira {

ContingencyTableFactory::ContingencyTableFactory(const Container::HostVector& outcomes) :
    outcomes(outcomes), numberOfIndividualsToInclude(outcomes.getNumberOfRows()) {

}

ContingencyTableFactory::~ContingencyTableFactory() {

}

ContingencyTable* ContingencyTableFactory::constructContingencyTable(const Container::SNPVector& snpVector,
    const Container::EnvironmentVector& environmentVector) const {
  const int size = 8;
  std::vector<int>* tableCellNumbers = new std::vector<int>(size);

  const Container::HostVector& snpData = snpVector.getRecodedData();
  const Container::HostVector& envData = environmentVector.getRecodedData();

  for(int i = 0; i < size; ++i){
    (*tableCellNumbers)[i] = 0;
  }

  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    int pos = -1;
    if(outcomes(i) == 0){ //CONTROL
      pos = snpData(i) + 2 * envData(i);
    }else{ //CASE
      pos = size / 2 + snpData(i) + 2 * envData(i);
    }
    ((*tableCellNumbers)[pos])++;
  }

  return new ContingencyTable(tableCellNumbers);
}

} /* namespace CuEira */
