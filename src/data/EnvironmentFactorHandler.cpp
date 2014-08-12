#include "EnvironmentFactorHandler.h"

namespace CuEira {

EnvironmentFactorHandler::EnvironmentFactorHandler(Container::HostMatrix* dataMatrix,
    std::vector<EnvironmentFactor*>* environmentFactors) :
    dataMatrix(dataMatrix), environmentFactors(environmentFactors), numberOfColumns(environmentFactors->size()), numberOfIndividualsToInclude(
        dataMatrix->getNumberOfRows()), constEnvironmentFactors(environmentFactors->size())
{

  if(dataMatrix->getNumberOfColumns() != environmentFactors->size()){
    std::ostringstream os;
    os
        << "Number of columns in dataMatrix and number of environmental factors doens't match in EnvironmentFactorHandler."
        << std::endl;
    const std::string& tmp = os.str();
    throw EnvironmentFactorHandlerException(tmp.c_str());
  }

  //Check the variable type for each factor
  for(int i = 0; i < numberOfColumns; ++i){
    bool binary = true;
    EnvironmentFactor* environmentFactor = (*environmentFactors)[i];

    for(int j = 0; j < numberOfIndividualsToInclude; ++j){
      if((*dataMatrix)(j, i) != 0 && (*dataMatrix)(j, i) != 1){
        binary = false;
        break;
      }
    } //for j

    if(binary){
      environmentFactor->setVariableType(BINARY);
    }else{
      environmentFactor->setVariableType(OTHER);
    }

    //Set the factor in the constFactor vector
    constEnvironmentFactors[i] = environmentFactor;

  } // for i

}

EnvironmentFactorHandler::~EnvironmentFactorHandler() {
  delete dataMatrix;
  for(int i = 0; i < numberOfColumns; ++i){
    delete (*environmentFactors)[i];
  }
  delete environmentFactors;
}

int EnvironmentFactorHandler::getNumberOfIndividualsToInclude() const {
  return numberOfIndividualsToInclude;
}

const std::vector<const EnvironmentFactor*>& EnvironmentFactorHandler::getHeaders() const {
  return constEnvironmentFactors;
}

const Container::HostVector& EnvironmentFactorHandler::getData(const EnvironmentFactor& environmentFactor) const {
  for(int i = 0; i < numberOfColumns; ++i){
    if(*(*environmentFactors)[i] == environmentFactor){
      const Container::HostVector& vector = *((*dataMatrix)(i));
      return vector;
    } // if
  } // for i

  std::ostringstream os;
  os << "Can't find EnvironmentFactor " << environmentFactor.getId().getString() << " in EnvironmentFactorHandler."
      << std::endl;
  const std::string& tmp = os.str();
  throw EnvironmentFactorHandlerException(tmp.c_str());
}

} /* namespace CuEira */
