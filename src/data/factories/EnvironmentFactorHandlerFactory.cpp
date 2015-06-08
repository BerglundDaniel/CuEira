#include "EnvironmentFactorHandlerFactory.h"

namespace CuEira {

template<typename Matrix, typename Vector>
EnvironmentFactorHandlerFactory<Matrix, Vector>::EnvironmentFactorHandlerFactory(const Configuration& configuration,
    const std::vector<std::string>& columnNames, const Matrix& matrix) :
    environmentFactor(nullptr), envData(nullptr){

  const std::string environmentColumnName = configuration.getEnvironmentColumnName();
  const int numberOfIndividuals = matrix.getNumberOfRows();
  const int numberOfColumns = matrix.getNumberOfColumns();
  EnvironmentFactor* environmentFactor = new EnvironmentFactor(Id(environmentColumnName));

  for(int i = 0; i < numberOfColumns; ++i){
    if(columnNames[i] == environmentColumnName){
      envData = matrix(i);
      break;
    }
  }

  bool binary = true;
  int max = (*envData)(0);
  int min = (*envData)(0);

  for(int i = 0; i < numberOfIndividuals; ++i){
    if((*envData)(i) != 0 && (*envData)(i) != 1){
      binary = false;
    }

    if((*envData)(i) > max){
      max = (*envData)(i);
    }

    if((*envData)(i) < min){
      min = (*envData)(i);
    }
  }

  environmentFactor->setMax(max);
  environmentFactor->setMin(min);

  if(binary){
    environmentFactor->setVariableType(BINARY);
  }else{
    environmentFactor->setVariableType(OTHER);
  }

  this->environmentFactor.reset(environmentFactor);
}

template<typename Matrix, typename Vector>
EnvironmentFactorHandlerFactory<Matrix, Vector>::~EnvironmentFactorHandlerFactory(){
  delete envData;
}

}
/* namespace CuEira */
