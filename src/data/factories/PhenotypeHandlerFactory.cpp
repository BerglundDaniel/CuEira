#include "PhenotypeHandlerFactory.h"

namespace CuEira {

template<typename Vector>
PhenotypeHandlerFactory<Vector>::PhenotypeHandlerFactory(){

}

template<typename Vector>
PhenotypeHandlerFactory<Vector>::~PhenotypeHandlerFactory(){

}

template<typename Vector>
Vector* PhenotypeHandlerFactory<Vector>::createVectorOfPhenotypes(const PersonHandler& personHandler) const{
  const std::vector<Person*>& persons = personHandler.getPersons();
  const int numberOfIndividualsToInclude = personHandler.getNumberOfIndividualsToInclude();
  Vector* phenotypeOriginal = new Vector(numberOfIndividualsToInclude);

  int index = 0;
  for(auto person : persons){
    if(person->getInclude()){
      if(person->getPhenotype() == AFFECTED){
        (*phenotypeOriginal)(index) = 1;
      }else if(person->getPhenotype() == UNAFFECTED){
        (*phenotypeOriginal)(index) = 0;
      }
      index++;
    }
  }

  return phenotypeOriginal;
}

} /* namespace CuEira */
