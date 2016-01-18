#include "PhenotypeHandlerFactory.h"

namespace CuEira {

template<typename Vector>
PhenotypeHandlerFactory<Vector>::PhenotypeHandlerFactory(){

}

template<typename Vector>
PhenotypeHandlerFactory<Vector>::~PhenotypeHandlerFactory(){

}

template<typename Vector>
Vector* PhenotypeHandlerFactory<Vector>::createVectorOfPhenotypes(const PersonHandlerLocked& personHandlerLocked) const{
  //const std::vector<Person*>& persons = personHandlerLocked.getPersons();
  const int numberOfIndividualsToInclude = personHandlerLocked.getNumberOfIndividualsToInclude();
  Vector* phenotypeOriginal = new Vector(numberOfIndividualsToInclude);

  int index = 0;
  for(PersonHandlerLocked::const_iterator personIter = personHandlerLocked.begin(); personIter != personHandlerLocked.end();
      ++personIter){
    if((*personIter)->getInclude()){
      if((*personIter)->getPhenotype() == AFFECTED){
        (*phenotypeOriginal)(index) = 1;
      }else if((*personIter)->getPhenotype() == UNAFFECTED){
        (*phenotypeOriginal)(index) = 0;
      }
      index++;
    }
  }

  return phenotypeOriginal;
}

} /* namespace CuEira */
