#include "PhenotypeHandlerFactory.h"

namespace CuEira {

PhenotypeHandlerFactory::PhenotypeHandlerFactory() {

}

PhenotypeHandlerFactory::~PhenotypeHandlerFactory() {

}

PhenotypeHandler* PhenotypeHandlerFactory::constructPhenotypeHandler(const PersonHandler& personHandler) const {
  const std::vector<Person*>& persons = personHandler.getPersons();
  const int numberOfIndividualsToInclude = personHandler.getNumberOfIndividualsToInclude();

#ifdef CPU
  Container::HostVector* phenotypeOriginal = new Container::RegularHostVector(numberOfIndividualsToInclude);
#else
  Container::HostVector* phenotypeOriginal = new Container::PinnedHostVector(numberOfIndividualsToInclude);
#endif

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

#ifdef CPU
  return new CPU::CpuPhenotypeHandler(phenotypeOriginal);
#else
  //TODO om GPU transfer to GPU
  return new CUDA::CudaPhenotypeHandler(phenotypeOriginalDevice);
#endif
}

} /* namespace CuEira */
