#include "MissingDataHandler.h"

namespace CuEira {

template<typename Vector>
MissingDataHandler<Vector>::MissingDataHandler(const int numberOfIndividualsTotal) :
    numberOfIndividualsTotal(numberOfIndividualsTotal), numberOfIndividualsToInclude(0), initialised(false), indexesToCopy(
        nullptr) {
}

template<typename Vector>
MissingDataHandler<Vector>::~MissingDataHandler() {
  delete indexesToCopy;
}

template<typename Vector>
int MissingDataHandler<Vector>::getNumberOfIndividualsToInclude() const {
#ifdef DEBUG
  if(!initialised){
    throw new InvalidState("MissingDataHandler not initialised.");
  }
#endif
  return numberOfIndividualsToInclude;
}

template<typename Vector>
int MissingDataHandler<Vector>::getNumberOfIndividualsTotal() const {
  return numberOfIndividualsTotal;
}

template<typename Vector>
void MissingDataHandler<Vector>::setMissing(const std::set<int>& snpPersonsToSkip) {
  initialised = true;
  numberOfIndividualsToInclude = numberOfIndividualsTotal - snpPersonsToSkip.size();

#ifdef CPU
  indexesToCopy = new Container::RegularHostVector(numberOfIndividualsToInclude);
#else
  indexesToCopy = new Container::PinnedHostVector(numberOfIndividualsToInclude);
#endif

  auto personSkip = snpPersonsToSkip.begin();
  int orgDataIndex = 0;
  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    if(personSkip != snpPersonsToSkip.end()){
      if(*personSkip == orgDataIndex){
        ++orgDataIndex;
        ++personSkip;
      }
    }
    indexesToCopy(i) = orgDataIndex;
    ++orgDataIndex;
  }
}

} /* namespace CuEira */
