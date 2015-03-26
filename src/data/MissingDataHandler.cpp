#include "MissingDataHandler.h"

namespace CuEira {

MissingDataHandler::MissingDataHandler(const int numberOfIndividualsTotal) :
    numberOfIndividualsTotal(numberOfIndividualsTotal), numberOfIndividualsToInclude(0), initialised(false), indexesToCopy(
        nullptr) {
}

MissingDataHandler::~MissingDataHandler() {
  delete indexesToCopy;
}

int MissingDataHandler::getNumberOfIndividualsToInclude() const {
#ifdef DEBUG
  if(!initialised){
    throw new InvalidState("MissingDataHandler not initialised.");
  }
#endif
  return numberOfIndividualsToInclude;
}

int MissingDataHandler::getNumberOfIndividualsTotal() const {
  return numberOfIndividualsTotal;
}

void MissingDataHandler::setMissing(const std::set<int>& snpPersonsToSkip) {
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
