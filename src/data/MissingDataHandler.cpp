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

void MissingDataHandler::setMissing(const std::set<int>& snpPersonsToSkip, const std::set<int>& envPersonsToSkip) {
  initialised = true;
  std::set<int> personsToSkip;

  std::set_union(snpPersonsToSkip.begin(), snpPersonsToSkip.end(), envPersonsToSkip.begin(), envPersonsToSkip.end(),
      personsToSkip.begin());

  numberOfIndividualsToInclude = numberOfIndividualsTotal - personsToSkip.size();

#ifdef CPU
  indexesToCopy = new Container::RegularHostVector(numberOfIndividualsToInclude);
#else
  indexesToCopy = new Container::PinnedHostVector(numberOfIndividualsToInclude);
#endif

  auto personSkip = personsToSkip.begin();
  int orgDataIndex = 0;
  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    if(personSkip != personsToSkip.end()){
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
