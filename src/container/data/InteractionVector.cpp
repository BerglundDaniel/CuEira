#include "InteractionVector.h"

namespace CuEira {
namespace Container {

InteractionVector::InteractionVector(const EnvironmentVector& environmentVector) :
    environmentVector(&environmentVector), numberOfIndividualsToInclude(
        environmentVector.getNumberOfIndividualsToInclude()), state(NOT_INITIALISED),
#ifdef CPU
        interactionVector(new LapackppHostVector(new LaVectorDouble(numberOfIndividualsToInclude)))
#else
        interactionVector(new PinnedHostVector(numberOfIndividualsToInclude))
#endif
{

}

InteractionVector::InteractionVector() :
    interactionVector(nullptr), numberOfIndividualsToInclude(0), state(NOT_INITIALISED), environmentVector(nullptr) {

}

InteractionVector::~InteractionVector() {
  delete interactionVector;
}

const Container::HostVector& InteractionVector::getRecodedData() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using the getRecodedData use recode() at least once.");
  }
#endif

  return *interactionVector;
}

int InteractionVector::getNumberOfIndividualsToInclude() const {
  return numberOfIndividualsToInclude;
}

void InteractionVector::recode(const SNPVector& snpVector) {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    state=INITIALISED;
  }
#endif

  const HostVector& envData = environmentVector->getRecodedData();
  const HostVector& snpData = snpVector.getRecodedData();

  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    (*interactionVector)(i) = envData(i) * snpData(i);
  }
}

} /* namespace Container */
} /* namespace CuEira */
