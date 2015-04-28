#include "InteractionVector.h"

namespace CuEira {
namespace Container {

template<typename Vector>
InteractionVector<Vector>::InteractionVector(int numberOfIndividualsTotal) :
    interactionExMissing(new Vector(numberOfIndividualsTotal)), numberOfIndividualsToInclude(0), initialised(false) {

}

template<typename Vector>
InteractionVector<Vector>::~InteractionVector() {
  delete interactionExMissing;
}

template<typename Vector>
int InteractionVector<Vector>::getNumberOfIndividualsToInclude() const {
#ifdef DEBUG
  if(!initialised){
    throw InvalidState("InteractionVector is not initialised.");
  }
#endif
  return numberOfIndividualsToInclude;
}

template<typename Vector>
const Vector& InteractionVector<Vector>::getInteractionData() const {
#ifdef DEBUG
  if(!initialised){
    throw InvalidState("InteractionVector is not initialised.");
  }
#endif

  return *interactionExMissing;
}

template<typename Vector>
Vector& InteractionVector<Vector>::getInteractionData() {
#ifdef DEBUG
  if(!initialised){
    throw InvalidState("InteractionVector is not initialised.");
  }
#endif

  return *interactionExMissing;
}

template<typename Vector>
void InteractionVector<Vector>::updateSize(int size) {
#ifdef DEBUG
  initialised = true;
#endif

  if(size > numberOfIndividualsToInclude){
    throw InvalidArgument(
        "Can't set size of InteractionVector to a larger number than the total number of individuals.");
  }

  numberOfIndividualsToInclude = size;
  interactionExMissing->updateSize(size);
}

} /* namespace Container */
} /* namespace CuEira */
