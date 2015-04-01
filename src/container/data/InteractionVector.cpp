#include "InteractionVector.h"

namespace CuEira {
namespace Container {

template<typename Vector>
InteractionVector<Vector>::InteractionVector() :
    interactionExMissing(nullptr), numberOfIndividualsToInclude(0), initialised(false) {

}

template<typename Vector>
InteractionVector<Vector>::~InteractionVector() {
  delete interactionExMissing;
}

template<typename Vector>
int InteractionVector<Vector>::getNumberOfIndividualsToInclude() const {
  return numberOfIndividualsToInclude;
}

template<typename Vector>
const Vector& InteractionVector<Vector>::getInteractionData() const {
#ifdef DEBUG
  if(!initialised){
    throw InvalidState("Before using the getRecodedData use recode() at least once.");
  }
#endif

  return *interactionExMissing;
}

template<typename Vector>
Vector& InteractionVector<Vector>::getInteractionData() {
#ifdef DEBUG
  if(!initialised){
    throw InvalidState("Before using the getRecodedData use recode() at least once.");
  }
#endif

  return *interactionExMissing;
}

template<typename Vector>
void InteractionVector<Vector>::updateSize(int size) {
#ifdef DEBUG
  initialised = true;
#endif
  numberOfIndividualsToInclude = size;
  delete interactionExMissing;
  interactionExMissing = new Vector(size);
}

} /* namespace Container */
} /* namespace CuEira */
