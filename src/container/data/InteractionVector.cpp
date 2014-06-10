#include "InteractionVector.h"

namespace CuEira {
namespace Container {

InteractionVector::InteractionVector(const EnvironmentVector& environmentVector, const SNPVector& snpVector) :
    environmentVector(environmentVector), snpVector(snpVector), numberOfIndividualsToInclude(
        snpVector.getNumberOfIndividualsToInclude()),
#ifdef CPU
        interactionVector(new LapackppHostVector(new LaVectorDouble(numberOfIndividualsToInclude)))
#else
        interactionVector(new PinnedHostVector(numberOfIndividualsToInclude))
#endif
{
  recode();
}

InteractionVector::~InteractionVector() {
  delete interactionVector;
}

const Container::HostVector& InteractionVector::getRecodedData() const {
  return *interactionVector;
}

int InteractionVector::getNumberOfIndividualsToInclude() const {
  return numberOfIndividualsToInclude;
}

void InteractionVector::recode() {
  const HostVector& envData = environmentVector.getRecodedData();
  const HostVector& snpData = snpVector.getRecodedData();

  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    (*interactionVector)(i) = envData(i) * snpData(i);
  }
}

} /* namespace Container */
} /* namespace CuEira */
