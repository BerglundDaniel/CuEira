#include "ModelHandler.h"

namespace CuEira {
namespace Model {

template<typename Matrix, typename Vector>
ModelHandler<Matrix, Vector>::ModelHandler(const CombinedResultsFactory& combinedResultsFactory,
    DataHandler<Matrix, Vector>* dataHandler) :
    combinedResultsFactory(combinedResultsFactory), dataHandler(dataHandler), snpData(nullptr), environmentData(
        nullptr), interactionData(nullptr), currentSNP(nullptr), currentEnvironmentFactor(nullptr), oldSNP(nullptr), oldEnvironmentFactor(
        nullptr), state(NOT_INITIALISED){

}

template<typename Matrix, typename Vector>
ModelHandler<Matrix, Vector>::~ModelHandler(){
  delete dataHandler;
}

template<typename Matrix, typename Vector>
DataHandlerState ModelHandler<Matrix, Vector>::next(){
  DataHandlerState dataHandlerState = dataHandler->next();
  if(dataHandlerState == DONE){
    return dataHandlerState;
  }

#ifdef DEBUG
  if(state == NOT_INITIALISED){
    state = INITIALISED_READY;
  } else if(state == INITIALISED_READY){
    state = INITIALISED_FULL;
  }
#endif

  oldSNP = currentSNP;
  oldEnvironmentFactor = currentEnvironmentFactor;

  currentSNP = &dataHandler->getCurrentSNP();
  currentEnvironmentFactor = &dataHandler->getCurrentEnvironmentFactor();

  if(dataHandlerState == SKIP){
    snpData = nullptr;
    environmentData = nullptr;
    interactionData = nullptr;

    return dataHandlerState;
  }else{
    snpData = &dataHandler->getSNPVector().getRecodedData();
    environmentData = &dataHandler->getEnvironmentVector().getRecodedData();
    interactionData = &dataHandler->getInteractionVector().getRecodedData();

    return dataHandlerState;
  }
}

template<typename Matrix, typename Vector>
const ModelInformation& ModelHandler<Matrix, Vector>::getCurrentModelInformation() const{
  return dataHandler->getCurrentModelInformation();
}

template<typename Matrix, typename Vector>
const SNP& ModelHandler<Matrix, Vector>::getCurrentSNP() const{
  return dataHandler->getCurrentSNP();
}

template<typename Matrix, typename Vector>
const EnvironmentFactor& ModelHandler<Matrix, Vector>::getCurrentEnvironmentFactor() const{
  return dataHandler->getCurrentEnvironmentFactor();
}

template<typename Matrix, typename Vector>
const Container::SNPVector<Vector>& ModelHandler<Matrix, Vector>::getSNPVector() const{
  return dataHandler->getSNPVector();
}

template<typename Matrix, typename Vector>
const Container::InteractionVector<Vector>& ModelHandler<Matrix, Vector>::getInteractionVector() const{
  return dataHandler->getInteractionVector();
}

template<typename Matrix, typename Vector>
const Container::EnvironmentVector<Vector>& ModelHandler<Matrix, Vector>::getEnvironmentVector() const{
  return dataHandler->getEnvironmentVector();
}

} /* namespace Model */
} /* namespace CuEira */
