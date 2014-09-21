#include "ModelHandler.h"

namespace CuEira {
namespace Model {

ModelHandler::ModelHandler(const CombinedResultsFactory& combinedResultsFactory, DataHandler* dataHandler) :
    combinedResultsFactory(combinedResultsFactory), dataHandler(dataHandler), snpData(nullptr), environmentData(
        nullptr), interactionData(nullptr), currentSNP(nullptr), currentEnvironmentFactor(nullptr), oldSNP(nullptr), oldEnvironmentFactor(
        nullptr), state(NOT_INITIALISED) {

}

ModelHandler::~ModelHandler() {
  delete dataHandler;
}

ModelInformation* ModelHandler::next() {
  ModelInformation* modelInformation = dataHandler->next();
  ModelState modelState = modelInformation->getModelState();
  if(modelState == DONE){
    return modelInformation;
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

  if(modelState == SKIP){
    snpData = nullptr;
    environmentData = nullptr;
    interactionData = nullptr;

    return modelInformation;
  }else{
    snpData = &dataHandler->getSNPVector().getRecodedData();
    environmentData = &dataHandler->getEnvironmentVector().getRecodedData();
    interactionData = &dataHandler->getInteractionVector().getRecodedData();

    return modelInformation;
  }
}

const SNP& ModelHandler::getCurrentSNP() const {
  return dataHandler->getCurrentSNP();
}

const EnvironmentFactor& ModelHandler::getCurrentEnvironmentFactor() const {
  return dataHandler->getCurrentEnvironmentFactor();
}

const Container::SNPVector& ModelHandler::getSNPVector() const {
  return dataHandler->getSNPVector();
}

const Container::InteractionVector& ModelHandler::getInteractionVector() const {
  return dataHandler->getInteractionVector();
}

const Container::EnvironmentVector& ModelHandler::getEnvironmentVector() const {
  return dataHandler->getEnvironmentVector();
}

} /* namespace Model */
} /* namespace CuEira */
