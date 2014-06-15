#include "ModelHandler.h"

namespace CuEira {
namespace Model {

ModelHandler::ModelHandler(DataHandler* dataHandler) :
    dataHandler(dataHandler), snpData(nullptr), environmentData(nullptr), interactionData(nullptr), currentSNP(nullptr), currentEnvironmentFactor(
        nullptr), oldSNP(nullptr), oldEnvironmentFactor(nullptr), state(NOT_INITIALISED) {

}

ModelHandler::~ModelHandler() {
  delete dataHandler;
}

bool ModelHandler::next() {
  bool hasNext = dataHandler->next();
  if(!hasNext){
    return false;
  }

  if(state == NOT_INITIALISED){
    state = INITIALISED_READY;
  }else if(state == INITIALISED_READY){
    state = INITIALISED_FULL;
  }

  snpData = &dataHandler->getSNP();
  environmentData = &dataHandler->getEnvironment();
  interactionData = &dataHandler->getInteraction();

  oldSNP = currentSNP;
  oldEnvironmentFactor = currentEnvironmentFactor;

  currentSNP = &dataHandler->getCurrentSNP();
  currentEnvironmentFactor = &dataHandler->getCurrentEnvironmentFactor();

  return true;
}

} /* namespace Model */
} /* namespace CuEira */
