#include "ModelHandler.h"

namespace CuEira {
namespace Model {

ModelHandler::ModelHandler(DataHandler& dataHandler) :
    dataHandler(dataHandler), snpData(nullptr), environmentData(nullptr), interactionData(nullptr) {

}

ModelHandler::~ModelHandler() {

}

bool ModelHandler::next() {
  bool hasNext = dataHandler.next();
  if(!hasNext){
    return false;
  }

  snpData = &dataHandler.getSNP();
  environmentData = &dataHandler.getEnvironment();
  interactionData = &dataHandler.getInteraction();

  currentSNP = dataHandler.getCurrentSNP();
  currentEnvironmentFactor = dataHandler.getCurrentEnvironmentFactor();

  return true;
}

} /* namespace Model */
} /* namespace CuEira */
