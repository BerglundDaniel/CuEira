#include "ModelHandler.h"

namespace CuEira {
namespace Model {

ModelHandler::ModelHandler(Container::DataHandler& dataHandler) :
    dataHandler(dataHandler) {

}

ModelHandler::~ModelHandler() {

}

bool ModelHandler::next() {
  bool hasNext=dataHandler.next();
  if(!hasNext){
    return false;
  }

  //TODO
}

} /* namespace Model */
} /* namespace CuEira */
