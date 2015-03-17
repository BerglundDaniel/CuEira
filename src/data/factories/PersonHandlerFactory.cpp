#include "PersonHandlerFactory.h"

namespace CuEira {

PersonHandlerFactory::PersonHandlerFactory() {

}

PersonHandlerFactory::~PersonHandlerFactory() {

}

PersonHandler* PersonHandlerFactory::constructPersonHandler(std::vector<Person*>* persons) const {
  return new PersonHandler(persons);
}

} /* namespace CuEira */
