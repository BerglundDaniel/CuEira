#include "PhenotypeHandler.h"

namespace CuEira {

PhenotypeHandler::PhenotypeHandler(int numberOfIndividuals) :
    numberOfIndividuals(numberOfIndividuals) {

}

PhenotypeHandler::~PhenotypeHandler() {

}

int PhenotypeHandler::getNumberOfIndividuals() const {
  return numberOfIndividuals;
}

} /* namespace CuEira */
