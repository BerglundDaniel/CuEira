#include "FamReader.h"

namespace CuEira {
namespace FileIO {

FamReader::FamReader(Configuration& configuration) :
    configuration(configuration), idToPersonMap(std::map<Id, Person>()) {

  numberOfIndividuals = 0;

//Open file

//Read file

//close file

}

FamReader::~FamReader() {

}

Container::HostVector FamReader::getOutcomes() {
  return outcomes;
}

int FamReader::getNumberOfIndividuals() {
  return numberOfIndividuals;
}

std::map<Id, Person>& FamReader::getIdToPersonMap() {
  return idToPersonMap;
}

} /* namespace FileIO */
} /* namespace CuEira */
