#include "FamReader.h"

namespace CuEira {
namespace FileIO {

FamReader::FamReader(Configuration& configuration) :
    configuration(configuration), idToPersonMap(std::map<Id, Person>()), numberOfIndividuals(0) {

  std::string famFileStr = configuration.getFamFilePath();
  std::string line;
  std::ifstream famFile;

  try{

//Open file

//Read file

//close file

  } catch(const std::ios_base::failure& exception){
    std::ostringstream os;
    os << "Problem reading fam file " << famFileStr << std::endl;
#ifdef DEBUG
    os << exception.what();
#endif
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
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
