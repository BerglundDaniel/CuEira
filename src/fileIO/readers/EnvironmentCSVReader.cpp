#include "EnvironmentCSVReader.h"

namespace CuEira {
namespace FileIO {

EnvironmentCSVReader::EnvironmentCSVReader(const EnvironmentFactorHandlerFactory* environmentFactorHandlerFactory,
    PersonHandler& personHandler, std::string filePath, std::string idColumnName, std::string delim) :
    CSVReader(personHandler, filePath, idColumnName, delim), environmentFactorHandlerFactory(
        environmentFactorHandlerFactory) {

}

EnvironmentCSVReader::~EnvironmentCSVReader() {
  delete environmentFactorHandlerFactory;
}

bool EnvironmentCSVReader::rowHasMissingData(const std::vector<std::string>& lineSplit) const {
  for(auto& lineSplitPart : lineSplit){
    if(!stringIsEmpty(lineSplitPart)){
      return false;
    }
  }

  return true;
}

EnvironmentFactorHandler* EnvironmentCSVReader::readEnvironmentFactorInformation() const {
  Container::HostMatrix* dataMatrix = readData();

  std::vector<EnvironmentFactor*>* environmentFactors = new std::vector<EnvironmentFactor*>(numberOfDataColumns);
  std::vector<std::set<int>>* personsToSkip = new std::vector<std::set<int>>(numberOfDataColumns);

  for(int i = 0; i < numberOfDataColumns; ++i){
    Id id((*dataColumnNames)[i]);
    (*environmentFactors)[i] = new EnvironmentFactor(id);
  }

  return environmentFactorHandlerFactory->constructEnvironmentFactorHandler(dataMatrix, environmentFactors,
      personsToSkip);
}

} /* namespace FileIO */
} /* namespace CuEira */
