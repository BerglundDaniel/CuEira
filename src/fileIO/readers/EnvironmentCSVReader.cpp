#include "EnvironmentCSVReader.h"

namespace CuEira {
namespace FileIO {

EnvironmentCSVReader::EnvironmentCSVReader(PersonHandler& personHandler, std::string filePath, std::string idColumnName,
    std::string delim) :
    CSVReader(personHandler, filePath, idColumnName, delim) {

}

EnvironmentCSVReader::~EnvironmentCSVReader() {

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

  for(int i = 0; i < numberOfDataColumns; ++i){
    Id id((*dataColumnNames)[i]);
    (*environmentFactors)[i] = new EnvironmentFactor(id);
  }

  return new EnvironmentFactorHandler(dataMatrix, environmentFactors);
}

} /* namespace FileIO */
} /* namespace CuEira */
