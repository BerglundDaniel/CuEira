#include "EnvironmentCSVReader.h"

namespace CuEira {
namespace FileIO {

EnvironmentCSVReader::EnvironmentCSVReader(std::string filePath, std::string idColumnName, std::string delim) :
    CSVReader(filePath, idColumnName, delim) {

}

EnvironmentCSVReader::~EnvironmentCSVReader() {

}

EnvironmentFactorHandler* EnvironmentCSVReader::readEnvironmentFactorInformation(
    const PersonHandler& personHandler) const {
  std::pair<Container::HostMatrix*, std::vector<std::string>*>* csvData = readData(personHandler);

  Container::HostMatrix* dataMatrix = csvData->first;
  std::vector<std::string>* headers = csvData->second;
  delete csvData;

  const int numberOfColumns = headers->size();
  std::vector<EnvironmentFactor*>* environmentFactors = new std::vector<EnvironmentFactor*>(numberOfColumns);

  for(int i = 0; i < numberOfColumns; ++i){
    Id id((*headers)[i]);
    (*environmentFactors)[i] = new EnvironmentFactor(id);
  }

  return new EnvironmentFactorHandler(dataMatrix, environmentFactors);
}

} /* namespace FileIO */
} /* namespace CuEira */
