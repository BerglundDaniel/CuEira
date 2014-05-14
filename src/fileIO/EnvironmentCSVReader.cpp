#include "EnvironmentCSVReader.h"

namespace CuEira {
namespace FileIO {

EnvironmentCSVReader::EnvironmentCSVReader(std::string filePath, std::string idColumnName, std::string delim,
    const PersonHandler& personHandler) :
    CSVReader(filePath, idColumnName, delim, personHandler), environmentFactors(
        new std::vector<EnvironmentFactor*>(numberOfColumns)) {

  const std::vector<std::string>& headers = getDataColumnHeaders();

  for(int i = 0; i < numberOfColumns; ++i){
    Id id(headers[i]);
    (*environmentFactors)[i] = new EnvironmentFactor(id);
  }

}

EnvironmentCSVReader::~EnvironmentCSVReader() {
  delete environmentFactors;
}

const Container::HostVector& EnvironmentCSVReader::getData(EnvironmentFactor& environmentFactor) const {
  std::string columnName = environmentFactor.getId().getString();
  for(int i = 0; i < numberOfColumns; ++i){
    if(strcmp(dataColumnNames[i].c_str(), columnName.c_str()) == 0){
      const Container::HostVector& vector = *((*dataMatrix)(i));

      //What is the variable type?
      bool binary = true;
      for(int j = 0; j < numberOfIndividualsToInclude; ++j){
        if(vector(j) != 0 && vector(j) != 1){
          binary = false;
        }
      }

      if(binary){
        environmentFactor.setVariableType(BINARY);
#ifdef DEBUG
        std::cerr << "Environmentfactor " << environmentFactor.getId().getString() << " is binary." << std::endl;
#endif
      }else{
        environmentFactor.setVariableType(OTHER);
#ifdef DEBUG
        std::cerr << "Environmentfactor " << environmentFactor.getId().getString() << " is other." << std::endl;
#endif
      }

      return vector;
    }
  }
  std::ostringstream os;
  os << "Can't find column name " << columnName << std::endl;
  const std::string& tmp = os.str();
  throw FileReaderException(tmp.c_str());
}

const std::vector<EnvironmentFactor*>& EnvironmentCSVReader::getEnvironmentFactorInformation() const {
  return *environmentFactors;
}

} /* namespace FileIO */
} /* namespace CuEira */
