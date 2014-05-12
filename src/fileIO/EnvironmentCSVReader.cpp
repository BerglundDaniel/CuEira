#include "EnvironmentCSVReader.h"

namespace CuEira {
namespace FileIO {

EnvironmentCSVReader::EnvironmentCSVReader(std::string filePath, std::string idColumnName, std::string delim,
    const PersonHandler& personHandler) :
    CSVReader(filePath, idColumnName, delim, personHandler) {

}

EnvironmentCSVReader::~EnvironmentCSVReader() {

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
      }else{
        environmentFactor.setVariableType(OTHER);
      }

      return vector;
    }
  }
  std::ostringstream os;
  os << "Can't find column name " << columnName << std::endl;
  const std::string& tmp = os.str();
  throw FileReaderException(tmp.c_str());
}

} /* namespace FileIO */
} /* namespace CuEira */
