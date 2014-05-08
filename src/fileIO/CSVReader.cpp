#include "CSVReader.h"

namespace CuEira {
namespace FileIO {

CSVReader::CSVReader(std::string filePath, std::string idColumnName, std::string delim,
    const PersonHandler& personHandler) :
    filePath(filePath), idColumnName(idColumnName), delim(delim), personHandler(personHandler), numberOfRows(0), numberOfColumns(
        0), numberOfIndividualsTotal(personHandler.getNumberOfIndividualsTotal()), numberOfIndividualsToInclude(
        personHandler.getNumberOfIndividualsToInclude()) {

  std::string line;
  std::ifstream csvFile;
  bool header = true;

  csvFile.open(filePath, std::ifstream::in);
  if(!csvFile){
    std::ostringstream os;
    os << "Problem opening csv file " << filePath << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }

  while(std::getline(csvFile, line)){
    std::vector<std::string> lineSplit;
    boost::split(lineSplit, line, boost::is_any_of(delim));
    int lineSplitSize = lineSplit.size();

    if(header){
      header = false;
      numberOfColumns = lineSplitSize - 1;
      dataColumnNames = std::vector<std::string>(numberOfColumns);

      //Read column names
      for(int i = 0; i < lineSplitSize; ++i){
        if(strcmp(lineSplit[i].c_str(), idColumnName.c_str()) == 0){
          idColumnNumber = i;
        }else{
          dataColumnNames[i] = lineSplit[i];
        }
      }

      //Initialise matrix
#ifdef CPU
      LaGenMatDouble* lapackppMatrix = new LaGenMatDouble(numberOfIndividualsToInclude, numberOfColumns);
      dataMatrix = new Container::LapackppHostMatrix(lapackppMatrix);
#else
      dataMatrix = new Container::PinnedHostMatrix(numberOfIndividualsToInclude, numberOfColumns);
#endif

    }else{ /* if header */
      storeData(lineSplit);
      numberOfRows++;
    } /* if header */
  } /* while getline */

  csvFile.close();

  if(numberOfRows != numberOfIndividualsTotal){
    std::ostringstream os;
    os << "Not the same number of individuals in the CSV file as the Plink Files " << filePath << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }

}

CSVReader::~CSVReader() {
#ifdef CPU
  delete dataMatrix;
#else

#endif
}

int CSVReader::getNumberOfColumns() {
  return numberOfColumns;
}

int CSVReader::getNumberOfRows() {
  return numberOfRows;
}

std::vector<std::string> CSVReader::getDataColumnHeaders() {
  return dataColumnNames;
}
Container::HostMatrix& CSVReader::getData() {
  return *dataMatrix;
}

Container::HostVector& CSVReader::getData(std::string column) {
  for(int i = 0; i < numberOfColumns; ++i){
    if(strcmp(dataColumnNames[i].c_str(), column.c_str()) == 0){
      return *((*dataMatrix)(i));
    }
  }
  std::ostringstream os;
  os << "Can't find column name " << column << std::endl;
  const std::string& tmp = os.str();
  throw FileReaderException(tmp.c_str());
}

void CSVReader::storeData(std::vector<std::string> lineSplit) {
  Id id(lineSplit[idColumnNumber]);
  const Person& person = personHandler.getPersonFromId(id);
  if(person.getInclude()){
    const unsigned int personRowNumber = personHandler.getRowIncludeFromPerson(person);
    int index = 0;
    for(int i = 0; i < numberOfColumns + 1; ++i){

      if(i != idColumnNumber){

        char * temp; //Used for error checking of string conversion
#ifdef CPU
        double dataNumber = strtod(lineSplit[i].c_str(), &temp);
#else
        float dataNumber = strtof(lineSplit[i].c_str(), &temp);
#endif

        if(*temp != '\0'){ //Check if there was an error with conversion
          std::ostringstream os;
          os << "Problem with string to PRECISION conversion of data in csv file " << filePath << std::endl;
          const std::string& tmp = os.str();
          throw FileReaderException(tmp.c_str());
        }

        (*dataMatrix)(personRowNumber, index) = dataNumber;
        ++index;
      } /* if i!=idColumnNumber */
    }/* for numberOfColums */
  } /* if include */
}

} /* namespace FileIO */
} /* namespace CuEira */
