#include "CSVReader.h"

namespace CuEira {
namespace FileIO {

CSVReader::CSVReader(std::string filePath, std::string idColumnName, std::string delim,
    const PersonHandler& personHandler) :
    filePath(filePath), idColumnName(idColumnName), delim(delim), personHandler(personHandler), numberOfRows(0), numberOfColumns(
        0), numberOfIndividualsToInclude(personHandler.getNumberOfIndividualsToInclude()) {

  std::string line;
  std::ifstream csvFile;
  bool header = true;

  try{
    csvFile.open(filePath, std::ifstream::in);
    while(std::getline(csvFile, line)){
      std::vector<std::string> lineSplit;
      int lineSplitSize = lineSplit.size();
      boost::split(lineSplit, line, boost::is_any_of(delim));

      if(header){
        header = false;
        numberOfColumns = lineSplitSize - 1;
        dataColumnNames(numberOfColumns);

        //Read column names
        for(int i = 0; i < lineSplitSize; ++i){
          if(strcmp(lineSplit[i], idColumnName) == 0){
            idColumnNumber = i;
          }else{
            dataColumnNames[i] = lineSplit[i];
          }
        } /* for lineSplit */

        //Initiate matrix
        dataMatrix(numberOfIndividualsToInclude, numberOfColumns);

      }else{ /* if header */
        storeData(lineSplit);

        numberOfRows++;
      } /* if header */
    } /* while getline */
    csvFile.close();
  } catch(const std::ios_base::failure& exception){
    std::ostringstream os;
    os << "Problem reading csv file " << bimFileStr << std::endl;
#ifdef DEBUG
    os << exception.what();
#endif
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
}

CSVReader::~CSVReader() {

}

int CSVReader::getNumberOfColumns() {
  return numberOfColumns;
}

std::vector<std::string> CSVReader::getDataColumnHeaders() {
  return dataColumnNames;
}
Container::HostMatrix& CSVReader::getData() {
  return dataMatrix;
}

Container::HostVector& CSVReader::getData(std::string column) {
  for(int i = 0; i < numberOfColumns; ++i){
    if(strcmp(dataColumnNames[i], column) == 0){
      return dataMatrix(i);
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
  const int personRowNumber = personHandler.getRowIncludeFromPerson(Person);

  int index = 0;
  for(int i = 0; i < numberOfColumns + 1; ++i){
    if(i != idColumnNumber){
      dataMatrix(personRowNumber, index) = lineSplit(i);
      ++index;
    }
  }
}

} /* namespace FileIO */
} /* namespace CuEira */
