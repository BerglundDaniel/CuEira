#include "CSVReader.h"

namespace CuEira {
namespace FileIO {

CSVReader::CSVReader(std::string filePath, std::string idColumnName, std::string delim) :
    filePath(filePath), idColumnName(idColumnName), delim(delim) {

}

CSVReader::~CSVReader() {

}

std::pair<Container::HostMatrix*, std::vector<std::string>*>* CSVReader::readData(
    const PersonHandler& personHandler) const {
  const int numberOfIndividualsToInclude = personHandler.getNumberOfIndividualsToInclude();
  const int numberOfIndividualsTotal = personHandler.getNumberOfIndividualsTotal();

  Container::HostMatrix* dataMatrix;
  std::string line;
  std::ifstream csvFile;
  bool header = true;
  int numberOfRows = 0; //Not including header
  std::vector<std::string>* dataColumnNames;
  int idColumnNumber = -1;

  csvFile.open(filePath, std::ifstream::in);
  if(!csvFile){
    std::ostringstream os;
    os << "Problem opening csv file " << filePath << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }

  while(std::getline(csvFile, line)){
    std::vector<std::string> lineSplit;
    boost::trim(line);
    boost::split(lineSplit, line, boost::is_any_of(delim));
    int lineSplitSize = lineSplit.size();

    if(header){
      header = false;
      const int numberOfDataColumns = lineSplitSize - 1;
      dataColumnNames = new std::vector<std::string>(numberOfDataColumns);
      int dataColumnPos = 0;

      //Read column names
      for(int i = 0; i < lineSplitSize; ++i){
        if(boost::iequals(lineSplit[i], idColumnName)){
          if(idColumnNumber >= 0){
            std::ostringstream os;
            os << "There is more than one column(case insensitive) named " << idColumnName << " in csv file "
                << filePath << std::endl;
            const std::string& tmp = os.str();
            throw FileReaderException(tmp.c_str());
          }
          idColumnNumber = i;
        }else{
          (*dataColumnNames)[dataColumnPos] = lineSplit[i];
          dataColumnPos++;
        }
      }

      if(idColumnNumber < 0){
        std::ostringstream os;
        os << "Can't find the column named " << idColumnName << " in csv file " << filePath << std::endl;
        const std::string& tmp = os.str();
        throw FileReaderException(tmp.c_str());
      }

      //Initialise matrix
#ifdef CPU
      LaGenMatDouble* lapackppMatrix = new LaGenMatDouble(numberOfIndividualsToInclude, numberOfDataColumns);
      dataMatrix = new Container::LapackppHostMatrix(lapackppMatrix);
#else
      dataMatrix = new Container::PinnedHostMatrix(numberOfIndividualsToInclude, numberOfDataColumns);
#endif

    }else{ /* if header */
      Id id(lineSplit[idColumnNumber]);
      const Person& person = personHandler.getPersonFromId(id);
      if(person.getInclude()){
        const unsigned int personRowNumber = personHandler.getRowIncludeFromPerson(person);
        storeData(lineSplit, idColumnNumber, dataMatrix, personRowNumber);
      }
      numberOfRows++;
    } /* if header */
  } /* while getline */

  csvFile.close();

  if(numberOfRows != numberOfIndividualsTotal){
    std::ostringstream os;
    os << "Not the same number of individuals in the CSV file as the Plink Files " << filePath << std::endl;
    const std::string& tmp = os.str();
    delete dataMatrix;
    delete dataColumnNames;
    throw FileReaderException(tmp.c_str());
  }

  return new std::pair<Container::HostMatrix*, std::vector<std::string>*>(dataMatrix, dataColumnNames);
}

void CSVReader::storeData(std::vector<std::string> line, int idColumnNumber, Container::HostMatrix* dataMatrix,
    unsigned int dataRowNumber) const {
  const int numberOfColumns = dataMatrix->getNumberOfColumns();

  int index = 0;
  for(int i = 0; i < numberOfColumns + 1; ++i){
    if(i != idColumnNumber){
      char * temp; //Used for error checking of string conversion
#ifdef CPU
      double dataNumber = strtod(line[i].c_str(), &temp);
#else
      float dataNumber = strtof(line[i].c_str(), &temp);
#endif

      if(*temp != '\0'){ //Check if there was an error with conversion
        std::ostringstream os;
        os << "Problem with string to PRECISION conversion of data in csv file " << filePath << std::endl;
        const std::string& tmp = os.str();
        delete dataMatrix;
        throw FileReaderException(tmp.c_str());
      }

      (*dataMatrix)(dataRowNumber, index) = dataNumber;
      ++index;
    } /* if i!=idColumnNumber */
  }/* for numberOfColums */

}

} /* namespace FileIO */
} /* namespace CuEira */
