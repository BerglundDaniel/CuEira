#include "CSVReader.h"

namespace CuEira {
namespace FileIO {

CSVReader::CSVReader(PersonHandler& personHandler, std::string filePath, std::string idColumnName, std::string delim) :
    filePath(filePath), idColumnName(idColumnName), delim(delim), numberOfIndividualsTotal(0), idColumnNumber(-1), dataColumnNames(
        nullptr) {
  readBasicFileInformation();
}

CSVReader::~CSVReader() {
  delete dataColumnNames;
}

void CSVReader::readBasicFileInformation() {
  std::string line;
  std::ifstream csvFile;
  bool header = true;
  int individualNumber = 0; //The individuals might not be in order so don't use this for anything except counting the total number

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
      numberOfDataColumns = lineSplitSize - 1;
      dataColumnNames = new std::vector<std::string>(numberOfDataColumns);
      idColumnNumber = -1;
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
      } /* while for i */

      if(idColumnNumber < 0){
        std::ostringstream os;
        os << "Can't find the column named " << idColumnName << " in csv file " << filePath << std::endl;
        const std::string& tmp = os.str();
        throw FileReaderException(tmp.c_str());
      }
    }else{ /* if header */

      if(idColumnNumber != lineSplitSize){
        std::ostringstream os;
        os << "Number of columns on row " << individualNumber + 1 << " does not match the header in csv file "
            << filePath << std::endl;
        const std::string& tmp = os.str();
        throw FileReaderException(tmp.c_str());
      }

      if(rowHasMissingData(lineSplit)){
        Person& person = personHandler.getPersonFromId(Id(lineSplit[idColumnNumber]));
        person.setInclude(false);
      }

      individualNumber++;
    } /* if header else */

  }

  csvFile.close();

  numberOfIndividualsTotal = individualNumber;

  if(numberOfIndividualsTotal != personHandler.getNumberOfIndividualsTotal()){
    std::ostringstream os;
    os << "Different number of individuals in fam(" << personHandler.getNumberOfIndividualsTotal() << " ) and csv "
        << filePath << " (" << covNumberOfIndividuals << ") files." << std::endl;
    const std::string& tmp = os.str();

    throw FileReaderException(tmp.c_str());
  }
}

bool CSVReader::rowHasMissingData(const std::vector<std::string>& lineSplit) const {
  for(auto& lineSplitPart : lineSplit){
    if(stringIsEmpty(lineSplitPart)){
      return true;
    }
  }
  return false;
}

bool CSVReader::stringIsEmpty(const std::string& string) const {
  //FIXME could also be other things?
  if(string.empty()){
    return true;
  }

  if(string == " "){
    return true;
  }

  return false;
}

int CSVReader::getNumberOfIndividualsTotal() const {
  return numberOfIndividualsTotal;
}

const std::vector<std::string>& getDataColumnNames() const {
  return *dataColumnNames;
}

Container::HostMatrix* CSVReader::readData() const {
  const int numberOfIndividualsToInclude = personHandler.getNumberOfIndividualsToInclude();
  std::string line;
  std::ifstream csvFile;
  bool header = true;

  std::vector<std::string>* dataColumnNames = new std::vector<std::string>(numberOfDataColumns);
#ifdef CPU
  Container::HostMatrix* dataMatrix = new Container::RegularHostMatrix(numberOfIndividualsToInclude,
      numberOfDataColumns);
#else
  Container::HostMatrix* dataMatrix = new Container::PinnedHostMatrix(numberOfIndividualsToInclude,
      numberOfDataColumns);
#endif

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

    if(header){
      header = false;
    }else{ /* if header */
      Id id(lineSplit[idColumnNumber]);
      const Person& person = personHandler.getPersonFromId(id);

      if(person.getInclude()){
        storeData(lineSplit, idColumnNumber, dataMatrix, personHandler.getRowIncludeFromPerson(person));
      }
    } /* if header */
  } /* while getline */

  csvFile.close();

  return dataMatrix;
}

void CSVReader::storeData(std::vector<std::string> line, int idColumnNumber, Container::HostMatrix* dataMatrix,
    const int dataRowNumber) const {
  const int numberOfColumns = dataMatrix->getNumberOfColumns();

  int index = 0;
  for(int i = 0; i < numberOfColumns + 1; ++i){
    if(i != idColumnNumber){
      if(stringIsEmpty(line[i])){
        (*dataMatrix)(dataRowNumber, index) = -1;
      }else{
        char * temp; //Used for error checking of string conversion
#ifdef CPU
        double dataNumber = strtod(line[i].c_str(), &temp);
#else
        float dataNumber = strtof(line[i].c_str(), &temp);
#endif

        if(*temp != '\0'){ //Check if there was an error with conversion
          std::ostringstream os;
#ifdef CPU
          os << "Problem with string to double conversion of data in csv file " << filePath << std::endl;
#else
          os << "Problem with string to float conversion of data in csv file " << filePath << std::endl;
#endif
          const std::string& tmp = os.str();
          delete dataMatrix;
          throw FileReaderException(tmp.c_str());
        }

        (*dataMatrix)(dataRowNumber, index) = dataNumber;
      }/* else if stringIsEmpty */

      ++index;

    } /* if i!=idColumnNumber */
  }/* for numberOfColums */

}

} /* namespace FileIO */
} /* namespace CuEira */
