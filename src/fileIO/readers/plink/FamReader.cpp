#include "FamReader.h"

namespace CuEira {
namespace FileIO {

FamReader::FamReader(const Configuration& configuration, const PersonHandlerFactory* personHandlerFactory) :
    configuration(configuration), personHandlerFactory(personHandlerFactory), famFileStr(
        configuration.getFamFilePath()), phenotypeCoding(configuration.getPhenotypeCoding()), numberOfIndividualsTotal(
        0) {
  readBasicFileInformation();
}

FamReader::~FamReader() {
  delete personHandlerFactory;
}

void FamReader::readBasicFileInformation() {
  std::string line;
  std::ifstream famFile;
  int individualNumber = 0;

  famFile.open(famFileStr, std::ifstream::in);
  if(!famFile){
    std::ostringstream os;
    os << "Problem opening fam file " << famFileStr << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }

  //Read file
  while(std::getline(famFile, line)){
    std::vector<std::string> lineSplit;
    boost::trim(line);
    boost::split(lineSplit, line, boost::is_any_of("\t "));

    individualNumber++;
  } /* while getline */

  famFile.close();

  numberOfIndividualsTotal = individualNumber;
}

int FamReader::getNumberOfIndividualsTotal() const {
  return numberOfIndividualsTotal;
}

PersonHandler* FamReader::readPersonInformation() const {
  /*
   * Columns in the file
   * Family ID
   * Individual ID
   * Paternal ID
   * Maternal ID
   * Sex (1=male; 2=female; other=unknown)
   * Phenotype
   *
   * */

  std::string line;
  std::ifstream famFile;
  std::vector<Person*>* persons = new std::vector<Person*>(numberOfIndividualsTotal);
  int individualNumber = 0;

  famFile.open(famFileStr, std::ifstream::in);
  if(!famFile){
    std::ostringstream os;
    os << "Problem opening fam file " << famFileStr << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }

#ifdef PROFILE
  boost::chrono::system_clock::time_point beforeReadFilePoint = boost::chrono::system_clock::now();
#endif

  //Read file
  while(std::getline(famFile, line)){
    std::vector<std::string> lineSplit;
    boost::trim(line);
    boost::split(lineSplit, line, boost::is_any_of("\t"));

    Id id(lineSplit[1]);
    Sex sex = stringToSex(lineSplit[4]);
    Phenotype phenotype = stringToPhenotype(lineSplit[5]);

    Person* person = new Person(id, sex, phenotype, phenotype != MISSING);
    (*persons)[individualNumber] = person;

    individualNumber++;
  } /* while getline */

  famFile.close();

  return personHandlerFactory->constructPersonHandler(persons);
}

Phenotype FamReader::stringToPhenotype(std::string phenotypeString) const {
  char * temp; //Used for error checking of string to long int conversion (strtol)
  long int phenotypeInt = strtol(phenotypeString.c_str(), &temp, 0);
  if(*temp != '\0'){ //Check if there was an error with strtol
    std::ostringstream os;
    os << "Problem with string to int conversion of phenotype in fam file " << famFileStr << " of string "
        << phenotypeString << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
  if(phenotypeCoding == ONE_TWO_CODING){
    if(phenotypeInt == 2){
      return (AFFECTED);
    }else if(phenotypeInt == 1){
      return (UNAFFECTED);
    }else if(phenotypeInt == 9 || phenotypeInt == 0){
      return (MISSING);
    }else{
      std::ostringstream os;
      os << "Unknown phenotype status in fam file " << famFileStr << std::endl;
      const std::string& tmp = os.str();
      throw FileReaderException(tmp.c_str());
    }
  }else if(phenotypeCoding == ZERO_ONE_CODING){
    if(phenotypeInt == 1){
      return (AFFECTED);
    }else if(phenotypeInt == 0){
      return (UNAFFECTED);
    }else if(phenotypeInt == 9){
      return (MISSING);
    }else{
      std::ostringstream os;
      os << "Unknown phenotype status in fam file " << famFileStr << std::endl;
      const std::string& tmp = os.str();
      throw FileReaderException(tmp.c_str());
    }
  }else{
    std::ostringstream os;
    os << "Unknown phenotype coding" << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
}

Sex FamReader::stringToSex(std::string sexString) const {
  char * temp; //Used for error checking of string to long int conversion (strtol)
  long int sexInt = strtol(sexString.c_str(), &temp, 0);
  if(*temp != '\0'){ //Check if there was an error with strtol
    std::ostringstream os;
    os << "Problem with string to int conversion of sex in fam file " << famFileStr << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }

  if(sexInt == 1){
    return (MALE);
  }else if(sexInt == 2){
    return (FEMALE);
  }else{
    return (UNKNOWN);
  }
}

} /* namespace FileIO */
} /* namespace CuEira */
