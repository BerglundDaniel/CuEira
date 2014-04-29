#include "FamReader.h"

namespace CuEira {
namespace FileIO {

FamReader::FamReader(const Configuration& configuration, PersonHandler& personHandler) :
    configuration(configuration), personHandler(personHandler) {

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

  std::string famFileStr = configuration.getFamFilePath();
  std::string line;
  std::ifstream famFile;

  try{
    famFile.open(famFileStr, std::ifstream::in);

    //Read file
    while(std::getline(famFile, line)){
      std::vector<std::string> lineSplit;
      boost::split(lineSplit, line, boost::is_any_of("\t "));

      Id id(lineSplit[1]);
      Sex sex = strToSex(lineSplit[4]);
      Phenotype phenotype = strToPhenoType(lineSplit[5]);
      Person person(id, sex, phenotype);

      //Add the person
      personHandler.addPerson(person, individualNumber);
      individualNumber++;

    } /* while getline */

    famFile.close();

  } catch(const std::ios_base::failure& exception){
    std::ostringstream os;
    os << "Problem reading fam file " << famFileStr << std::endl;
#ifdef DEBUG
    os << exception.what();
#endif
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
}

FamReader::~FamReader() {
  delete personHandler;
}

PersonHandler& FamReader::getPersonHandler() const {
  return personHandler;
}

Phenotype FamReader::stringToPhenoType(std::string phenotypeString) const {
  char * temp; //Used for error checking of string to long int conversion (strtol)
  long int phenotypeInt = strtol(phenotypeString.c_str(), &temp, 0);
  if(*temp != '\0'){ //Check if there was an error with strtol
    std::ostringstream os;
    os << "Problem with string to int conversion of phenotype in fam file " << famFileStr << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
  if(configuration.getPhenotypeCoding() == ONE_TWO_CODING){
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
  }else if(configuration.getPhenotypeCoding() == ZERO_ONE_CODING){
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
