#include "FamReader.h"

namespace CuEira {
namespace FileIO {

FamReader::FamReader(Configuration& configuration) :
    configuration(configuration), numberOfIndividuals(0) {

  std::string famFileStr = configuration.getFamFilePath();
  std::string line;
  std::ifstream famFile;

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

  //Read the whole file once to count the number of individuals
  try{
    famFile.open(famFileStr, std::ifstream::in);
    while(std::getline(bimFile, line)){
      numberOfIndividuals++;
    }
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

  int rowNumber = 0;
  char * temp; //Used for error checking of string to long int conversion (strtol)
  persons = std::vector<Person*>(numberOfIndividuals);

  try{
    famFile.open(famFileStr, std::ifstream::in);

    //Read file
    while(std::getline(famFile, line)){
      std::vector<std::string> lineSplit;
      boost::split(lineSplit, line, boost::is_any_of("\t "));

      //Handle the persons id
      Id id = Id(lineSplit[1]);

      //Handle the persons sex
      long int sexInt = strtol(lineSplit[4].c_str(), &temp, 0);
      Sex sex;
      if(*temp != '\0'){ //Check if there was an error with strtol
        std::ostringstream os;
        os << "Problem with string to int conversion of sex in fam file " << famFileStr << std::endl;
        const std::string& tmp = os.str();
        throw FileReaderException(tmp.c_str());
      }

      if(sexInt == 1){
        sex = MALE;
      }else if(sexInt == 2){
        sex = FEMALE;
      }else{
        sex = UNKNOWN;
      }

      //Handle the persons phenotype
      long int phenotypeInt = strtol(lineSplit[5].c_str(), &temp, 0);
      Phenotype phenotype;
      if(*temp != '\0'){ //Check if there was an error with strtol
        std::ostringstream os;
        os << "Problem with string to int conversion of phenotype in fam file " << famFileStr << std::endl;
        const std::string& tmp = os.str();
        throw FileReaderException(tmp.c_str());
      }
      if(configuration.getPhenotypeCoding() == ONE_TWO_CODING){
        if(phenotypeInt == 2){
          phenotype = AFFECTED;
        }else if(phenotypeInt == 1){
          phenotype = UNAFFECTED;
        }else if(phenotypeInt == 9 || phenotypeInt == 0){
          phenotype = MISSING;
        }else{
          std::ostringstream os;
          os << "Unknown phenotype status in fam file " << famFileStr << std::endl;
          const std::string& tmp = os.str();
          throw FileReaderException(tmp.c_str());
        }
      }else if(configuration.getPhenotypeCoding() == ZERO_ONE_CODING){
        if(phenotypeInt == 1){
          phenotype = AFFECTED;
        }else if(phenotypeInt == 0){
          phenotype = UNAFFECTED;
        }else if(phenotypeInt == 9){
          phenotype = MISSING;
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

      Person* person = new Person(id, sex, phenotype, rowNumber);
      persons[rowNumber] = person;

      rowNumber++;
    }
    /* while getline */

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

}

Container::HostVector FamReader::getOutcomes() {
  return outcomes;
}

int FamReader::getNumberOfIndividuals() {
  return numberOfIndividuals;
}

std::vector<Person*> FamReader::getPersons() {
  return persons;
}

} /* namespace FileIO */
} /* namespace CuEira */
