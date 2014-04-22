#include "BimReader.h"

namespace CuEira {
namespace FileIO {

BimReader::BimReader(Configuration& configuration) :
    configuration(configuration), numberOfSNPs(0) {
  std::string bimFileStr = configuration.getBimFilePath();
  std::string line;
  std::ifstream bimFile;

  try{
    bimFile.open(bimFileStr, std::ifstream::in);
    while(std::getline(bimFile, line)){
      numberOfSNPs++;
    }
    bimFile.close();
  } catch(const std::ios_base::failure& exception){
    std::ostringstream os;
    os << "Problem reading bim file " << bimFileStr << std::endl;
#ifdef DEBUG
    os << exception.what();
#endif
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }

  std::vector<SNP*> SNPVector = std::vector<SNP*>(numberOfSNPs);
  int pos = 0;
  SNP* snp = nullptr;
  long int baseSNP;
  char * temp; //Used for error checking of string to long int conversion (strtol)

  try{
    bimFile.open(bimFileStr, std::ifstream::in);
    while(std::getline(bimFile, line)){
      std::vector<std::string> lineSplit;
      boost::split(lineSplit, line, boost::is_any_of("\t "));

      Id id = Id(lineSplit[1]);

      baseSNP = strtol(lineSplit[3].c_str(), &temp, 0);
      if(*temp != '\0'){ //Check if there was an error with strtol
        throw FileReaderException("Problem with string to int conversion of basepositions in bim file");
      }

      if(baseSNP < 0){
        snp = new SNP(id, false);
      }else{
        snp = new SNP(id, false);
      }
      SNPVector[pos] = snp;

      //Read allels?

      pos++;
    } /* while getline */
    bimFile.close();
  } catch(const std::ios_base::failure& exception){
    std::ostringstream os;
    os << "Problem reading bim file " << bimFileStr << std::endl;
#ifdef DEBUG
    os << exception.what();
#endif
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }

}

BimReader::~BimReader() {

}

int BimReader::getNumberOfSNPs() {
  return numberOfSNPs;
}

std::vector<SNP*> BimReader::getSNPs() {
  return SNPVector;
}

} /* namespace FileIO */
} /* namespace CuEira */
