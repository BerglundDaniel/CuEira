#include "BimReader.h"

namespace CuEira {
namespace FileIO {

BimReader::BimReader(Configuration& configuration) :
    configuration(configuration), numberOfSNPs(0) {
  std::ifstream bimFile(configuration.getBimFilePath().c_str(),std::ifstream::in);
  std::string line;

  while(std::getline(bimFile, line)){
    numberOfSNPs++;
    std::vector<std::string> lineSplit;
    boost::split(lineSplit, line, boost::is_any_of("\t "));

    std::cerr << lineSplit[1] << std::endl; //Read snp name, column 2
    std::cerr << lineSplit[3] << std::endl; //Read position, column 4, if negative, set include to false

  }

}

BimReader::~BimReader() {

}

int BimReader::getNumberOfSNPs() {
  return numberOfSNPs;
}

} /* namespace FileIO */
} /* namespace CuEira */
