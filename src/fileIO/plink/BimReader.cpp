#include "BimReader.h"

namespace CuEira {
namespace FileIO {

BimReader::BimReader(Configuration& configuration) :
    configuration(configuration) {
  std::ifstream bimFile(configuration.getBimFilePath());
  std::string line;
  while(std::getline(bimFile, line)){
    std::vector<std::string> lineSplit;
    boost::split(lineSplit, line, boost::is_any_of("\t "));

    std::cerr << lineSplit[1] << std::endl;
  }

}

BimReader::~BimReader() {

}

} /* namespace FileIO */
} /* namespace CuEira */
