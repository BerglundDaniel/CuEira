#include "BedReader.h"

namespace CuEira {
namespace FileIO {

BedReader::BedReader(Configuration& configuration) :
    configuration(configuration), bedFileStr(configuration.getBedFilePath()) {

  try{
    bedFile.open(bedFileStr, std::ifstream::binary);

    //Check version TODO

    //Check mode TODO

  } catch(const std::ios_base::failure& exception){
    std::ostringstream os;
    os << "Problem opening and reading header of bed file " << bedFileStr << std::endl;
#ifdef DEBUG
    os << exception.what();
#endif
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
}

BedReader::~BedReader() {
  try{
    bedFile.close();
  } catch(const std::ios_base::failure& exception){
    std::ostringstream os;
    os << "Problem closing bed file " << bedFileStr << std::endl;
#ifdef DEBUG
    os << exception.what();
#endif
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
}

Container::HostVector BedReader::readSNP(SNP& snp) {
  try{
    //Read SNP TODO

  } catch(const std::ios_base::failure& exception){
    std::ostringstream os;
    os << "Problem reading SNP " << snp.getId() << " from bed file " << bedFileStr << std::endl;
#ifdef DEBUG
    os << exception.what();
#endif
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
}

} /* namespace FileIO */
} /* namespace CuEira */
