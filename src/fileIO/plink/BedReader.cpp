#include "BedReader.h"

namespace CuEira {
namespace FileIO {

BedReader::BedReader(Configuration& configuration) :
    configuration(configuration), bedFileStr(configuration.getBedFilePath()) {

  try{
    bedFile.open(bedFileStr, std::ifstream::binary);
    int bufferSize = 3;
    char buffer[bufferSize];
    bedFile.seekg(0, std::ios::beg);
    bedFile.read(buffer, bufferSize);

    //Check version
    if(true){
      //01101100
      //00011011

    }else{
      std::ostringstream os;
      os << "Provided bed file is not a bed file " << bedFileStr << std::endl;
      const std::string& tmp = os.str();
      throw FileReaderException(tmp.c_str());
    }

    //Check mode
    if(buffer[2]==1){
      //00000001
      mode = SNPMAJOR;
    }else if(buffer[2]==0){
      // 00000000
      mode = INDIVIDUALMAJOR;
    }else{
      std::ostringstream os;
      os << "Unknown major mode of the bed file " << bedFileStr << std::endl;
      const std::string& tmp = os.str();
      throw FileReaderException(tmp.c_str());
    }

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
    os << "Problem reading SNP " << snp.getId().getString() << " from bed file " << bedFileStr << std::endl;
#ifdef DEBUG
    os << exception.what();
#endif
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
}

} /* namespace FileIO */
} /* namespace CuEira */
