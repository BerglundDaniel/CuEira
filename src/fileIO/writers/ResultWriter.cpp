#include "ResultWriter.h"

namespace CuEira {
namespace FileIO {

ResultWriter::ResultWriter(const Configuration& configuration) :
    configuration(configuration), outputFileName(configuration.getOutputFilePath()) {
  openFile();
  printHeader();
}

ResultWriter::~ResultWriter() {
  closeFile();
}

void ResultWriter::writeFullResult(const Model::ModelInformation* modelInformation,
    const Model::CombinedResults* combinedResults) {
  fileLock.lock();
  outputStream << *modelInformation << "," << *combinedResults << std::endl;
  fileLock.unlock();

  delete modelInformation;
  delete combinedResults;
}

void ResultWriter::writePartialResult(const Model::ModelInformation* modelInformation) {
  fileLock.lock();
  outputStream << *modelInformation << std::endl;
  fileLock.unlock();

  delete modelInformation;
}

void ResultWriter::printHeader() {
  outputStream
      << "snp_id,pos,skip,risk_allele,minor,major,env_id,ap,reri,OR_snp,OR_snp_L,OR_snp_H,OR_env,OR_env_L,OR_env_H,OR_inter,OR_inter_L,OR_inter_H,";

  /*
   for(int i = 0; i < numberOfCovariates; ++i){
   outputStream << (*covariatesNames)[i] << "_cov_OR," << (*covariatesNames)[i] << "_cov_OR_L,"
   << (*covariatesNames)[i] << "_cov_OR_H,";
   }*/

  outputStream << "recode" << std::endl;
}

void ResultWriter::openFile() {
  outputStream.open(outputFileName, std::ofstream::ios_base::app);
  if(!outputStream){
    std::ostringstream os;
    os << "Problem opening output file " << outputFileName << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
}

void ResultWriter::closeFile() {
  if(outputStream.is_open()){
    outputStream.close();
  }
  if(!outputStream){
    std::ostringstream os;
    os << "Problem closing output file " << outputFileName << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
}

} /* namespace FileIO */
} /* namespace CuEira */
