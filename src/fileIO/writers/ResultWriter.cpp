#include "ResultWriter.h"

namespace CuEira {
namespace FileIO {

ResultWriter::ResultWriter(const Configuration& configuration) :
    configuration(configuration), outputFileName(configuration.getOutputFilePath()) {
  printHeader();
}

ResultWriter::~ResultWriter() {

}

void ResultWriter::writeFullResult(const SNP& snp, const EnvironmentFactor& environmentFactor,
    const Statistics& statistics, const Container::SNPVector& snpVector) {
  openFile();

  //TODO need to add allele freqs and such
  outputStream << snp << "," << environmentFactor << "," << statistics << "," << snpVector << std::endl;

  closeFile();
}

void ResultWriter::writePartialResult(const SNP& snp, const EnvironmentFactor& environmentFactor) {
  openFile();

  //TODO need to add allele freqs and such
  //TODO add reason for skip
  outputStream << snp << "," << environmentFactor << std::endl;

  closeFile();
}

void ResultWriter::printHeader() {
  openFile();

  //FIXME
  outputStream
      << "snp_id,pos,risk_allele,minor,major,env_id,ap,reri,OR_snp,OR_snp_L,OR_snp_H,OR_env,OR_env_L,OR_env_H,OR_inter,OR_inter_L,OR_inter_H,";

  /*
   for(int i = 0; i < numberOfCovariates; ++i){
   outputStream << (*covariatesNames)[i] << "_cov_OR," << (*covariatesNames)[i] << "_cov_OR_L,"
   << (*covariatesNames)[i] << "_cov_OR_H,";
   }*/

  outputStream << "recode" << std::endl;

  closeFile();
}

void ResultWriter::openFile() {
  fileLock.lock();
  outputStream.open(outputFileName, std::ofstream::ios_base::app);
  if(!outputStream){
    fileLock.unlock();
    std::ostringstream os;
    os << "Problem opening output file " << outputFileName << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
}

void ResultWriter::closeFile() {
  if(outputStream.is_open()){
    outputStream.close();
    fileLock.unlock();
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
