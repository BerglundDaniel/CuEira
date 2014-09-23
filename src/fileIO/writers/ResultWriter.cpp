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

#ifdef PROFILE
  std::cerr << "Time spent waiting at locks: " << timeWaitTotalLock << " seconds" << std::endl;
#endif
}

void ResultWriter::writeFullResult(const Model::ModelInformation* modelInformation,
    const Model::CombinedResults* combinedResults) {
#ifdef PROFILE
  boost::chrono::system_clock::time_point beforeLock = boost::chrono::system_clock::now();
#endif
  fileLock.lock();
#ifdef PROFILE
  boost::chrono::system_clock::time_point afterLock = boost::chrono::system_clock::now();
  timeWaitTotalLock+=afterLock - beforeLock;
#endif
  outputStream << *modelInformation << "," << *combinedResults << std::endl;
  fileLock.unlock();

  delete modelInformation;
  delete combinedResults;
}

void ResultWriter::writePartialResult(const Model::ModelInformation* modelInformation) {
#ifdef PROFILE
  boost::chrono::system_clock::time_point beforeLock = boost::chrono::system_clock::now();
#endif
  fileLock.lock();
#ifdef PROFILE
  boost::chrono::system_clock::time_point afterLock = boost::chrono::system_clock::now();
  timeWaitTotalLock+=afterLock - beforeLock;
#endif
  outputStream << *modelInformation << std::endl;
  fileLock.unlock();

  delete modelInformation;
}

void ResultWriter::printHeader() {
  outputStream << "snp_id,pos,skip,risk_allele,minor,major,env_id,"
      << "no_alleles_minor_case,no_alleles_major_case,no_alleles_minor_control,no_alleles_major_control,no_alleles_minor,no_alleles_major,"
      << "freq_alleles_minor_case,freq_alleles_major_case,freq_alleles_minor_control,freq_alleles_major_control,freq_alleles_minor,freq_alleles_major,"
      << "no_snp0_env0_case,no_snp0_env0_control,no_snp1_env0_case,no_snp1_env0_control,no_snp0_env1_case,no_snp0_env1_control,no_snp1_env1_case,no_snp1_env1_control,"
      << "ap,reri,OR_snp,OR_snp_L,OR_snp_H,OR_env,OR_env_L,OR_env_H,OR_inter,OR_inter_L,OR_inter_H,";

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
