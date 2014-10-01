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
  std::cerr << "ResultWriter, time spent waiting at locks: " << boost::chrono::duration_cast<boost::chrono::microseconds>(timeWaitTotalLock)<< std::endl;
#endif
}

void ResultWriter::writeFullResult(const Model::ModelInformation& modelInformation,
    const Model::CombinedResults* combinedResults) {
#ifdef PROFILE
  boost::chrono::system_clock::time_point beforeLock = boost::chrono::system_clock::now();
#endif
  fileLock.lock();
#ifdef PROFILE
  boost::chrono::system_clock::time_point afterLock = boost::chrono::system_clock::now();
  timeWaitTotalLock+=afterLock - beforeLock;
#endif
  outputStream << modelInformation << "," << *combinedResults << std::endl;
  fileLock.unlock();

  delete combinedResults;
}

void ResultWriter::writePartialResult(const Model::ModelInformation& modelInformation) {
#ifdef PROFILE
  boost::chrono::system_clock::time_point beforeLock = boost::chrono::system_clock::now();
#endif
  fileLock.lock();
#ifdef PROFILE
  boost::chrono::system_clock::time_point afterLock = boost::chrono::system_clock::now();
  timeWaitTotalLock+=afterLock - beforeLock;
#endif
  outputStream << modelInformation << std::endl;
  fileLock.unlock();
}

void ResultWriter::printHeader() {
  outputStream << "snp_id,pos,skip,risk_allele,minor,major,env_id,"
      << "no_alleles_minor_case,no_alleles_major_case,no_alleles_minor_control,no_alleles_major_control,no_alleles_minor,no_alleles_major,"
      << "freq_alleles_minor_case,freq_alleles_major_case,freq_alleles_minor_control,freq_alleles_major_control,freq_alleles_minor,freq_alleles_major,"
      << "no_snp0_env0_case,no_snp0_env0_control,no_snp1_env0_case,no_snp1_env0_control,no_snp0_env1_case,no_snp0_env1_control,no_snp1_env1_case,no_snp1_env1_control,"
      << "ap,reri,OR_add_snp,OR_add_snp_L,OR_add_snp_H,OR_add_env,OR_env_add_L,OR_env_add_H,OR_add_inter,OR_add_inter_L,OR_add_inter_H,";

  outputStream
      << "OR_mult_snp,OR_mult_snp_L,OR_mult_snp_H,OR_mult_env,OR_mult_env_L,OR_mult_env_H,OR_mult_inter,OR_mult_inter_L,OR_mult_inter_H,";

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
