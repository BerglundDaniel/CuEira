#include "DataFilesReader.h"

namespace CuEira {
namespace FileIO {

DataFilesReader::DataFilesReader(BimReader* bimReader, FamReader* famReader, EnvironmentCSVReader* environmentCSVReader,
    CSVReader* covariateCSVReader) :
    famReader(famReader), bimReader(bimReader), environmentCSVReader(environmentCSVReader), covariateCSVReader(
        covariateCSVReader), useCovariates(true) {

}

DataFilesReader::DataFilesReader(BimReader* bimReader, FamReader* famReader, EnvironmentCSVReader* environmentCSVReader) :
    famReader(famReader), bimReader(bimReader), environmentCSVReader(environmentCSVReader), covariateCSVReader(nullptr), useCovariates(
        false) {

}

DataFilesReader::~DataFilesReader() {
  delete bimReader;
  delete famReader;
  delete environmentCSVReader;
  delete covariateCSVReader;
}

std::pair<Container::HostMatrix*, std::vector<std::string>*>* DataFilesReader::readCovariates(
    const PersonHandler& personHandler) const {
  if(!useCovariates){
    std::ostringstream os;
    os << "Can't get read covariates since no covariate file was specified." << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
  return covariateCSVReader->readData(personHandler);
}

PersonHandler* DataFilesReader::readPersonInformation() const {
  return famReader->readPersonInformation();
}

std::vector<SNP*>* DataFilesReader::readSNPInformation() const {
  return bimReader->readSNPInformation();
}

EnvironmentFactorHandler* DataFilesReader::readEnvironmentFactorInformation(const PersonHandler& personHandler) const {
  return environmentCSVReader->readEnvironmentFactorInformation(personHandler);
}

} /* namespace FileIO */
} /* namespace CuEira */
