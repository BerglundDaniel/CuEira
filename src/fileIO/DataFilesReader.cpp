#include "DataFilesReader.h"

namespace CuEira {
namespace FileIO {

DataFilesReader::DataFilesReader(BimReader* bimReader, FamReader* famReader, EnvironmentCSVReader* environmentCSVReader,
    CSVReader* covariateCSVReader) :
    famReader(famReader), bimReader(bimReader), environmentCSVReader(environmentCSVReader), covariateCSVReader(
        covariateCSVReader) {

}

DataFilesReader::~DataFilesReader() {
  delete bimReader;
  delete famReader;
  delete environmentCSVReader;
  delete covariateCSVReader;
}

std::pair<Container::HostMatrix*, std::vector<std::string>*>* DataFilesReader::readCovariates(
    const PersonHandler& personHandler) const {
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
