#include "DataFilesReader.h"

namespace CuEira {
namespace FileIO {

DataFilesReader::DataFilesReader(PlinkReader* plinkReader, EnvironmentCSVReader* environmentCSVReader,
    CSVReader* covariateCSVReader) :
    plinkReader(plinkReader), environmentCSVReader(environmentCSVReader), covariateCSVReader(covariateCSVReader) {

}

DataFilesReader::~DataFilesReader() {
  delete plinkReader;
  delete environmentCSVReader;
  delete covariateCSVReader;
}

Container::SNPVector* DataFilesReader::readSNP(SNP& snp) const {
  return plinkReader->readSNP(snp);
}

const Container::HostVector& DataFilesReader::getEnvironmentFactor(EnvironmentFactor& environmentFactor) const {
  return environmentCSVReader->getData(environmentFactor);
}

const Container::HostMatrix& DataFilesReader::getCovariates() const {
  return covariateCSVReader->getData();
}

int DataFilesReader::getNumberOfCovariates() const {
  return covariateCSVReader->getNumberOfColumns();
}

int DataFilesReader::getNumberOfEnvironmentFactors() const {
  return environmentCSVReader->getNumberOfColumns();
}

const PersonHandler& DataFilesReader::getPersonHandler() const {
  return plinkReader->getPersonHandler();
}

std::vector<SNP*> DataFilesReader::getSNPInformation() const {
  return plinkReader->getSNPInformation();
}

const std::vector<EnvironmentFactor*>& DataFilesReader::getEnvironmentFactorInformation() const {
  return environmentCSVReader->getEnvironmentFactorInformation();
}

} /* namespace FileIO */
} /* namespace CuEira */
