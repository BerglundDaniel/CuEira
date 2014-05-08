#include "DataFilesReader.h"

namespace CuEira {
namespace FileIO {

DataFilesReader::DataFilesReader(PlinkReader* plinkReader, const CSVReader& environmentCSVReader,
    const CSVReader& covariateCSVReader) :
    plinkReader(plinkReader), environmentCSVReader(environmentCSVReader), covariateCSVReader(covariateCSVReader) {

}

DataFilesReader::~DataFilesReader() {
  delete plinkReader;
}

Container::HostVector* DataFilesReader::readSNP(SNP& snp) const {
  return plinkReader->readSNP(snp);
}

const Container::HostVector& DataFilesReader::getEnvironmentFactor(EnvironmentFactor& environmentFactor) const {
  return environmentCSVReader.getData(environmentFactor.getId().getString());
}

const Container::HostMatrix& DataFilesReader::getCovariates() const {
  return covariateCSVReader.getData();
}

int DataFilesReader::getNumberOfCovariates() const {
  return covariateCSVReader.getNumberOfColumns();
}

int DataFilesReader::getNumberOfEnvironmentFactors() const {
  return environmentCSVReader.getNumberOfColumns();
}

const PersonHandler& DataFilesReader::getPersonHandler() const {
  return plinkReader->getPersonHandler();
}

} /* namespace FileIO */
} /* namespace CuEira */
