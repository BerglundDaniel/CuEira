#include "DataFilesReader.h"

namespace CuEira {
namespace FileIO {

DataFilesReader::DataFilesReader(PlinkReader& plinkReader, CSVReader& environmentCSVReader,
    CSVReader& covariateCSVReader) :
    plinkReader(plinkReader), environmentCSVReader(environmentCSVReader), covariateCSVReader(covariateCSVReader) {

}

DataFilesReader::~DataFilesReader() {

}

Container::HostVector DataFilesReader::readSNP(SNP& snp) {
  return plinkReader.readSNP(snpid);
}

Container::HostVector DataFilesReader::getEnvironmentFactor(EnvironmentFactor& environmentFactor) {
  return environmentCSVReader.getData(environmentFactor);
}

Container::HostMatrix DataFilesReader::getCovariates() {
  return covariateCSVReader.getData();
}

Container::HostVector DataFilesReader::getOutcomes() {
  return plinkReader.getOutcomes();
}

int DataFilesReader::getNumberOfIndividuals() {
  return plinkReader.getNumberOfIndividuals;
}

int DataFilesReader::getNumberOfCovariates() {
  return covariateCSVReader.getNumberOfColumns();
}

int DataFilesReader::getNumberOfEnvironmentFactors() {
  return environmentCSVReader.getNumberOfColumns();
}

} /* namespace FileIO */
} /* namespace CuEira */
