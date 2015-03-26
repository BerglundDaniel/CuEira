#include "DataFilesReader.h"

namespace CuEira {
namespace FileIO {

DataFilesReader::DataFilesReader(CovariatesHandlerFactory* covariatesHandlerFactory, PersonHandler* personHandler,
    BedReader* bedReader, BimReader* bimReader, EnvironmentCSVReader* environmentCSVReader,
    CSVReader* covariateCSVReader) :
    personHandler(personHandler), bedReader(bedReader), bimReader(bimReader), environmentCSVReader(
        environmentCSVReader), covariateCSVReader(covariateCSVReader), useCovariates(true), covariatesHandlerFactory(
        covariatesHandlerFactory) {
  personHandler->lockIndividuals();
}

DataFilesReader::DataFilesReader(PersonHandler* personHandler, BedReader* bedReader, BimReader* bimReader,
    EnvironmentCSVReader* environmentCSVReader) :
    personHandler(personHandler), bedReader(bedReader), bimReader(bimReader), environmentCSVReader(
        environmentCSVReader), covariateCSVReader(nullptr), useCovariates(false), covariatesHandlerFactory(nullptr) {
  personHandler->lockIndividuals();
}

DataFilesReader::~DataFilesReader() {
  delete bedReader;
  delete bimReader;
  delete environmentCSVReader;
  delete covariateCSVReader;
  delete personHandler;
  delete covariatesHandlerFactory;
}

CovariatesHandler* DataFilesReader::readCovariates() const {
  if(!useCovariates){
    std::ostringstream os;
    os << "Can't get read covariates since no covariate file was specified." << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
  return covariatesHandlerFactory->constructCovariatesHandler(covariateCSVReader->readData());
}

std::vector<SNP*>* DataFilesReader::readSNPInformation() const {
  return bimReader->readSNPInformation();
}

EnvironmentFactorHandler* DataFilesReader::readEnvironmentFactorInformation() const {
  return environmentCSVReader->readEnvironmentFactorInformation();
}

Container::SNPVector* DataFilesReader::readSNP(SNP& snp) {
  return bedReader->readSNP(snp);
}

const PersonHandler& DataFilesReader::getPersonHandler() const {
  return *personHandler;
}

} /* namespace FileIO */
} /* namespace CuEira */
