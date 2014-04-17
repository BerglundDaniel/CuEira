#include "DataFilesReaderFactory.h"

namespace CuEira {
namespace FileIO {

DataFilesReaderFactory::DataFilesReaderFactory(PlinkReaderFactory& plinkReaderFactory) :
    plinkReaderFactory(plinkReaderFactory) {

}

DataFilesReaderFactory::~DataFilesReaderFactory() {

}

DataFilesReader* DataFilesReaderFactory::constructDataFilesReader(Configuration& configuration) {
  PlinkReader plinkReader = plinkReaderFactory.constructPlinkReader(configuration);
  int numberOfIndividuals = plinkReader.getNumberOfIndividuals();
  std::map<Id, Person> idToPersonMap = plinkReader.getIdToPersonMap();

  CSVReader environmentCSVReader = CSVReader(configuration.getEnvironmentFilePath(),
      configuration.getEnvironmentIndividualIdColumnName(), idToPersonMap);
  CSVReader covariateCSVReader = CSVREader(configuration.getCovariateFilePath(),
      configuration.getCovariateIndividualIdColumnName(), idToPersonMap);

  if(environmentCSVReader.getNumberOfRows() != numberOfIndividuals){
    throw DimensionMismatch("Not the same number of individuals in the environment file as the Plink Files.");
  }

  if(covariateCSVReader.getNumberOfRows() != numberOfIndividuals){
    throw DimensionMismatch("Not the same number of individuals in the covariates file as the Plink Files.");
  }

  return new DataFilesReader(plinkReader, environmentCSVReader, covariateCSVReader);
}

} /* namespace FileIO */
} /* namespace CuEira */
