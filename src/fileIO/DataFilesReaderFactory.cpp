#include "DataFilesReaderFactory.h"

namespace CuEira {
namespace FileIO {

DataFilesReaderFactory::DataFilesReaderFactory(PlinkReaderFactory& plinkReaderFactory) :
    plinkReaderFactory(plinkReaderFactory) {

}

DataFilesReaderFactory::~DataFilesReaderFactory() {

}

DataFilesReader* DataFilesReaderFactory::constructDataFilesReader(Configuration& configuration) {
  PlinkReader* plinkReader = plinkReaderFactory.constructPlinkReader(configuration);
  const PersonHandler& personHandler = plinkReader->getPersonHandler();

  EnvironmentCSVReader* environmentCSVReader= new EnvironmentCSVReader(configuration.getEnvironmentFilePath(),
      configuration.getEnvironmentIndividualIdColumnName(), configuration.getEnvironmentDelimiter(), personHandler);
  CSVReader* covariateCSVReader= new CSVReader(configuration.getCovariateFilePath(), configuration.getCovariateIndividualIdColumnName(),
      configuration.getCovariateDelimiter(), personHandler);

  return new DataFilesReader(plinkReader, environmentCSVReader, covariateCSVReader);
}

} /* namespace FileIO */
} /* namespace CuEira */
