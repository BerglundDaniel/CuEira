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

  CSVReader environmentCSVReader(configuration.getEnvironmentFilePath(),
      configuration.getEnvironmentIndividualIdColumnName(), configuration.getEnvironmentDelimiter(), personHandler);

  CSVReader covariateCSVReader(configuration.getCovariateFilePath(), configuration.getCovariateIndividualIdColumnName(),
      configuration.getCovariateDelimiter(), personHandler);

  return new DataFilesReader(plinkReader, environmentCSVReader, covariateCSVReader);
}

} /* namespace FileIO */
} /* namespace CuEira */
