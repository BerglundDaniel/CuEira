#include "DataFilesReaderFactory.h"

namespace CuEira {
namespace FileIO {

DataFilesReaderFactory::DataFilesReaderFactory() {

}

DataFilesReaderFactory::~DataFilesReaderFactory() {

}

DataFilesReader* DataFilesReaderFactory::constructDataFilesReader(Configuration& configuration) {
  BimReader* bimReader = new BimReader(configuration);
  FamReader* famReader = new FamReader(configuration);

  EnvironmentCSVReader* environmentCSVReader = new EnvironmentCSVReader(configuration.getEnvironmentFilePath(),
      configuration.getEnvironmentIndividualIdColumnName(), configuration.getEnvironmentDelimiter());
  CSVReader* covariateCSVReader = new CSVReader(configuration.getCovariateFilePath(),
      configuration.getCovariateIndividualIdColumnName(), configuration.getCovariateDelimiter());

  return new DataFilesReader(bimReader, famReader, environmentCSVReader, covariateCSVReader);
}

} /* namespace FileIO */
} /* namespace CuEira */
