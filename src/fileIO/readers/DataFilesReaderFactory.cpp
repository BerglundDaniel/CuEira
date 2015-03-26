#include "DataFilesReaderFactory.h"

namespace CuEira {
namespace FileIO {

DataFilesReaderFactory::DataFilesReaderFactory() {

}

DataFilesReaderFactory::~DataFilesReaderFactory() {

}

DataFilesReader* DataFilesReaderFactory::constructDataFilesReader(Configuration& configuration) {
  Container::SNPVectorFactory* snpVectorFactory = new SNPVectorFactory(configuration);

  FamReader famReader(configuration);
  PersonHandler* personHandler = famReader.readPersonInformation();

  BimReader* bimReader = new BimReader(configuration);
  BedReader* bedReader = new BedReader(configuration, snpVectorFactory, *personHandler, bimReader->getNumberOfSNPs());

  EnvironmentCSVReader* environmentCSVReader = new EnvironmentCSVReader(*personHandler,
      configuration.getEnvironmentFilePath(), configuration.getEnvironmentIndividualIdColumnName(),
      configuration.getEnvironmentDelimiter());

  if(configuration.covariateFileSpecified()){
    CovariatesHandlerFactory covariatesHandlerFactory = new CovariatesHandlerFactory();
    CSVReader* covariateCSVReader = new CSVReader(*personHandler, configuration.getCovariateFilePath(),
        configuration.getCovariateIndividualIdColumnName(), configuration.getCovariateDelimiter());

    return new DataFilesReader(covariatesHandlerFactory, personHandler, bedReader, bimReader, environmentCSVReader,
        covariateCSVReader);
  }else{
    return new DataFilesReader(personHandler, bedReader, bimReader, environmentCSVReader);
  }
}

} /* namespace FileIO */
} /* namespace CuEira */
