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

  CSVReader* csvReader = new CSVReader(*personHandler, configuration.getCSVFilePath(),
      configuration.getCSVIdColumnName(), configuration.getCSVDelimiter());

  return new DataFilesReader(personHandler, bedReader, bimReader, csvReader);

}

} /* namespace FileIO */
} /* namespace CuEira */
