#include "DataFilesReaderFactory.h"

namespace CuEira {
namespace FileIO {

DataFilesReaderFactory::DataFilesReaderFactory(const Configuration& configuration) :
    configuration(configuration){
  FamReader famReader(configuration);
  PersonHandler* personHandler = famReader.readPersonInformation();

  csvReader(
      new CSVReader(*personHandler, configuration.getCSVFilePath(), configuration.getCSVIdColumnName(),
          configuration.getCSVDelimiter()));
  personHandlerLocked(new PersonHandlerLocked(personHandler));
  bimReader(new BimReader(configuration));

  delete personHandler;
}

DataFilesReaderFactory::~DataFilesReaderFactory(){

}

template<typename Vector>
DataFilesReader<Vector>* DataFilesReaderFactory::constructDataFilesReader(){

  Container::SNPVectorFactory<Vector>* snpVectorFactory = new SNPVectorFactory<Vector>(configuration); //TODO template, CPU/GPU
  BedReader<Vector>* bedReader = new BedReader<Vector>(configuration, snpVectorFactory, *personHandlerLocked,
      bimReader->getNumberOfSNPs());

  return new DataFilesReader<Vector>(personHandlerLocked, bimReader, csvReader, bedReader);
}

} /* namespace FileIO */
} /* namespace CuEira */
