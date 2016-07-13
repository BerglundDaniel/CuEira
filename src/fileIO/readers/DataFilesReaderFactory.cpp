#include "DataFilesReaderFactory.h"

namespace CuEira {
namespace FileIO {

DataFilesReaderFactory::DataFilesReaderFactory(const Configuration& configuration) :
    configuration(configuration){
  PersonHandlerFactory* personHandlerFactory = new PersonHandlerFactory();
  FamReader famReader(configuration, personHandlerFactory);
  PersonHandler* personHandler = famReader.readPersonInformation();

  csvReader.reset(
      new CSVReader(*personHandler, configuration.getCSVFilePath(), configuration.getCSVIdColumnName(),
          configuration.getCSVDelimiter()));
  personHandlerLocked.reset(new PersonHandlerLocked(*personHandler));
  bimReader.reset(new BimReader(configuration));

  delete personHandler;
}

DataFilesReaderFactory::~DataFilesReaderFactory(){

}

DataFilesReader<Container::RegularHostVector>* DataFilesReaderFactory::constructCpuDataFilesReader(){
  Container::SNPVectorFactory<Container::RegularHostVector>* snpVectorFactory = new Container::CPU::CpuSNPVectorFactory(
      configuration);
  BedReader<Container::RegularHostVector>* bedReader = new BedReader<Container::RegularHostVector>(configuration,
      snpVectorFactory, *personHandlerLocked, bimReader->getNumberOfSNPs());

  return new DataFilesReader<Container::RegularHostVector>(personHandlerLocked, bimReader, csvReader, bedReader);
}

#ifndef CPU
DataFilesReader<Container::DeviceVector>* DataFilesReaderFactory::constructCudaDataFilesReader(
    const CUDA::Stream& stream){
  Container::SNPVectorFactory<Container::DeviceVector>* snpVectorFactory = new Container::CUDA::CudaSNPVectorFactory(
      configuration, stream);
  BedReader<Container::DeviceVector>* bedReader = new BedReader<Container::DeviceVector>(configuration,
      snpVectorFactory, *personHandlerLocked, bimReader->getNumberOfSNPs());

  return new DataFilesReader<Container::DeviceVector>(personHandlerLocked, bimReader, csvReader, bedReader);
}
#endif

} /* namespace FileIO */
} /* namespace CuEira */
