#include <iostream>
#include <stdexcept>
#include <vector>
#include <thread>

#include <Configuration.h>
#include <DataFilesReaderFactory.h>
#include <DataFilesReader.h>
#include <SNPVector.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <PersonHandler.h>
#include <EnvironmentFactor.h>
#include <Statistics.h>
#include <Recode.h>
#include <DataHandler.h>
#include <ModelHandler.h>
#include <EnvironmentVector.h>
#include <InteractionVector.h>
#include <StatisticsFactory.h>
#include <DataHandlerState.h>
#include <ContingencyTableFactory.h>
#include <AlleleStatisticsFactory.h>
#include <DataHandlerFactory.h>
#include <ResultWriter.h>

#ifdef CPU
//#include <CpuModelHandler.h>
#else
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <HostToDevice.h>
#include <DeviceToHost.h>
#include <CudaAdapter.cu>
#include <GpuModelHandler.h>
#include <LogisticRegressionConfiguration.h>
#include <CudaException.h>
#include <GPUWorkerThread.h>
#include <Device.h>
#include <CudaException.h>
#include <Stream.h>
#include <StreamFactory.h>
#endif

/**
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
int main(int argc, char* argv[]) {
  using namespace CuEira;
  std::cerr << "Starting" << std::endl;
#ifdef DEBUG
  std::cerr << "WARNING CuEira was compiled in debug mode, this can affect performance." << std::endl;
#endif

  Configuration configuration(argc, argv);
  FileIO::DataFilesReaderFactory dataFilesReaderFactory;
  Container::SNPVectorFactory snpVectorFactory(configuration);
  AlleleStatisticsFactory alleleStatisticsFactory;

  FileIO::DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configuration);
  FileIO::ResultWriter* resultWriter = new FileIO::ResultWriter(configuration);

  PersonHandler* personHandler = dataFilesReader->readPersonInformation();
  const int numberOfIndividualsToInclude = personHandler->getNumberOfIndividualsToInclude();
  const Container::HostVector& outcomes = personHandler->getOutcomes();

#ifndef CPU
  int numberOfDevices = -1;
  const int numberOfStreams = configuration.getNumberOfStreams();
  const int numberOfThreads = numberOfDevices * numberOfStreams;
  cudaGetDeviceCount(&numberOfDevices);
  CUDA::StreamFactory* streamFactory = new CUDA::StreamFactory();

  if(numberOfDevices == 0){
    throw new CudaException("No cuda devices found.");
  }
  std::cerr << "Calculating using " << numberOfDevices << " with " << numberOfStreams << " each." << std::endl;

  std::vector<CUDA::Device*> devices(numberOfDevices);
  std::vector<std::thread*> workers(numberOfThreads);
  std::vector<CUDA::Stream*>* outcomeTransferStreams = new std::vector<CUDA::Stream*>(numberOfDevices);

  for(int deviceNumber = 0; deviceNumber < numberOfDevices; ++deviceNumber){
    CUDA::Device* device = new CUDA::Device(deviceNumber);
    devices[deviceNumber] = device;

    device->setActiveDevice();
    CUDA::Stream* stream = streamFactory->constructStream(*device);
    (*outcomeTransferStreams)[deviceNumber] = stream;

    CUDA::HostToDevice hostToDevice(stream->getCudaStream());

    device->setOutcomes(hostToDevice.transferVector(&outcomes));
  }
#endif

  EnvironmentFactorHandler* environmentFactorHandler = dataFilesReader->readEnvironmentFactorInformation(
      *personHandler);
  std::vector<SNP*>* snpInformation = dataFilesReader->readSNPInformation();

  const int numberOfSNPs = snpInformation->size();

  FileIO::BedReader* bedReader = new FileIO::BedReader(configuration, snpVectorFactory, alleleStatisticsFactory,
      *personHandler, numberOfSNPs);
  Task::DataQueue* dataQueue = new Task::DataQueue(snpInformation);

  ContingencyTableFactory contingencyTableFactory(outcomes);
  DataHandlerFactory* dataHandlerFactory = new DataHandlerFactory(configuration, contingencyTableFactory, *bedReader,
      *environmentFactorHandler, *dataQueue);

  Container::HostMatrix* covariates = nullptr;
  std::vector<std::string>* covariatesNames = nullptr;
  if(configuration.covariateFileSpecified()){
    std::pair<Container::HostMatrix*, std::vector<std::string>*>* covPair = dataFilesReader->readCovariates(
        *personHandler);
    covariates = covPair->first;
    covariatesNames = covPair->second;
    delete covPair;

  }

#ifdef CPU
  //TODO
  //Model::ModelHandler* modelHandler = new Model::CpuModelHandler();
#else
  //GPU
  for(int deviceNumber = 0; deviceNumber < numberOfDevices; ++deviceNumber){
    CUDA::Device* device = devices[deviceNumber];
    device->setActiveDevice();
    CUDA::Stream* outcomeTransfeStream = (*outcomeTransferStreams)[deviceNumber];
    outcomeTransfeStream->syncStream();
    delete outcomeTransfeStream;

    //Start threads
    for(int streamNumber = 0; streamNumber < numberOfStreams; ++streamNumber){
      //TODO fix covariates
      std::thread* thread = new std::thread(CuEira::CUDA::GPUWorkerThread, &configuration, device, dataHandlerFactory,
          resultWriter);
      workers[deviceNumber * numberOfStreams + streamNumber] = thread;
    }
  }
  delete outcomeTransferStreams;

  for(int threadNumber = 0; threadNumber < numberOfThreads; ++threadNumber){
    workers[threadNumber]->join();
  }

  for(int threadNumber = 0; threadNumber < numberOfThreads; ++threadNumber){
    delete workers[threadNumber];
  }

  for(int deviceNumber = 0; deviceNumber < numberOfDevices; ++deviceNumber){
    delete devices[deviceNumber];
  }

#endif

#ifndef CPU
  delete streamFactory;
#endif

  delete resultWriter;
  delete bedReader;
  delete dataFilesReader;
  delete personHandler;
  delete environmentFactorHandler;
  delete dataQueue;
  delete covariates;
  delete covariatesNames;
  delete dataHandlerFactory;

  std::cerr << "Done" << std::endl;

}
